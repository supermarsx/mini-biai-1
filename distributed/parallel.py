import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import copy
import math
from collections import defaultdict

logger = logging.getLogger(__name__)

class TensorParallelLinear(nn.Module):
    """Tensor parallel linear layer for distributed inference.
    
    This layer splits the weight matrix across multiple devices/tensors
    to enable parallel computation and reduce memory usage.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tensor_parallel_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        bias: bool = True,
        use_cpu: bool = False
    ):
        """Initialize tensor parallel linear layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            tensor_parallel_size: Number of tensors to parallelize across
            device: Device to place the layer on
            dtype: Data type for weights and computations
            bias: Whether to include bias term
            use_cpu: Whether to use CPU for parameter storage
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_parallel_size = tensor_parallel_size
        self.use_cpu = use_cpu
        
        # Validate dimensions
        if out_features % tensor_parallel_size != 0:
            raise ValueError(f"out_features {out_features} must be divisible by tensor_parallel_size {tensor_parallel_size}")
        
        self.out_features_per_partition = out_features // tensor_parallel_size
        
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Initialize weights
        weight_shape = (self.out_features_per_partition, in_features)
        bias_shape = (self.out_features_per_partition,)
        
        if use_cpu:
            # Store on CPU to reduce GPU memory usage
            self.weight = nn.Parameter(torch.empty(weight_shape, dtype=dtype, device='cpu'))
            if bias:
                self.bias = nn.Parameter(torch.empty(bias_shape, dtype=dtype, device='cpu'))
            else:
                self.register_parameter('bias', None)
        else:
            # Standard initialization on device
            self.weight = nn.Parameter(torch.empty(weight_shape, dtype=dtype, device=device))
            if bias:
                self.bias = nn.Parameter(torch.empty(bias_shape, dtype=dtype, device=device))
            else:
                self.register_parameter('bias', None)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize tensor parallel weights."""
        # Xavier initialization scaled by tensor parallel size
        fan_in = self.in_features
        std = math.sqrt(2.0 / fan_in) / math.sqrt(self.tensor_parallel_size)
        
        with torch.no_grad():
            self.weight.normal_(0, std)
            if self.bias is not None:
                self.bias.zero_()
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with tensor parallel computation.
        
        Args:
            input: Input tensor of shape (batch_size, in_features)
        
        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # Move input to device if needed
        if input.device != self.device:
            input = input.to(self.device)
        
        # Compute partial output
        if self.use_cpu:
            # Move weights to device for computation
            partial_output = F.linear(input.to(self.weight.device), self.weight)
            if self.bias is not None:
                partial_output += self.bias.to(partial_output.device)
        else:
            # Standard computation
            partial_output = F.linear(input, self.weight)
            if self.bias is not None:
                partial_output += self.bias
        
        return partial_output
    
    def all_gather_output(self, partial_output: torch.Tensor) -> torch.Tensor:
        """Gather outputs from all tensor parallel processes.
        
        Args:
            partial_output: Output from this tensor's computation
        
        Returns:
            Combined output from all tensors
        """
        if not dist.is_initialized():
            return partial_output
        
        # Gather outputs from all processes
        gathered_outputs = [torch.zeros_like(partial_output) for _ in range(self.tensor_parallel_size)]
        dist.all_gather(gathered_outputs, partial_output)
        
        # Concatenate along output dimension
        return torch.cat(gathered_outputs, dim=-1)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage for this layer."""
        if self.use_cpu:
            weight_memory = self.weight.element_size() * self.weight.nelement() / 1024 / 1024
            bias_memory = self.bias.element_size() * self.bias.nelement() / 1024 / 1024 if self.bias is not None else 0
            return {
                'weight_mb': weight_memory,
                'bias_mb': bias_memory,
                'total_mb': weight_memory + bias_memory,
                'cpu_offloaded': True
            }
        else:
            total_memory = torch.cuda.memory_allocated() - torch.cuda.memory_allocated()
            return {
                'weight_mb': self.weight.element_size() * self.weight.nelement() / 1024 / 1024,
                'bias_mb': self.bias.element_size() * self.bias.nelement() / 1024 / 1024 if self.bias is not None else 0,
                'total_mb': self.weight.element_size() * self.weight.nelement() / 1024 / 1024,
                'cpu_offloaded': False
            }


class PipelineParallelLayer(nn.Module):
    """Pipeline parallel layer for training very large models.
    
    This layer implements pipeline parallelism where different stages of the model
    are distributed across different devices/machines, and micro-batches are
    processed through the pipeline.
    """
    
    def __init__(
        self,
        layer: nn.Module,
        pipeline_stage: int,
        num_pipeline_stages: int,
        micro_batch_size: int = 1,
        enable_pipelining: bool = True
    ):
        """Initialize pipeline parallel layer.
        
        Args:
            layer: The neural network layer
            pipeline_stage: This layer's position in the pipeline
            num_pipeline_stages: Total number of pipeline stages
            micro_batch_size: Size of micro-batches for pipeline processing
            enable_pipelining: Whether to enable pipeline execution
        """
        super().__init__()
        
        self.layer = layer
        self.pipeline_stage = pipeline_stage
        self.num_pipeline_stages = num_pipeline_stages
        self.micro_batch_size = micro_batch_size
        self.enable_pipelining = enable_pipelining
        
        # Pipeline state
        self.input_buffer = [] if pipeline_stage > 0 else None
        self.output_buffer = []
        
        # Check distributed environment
        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional pipeline parallel execution.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor after pipeline processing
        """
        if not self.enable_pipelining or not self.is_distributed:
            # Standard forward pass
            return self.layer(x)
        
        # Pipeline parallel execution
        if self.pipeline_stage == 0:
            # First stage - split into micro-batches
            micro_batches = self._split_micro_batches(x)
            return self._process_pipeline_micro_batches(micro_batches, is_first_stage=True)
        else:
            # Later stages - process inputs from previous stage
            return self._process_pipeline_input(x)
    
    def _split_micro_batches(self, batch: torch.Tensor) -> List[torch.Tensor]:
        """Split batch into micro-batches for pipeline processing."""
        batch_size = batch.size(0)
        if batch_size % self.micro_batch_size != 0:
            warnings.warn(f"Batch size {batch_size} not divisible by micro_batch_size {self.micro_batch_size}")
        
        micro_batches = []
        for i in range(0, batch_size, self.micro_batch_size):
            end_idx = min(i + self.micro_batch_size, batch_size)
            micro_batch = batch[i:end_idx]
            micro_batches.append(micro_batch)
        
        return micro_batches
    
    def _process_pipeline_micro_batches(
        self, 
        micro_batches: List[torch.Tensor], 
        is_first_stage: bool = False
    ) -> torch.Tensor:
        """Process micro-batches through the pipeline."""
        outputs = []
        
        for micro_batch in micro_batches:
            if is_first_stage:
                # First stage - process and send to next stage
                output = self.layer(micro_batch)
                
                # Send to next stage
                if self.pipeline_stage < self.num_pipeline_stages - 1:
                    dist.send(output, dst=self.rank + 1)
                
                outputs.append(output)
            else:
                # Later stage - receive from previous stage
                if self.pipeline_stage > 0:
                    # Receive from previous stage
                    input_tensor = torch.zeros_like(self.input_buffer[-1])
                    dist.recv(input_tensor, src=self.rank - 1)
                    self.input_buffer.append(input_tensor)
                
                # Process
                output = self.layer(self.input_buffer[-1])
                
                # Send to next stage if not last
                if self.pipeline_stage < self.num_pipeline_stages - 1:
                    dist.send(output, dst=self.rank + 1)
                else:
                    # Last stage - collect output
                    outputs.append(output)
        
        # Concatenate outputs from all micro-batches
        return torch.cat(outputs, dim=0)
    
    def _process_pipeline_input(self, x: torch.Tensor) -> torch.Tensor:
        """Process input for pipeline stages that are not first."""
        # Process the input through this layer
        output = self.layer(x)
        
        # Send to next stage if not last stage
        if self.pipeline_stage < self.num_pipeline_stages - 1:
            dist.send(output, dst=self.rank + 1)
        else:
            # Last stage - return output
            pass
        
        return output
    
    def receive_from_previous_stage(self) -> Optional[torch.Tensor]:
        """Receive tensor from previous pipeline stage.
        
        Returns:
            Received tensor or None if no data available
        """
        if not self.is_distributed or self.pipeline_stage == 0:
            return None
        
        # Check if data is available
        if not hasattr(self, '_receive_buffer'):
            return None
        
        return self._receive_buffer
    
    def send_to_next_stage(self, tensor: torch.Tensor) -> bool:
        """Send tensor to next pipeline stage.
        
        Args:
            tensor: Tensor to send
        
        Returns:
            True if sent successfully
        """
        if not self.is_distributed or self.pipeline_stage >= self.num_pipeline_stages - 1:
            return False
        
        dist.send(tensor, dst=self.rank + 1)
        return True
    
    def get_pipeline_efficiency(self) -> float:
        """Calculate pipeline efficiency.
        
        Returns:
            Efficiency ratio (0.0 to 1.0)
        """
        if not self.enable_pipelining or not self.is_distributed:
            return 1.0
        
        # Simple efficiency calculation based on pipeline depth
        # Actual implementation would consider forward/backward pass times
        if self.num_pipeline_stages <= 1:
            return 1.0
        
        # Assume linear scaling with pipeline depth
        efficiency = min(1.0, 1.0 / (1.0 + (self.num_pipeline_stages - 1) * 0.1))
        return efficiency


class ModelParallelWrapper(nn.Module):
    """Model parallel wrapper for distributing model components.
    
    This wrapper enables easy model parallelism by distributing different
    parts of the model across multiple devices or processes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_parallel_config: Dict[str, Any],
        device_map: Optional[Dict[str, torch.device]] = None
    ):
        """Initialize model parallel wrapper.
        
        Args:
            model: The base model
            model_parallel_config: Configuration for model parallelism
            device_map: Mapping of model components to devices
        """
        super().__init__()
        
        self.model = model
        self.config = model_parallel_config
        self.device_map = device_map or {}
        
        # Validate configuration
        self._validate_config()
        
        # Apply model parallelism
        self._apply_model_parallelism()
    
    def _validate_config(self):
        """Validate model parallel configuration."""
        required_keys = ['type', 'strategy']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Model parallel config must include '{key}'")
        
        if self.config['type'] not in ['tensor', 'pipeline', 'data']:
            raise ValueError(f"Invalid model parallel type: {self.config['type']}")
    
    def _apply_model_parallelism(self):
        """Apply the specified model parallelism strategy."""
        strategy = self.config['strategy']
        
        if self.config['type'] == 'tensor':
            self._apply_tensor_parallelism(strategy)
        elif self.config['type'] == 'pipeline':
            self._apply_pipeline_parallelism(strategy)
        elif self.config['type'] == 'data':
            self._apply_data_parallelism(strategy)
    
    def _apply_tensor_parallelism(self, strategy: str):
        """Apply tensor parallelism to model components."""
        if strategy == 'attention':
            # Distribute attention layers
            self._parallelize_attention_layers()
        elif strategy == 'mlp':
            # Distribute MLP layers
            self._parallelize_mlp_layers()
        else:
            raise ValueError(f"Invalid tensor parallel strategy: {strategy}")
    
    def _apply_pipeline_parallelism(self, strategy: str):
        """Apply pipeline parallelism to model components."""
        if strategy == 'depth':
            # Distribute layers along depth
            self._parallelize_by_depth()
        else:
            raise ValueError(f"Invalid pipeline parallel strategy: {strategy}")
    
    def _apply_data_parallelism(self, strategy: str):
        """Apply data parallelism configuration."""
        if strategy == 'sharding':
            # Set up data parallel sharding
            self._setup_data_sharding()
        else:
            raise ValueError(f"Invalid data parallel strategy: {strategy}")
    
    def _parallelize_attention_layers(self):
        """Parallelize attention layers using tensor parallelism."""
        tensor_parallel_size = self.config.get('tensor_parallel_size', 1)
        
        # Find attention layers
        attention_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer)):
                attention_layers.append((name, module))
        
        for name, layer in attention_layers:
            if isinstance(layer, nn.Linear):
                # Replace with tensor parallel linear layer
                parallel_layer = TensorParallelLinear(
                    in_features=layer.in_features,
                    out_features=layer.out_features,
                    tensor_parallel_size=tensor_parallel_size
                )
                
                # Copy weights
                with torch.no_grad():
                    parallel_layer.weight.data = layer.weight.data
                    if layer.bias is not None:
                        parallel_layer.bias.data = layer.bias.data
                
                # Replace in model
                self._replace_module(name, parallel_layer)
    
    def _parallelize_mlp_layers(self):
        """Parallelize MLP layers using tensor parallelism."""
        tensor_parallel_size = self.config.get('tensor_parallel_size', 1)
        
        # Find MLP layers
        mlp_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if it's likely an MLP layer (not attention)
                parent_name = '.'.join(name.split('.')[:-1])
                if 'attention' not in parent_name.lower():
                    mlp_layers.append((name, module))
        
        for name, layer in mlp_layers:
            # Replace with tensor parallel linear layer
            parallel_layer = TensorParallelLinear(
                in_features=layer.in_features,
                out_features=layer.out_features,
                tensor_parallel_size=tensor_parallel_size
            )
            
            # Copy weights
            with torch.no_grad():
                parallel_layer.weight.data = layer.weight.data
                if layer.bias is not None:
                    parallel_layer.bias.data = layer.bias.data
            
            # Replace in model
            self._replace_module(name, parallel_layer)
    
    def _parallelize_by_depth(self):
        """Parallelize model by depth (pipeline parallelism)."""
        num_stages = self.config.get('num_pipeline_stages', 1)
        
        # Get all layers
        layers = list(self.model.children())
        if not layers:
            return
        
        layers_per_stage = len(layers) // num_stages
        
        for stage in range(num_stages):
            start_idx = stage * layers_per_stage
            end_idx = start_idx + layers_per_stage if stage < num_stages - 1 else len(layers)
            stage_layers = layers[start_idx:end_idx]
            
            # Create pipeline parallel wrapper for this stage
            stage_module = nn.Sequential(*stage_layers)
            pipeline_layer = PipelineParallelLayer(
                layer=stage_module,
                pipeline_stage=stage,
                num_pipeline_stages=num_stages
            )
            
            # Replace in model (simplified - assumes single sequential)
            if len(layers) > 0:
                self._replace_module(f'stage_{stage}', pipeline_layer)
    
    def _setup_data_sharding(self):
        """Set up data parallel sharding configuration."""
        # This would configure the model for sharded data parallelism
        # Implementation depends on the specific sharding strategy
        logger.info("Setting up data sharding configuration")
    
    def _replace_module(self, module_path: str, new_module: nn.Module):
        """Replace a module in the model.
        
        Args:
            module_path: Path to the module to replace
            new_module: New module to replace it with
        """
        # Split path into parts
        path_parts = module_path.split('.')
        
        # Navigate to parent module
        parent = self.model
        for part in path_parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        
        # Replace the module
        if path_parts[-1].isdigit():
            parent[int(path_parts[-1])] = new_module
        else:
            setattr(parent, path_parts[-1], new_module)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the model parallel model."""
        return self.model(*args, **kwargs)
    
    def get_parallel_stats(self) -> Dict[str, Any]:
        """Get statistics about the parallelization setup.
        
        Returns:
            Dictionary with parallelization statistics
        """
        stats = {
            'parallel_type': self.config.get('type', 'unknown'),
            'strategy': self.config.get('strategy', 'unknown'),
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
        }
        
        if self.config.get('type') == 'tensor':
            stats['tensor_parallel_size'] = self.config.get('tensor_parallel_size', 1)
        elif self.config.get('type') == 'pipeline':
            stats['num_pipeline_stages'] = self.config.get('num_pipeline_stages', 1)
        
        # Add device information
        device_stats = defaultdict(int)
        for param in self.model.parameters():
            device_stats[param.device.type] += param.numel()
        
        stats['device_distribution'] = dict(device_stats)
        
        return stats
    
    def to(self, device: torch.device):
        """Move model to specified device."""
        return super().to(device)


class ExpertParallelLayer(nn.Module):
    """Expert layer for mixture of experts (MoE) parallel processing.
    
    This layer implements expert-based parallelism where different experts
    are distributed across processes and tokens are routed to appropriate experts.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int,
        expert_top_k: int = 1,
        capacity_factor: float = 1.0,
        gating_noise: float = 0.0
    ):
        """Initialize expert parallel layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            num_experts: Number of experts
            expert_top_k: Number of top experts to activate
            capacity_factor: Token capacity per expert
            gating_noise: Noise in gating network
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.capacity_factor = capacity_factor
        self.gating_noise = gating_noise
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Load balancing
        self.load_loss = 0.0
        self.importance_loss = 0.0
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with expert routing.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Tuple of (output, load_loss, importance_loss)
        """
        batch_size = x.size(0)
        
        # Compute gating scores
        gate_scores = self.gate(x)
        
        if self.gating_noise > 0:
            # Add noise to gating scores
            gate_scores = gate_scores + torch.randn_like(gate_scores) * self.gating_noise
        
        # Get top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.expert_top_k, dim=-1)
        
        # Create routing mask
        routing_mask = torch.zeros_like(gate_scores)
        routing_mask.scatter_(-1, top_k_indices, 1)
        
        # Apply gating
        gate_activations = F.softmax(top_k_scores, dim=-1)
        
        # Distribute tokens to experts
        expert_outputs = []
        expert_loads = []
        
        for expert_idx in range(self.num_experts):
            # Get tokens routed to this expert
            token_mask = routing_mask[:, expert_idx] > 0
            if not token_mask.any():
                # No tokens for this expert
                expert_outputs.append(torch.zeros_like(x))
                expert_loads.append(0)
                continue
            
            # Process tokens through expert
            expert_input = x[token_mask]
            expert_output = self.experts[expert_idx](expert_input)
            expert_outputs.append(expert_output)
            expert_loads.append(expert_input.size(0))
        
        # Combine expert outputs
        output = torch.zeros_like(x)
        for expert_idx, expert_output in enumerate(expert_outputs):
            token_mask = routing_mask[:, expert_idx] > 0
            if token_mask.any():
                # Weight by gating scores
                weights = gate_activations[token_mask, expert_idx:expert_idx+1]
                output[token_mask] += expert_output * weights
        
        # Compute load balancing loss
        load_loss = self._compute_load_loss(expert_loads, batch_size)
        
        # Compute importance loss
        importance_loss = self._compute_importance_loss(gate_scores)
        
        return output, load_loss, importance_loss
    
    def _compute_load_loss(self, expert_loads: List[int], batch_size: int) -> torch.Tensor:
        """Compute load balancing loss.
        
        Args:
            expert_loads: Number of tokens processed by each expert
            batch_size: Total number of tokens in batch
        
        Returns:
            Load balancing loss
        """
        if not expert_loads:
            return torch.tensor(0.0, device=self.experts[0].weight.device)
        
        # Normalize loads
        expected_load = batch_size / self.num_experts
        loads = torch.tensor(expert_loads, dtype=torch.float32, device=self.experts[0].weight.device)
        
        # Compute variance in loads
        load_variance = torch.var(loads)
        
        # Balance factor
        balance_factor = self.num_experts * expected_load
        
        loss = load_variance / balance_factor
        
        return loss
    
    def _compute_importance_loss(self, gate_scores: torch.Tensor) -> torch.Tensor:
        """Compute importance loss to ensure all experts are used.
        
        Args:
            gate_scores: Gating scores for all tokens and experts
        
        Returns:
            Importance loss
        """
        # Compute importance scores
        importance_scores = torch.sum(torch.softmax(gate_scores, dim=-1), dim=0)
        
        # Mean importance
        mean_importance = torch.mean(importance_scores)
        
        # Importance variance
        importance_variance = torch.var(importance_scores)
        
        loss = importance_variance / (mean_importance + 1e-8)
        
        return loss


# Export main classes
__all__ = [
    'TensorParallelLinear',
    'PipelineParallelLayer',
    'ModelParallelWrapper',
    'ExpertParallelLayer'
]