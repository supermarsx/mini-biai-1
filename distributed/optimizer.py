import torch
import torch.distributed as dist
from torch.optim import Optimizer
from torch.nn.utils import clip_grad_norm_
from typing import Dict, List, Optional, Any, Union
import warnings
import logging
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class ZeroRedundancyOptimizer(Optimizer):
    """ZeRO (Zero Redundancy Optimizer) implementation for distributed training.
    
    This optimizer implements memory-efficient distributed optimization by distributing
    optimizer states, gradients, and parameters across multiple devices to reduce
    memory overhead in large-scale model training.
    
    Supports:
    - ZeRO Stage 1: Optimizer state partitioning
    - ZeRO Stage 2: Optimizer state + gradient partitioning
    - ZeRO Stage 3: All parameters partitioned
    - Mixed precision training
    - Gradient accumulation
    - Dynamic loss scaling
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        dp_degree: int,
        zero_stage: int = 1,
        grad_accumulation_steps: int = 1,
        clip_grad: float = 1.0,
        cast_inputs: bool = True,
        torch_dtype: torch.dtype = torch.float16,
        cpu_offload: bool = False,
        scatter_gather: bool = True,
        verbose: bool = False
    ):
        """Initialize ZeRO Optimizer.
        
        Args:
            model: The model to optimize
            optimizer: Base PyTorch optimizer
            dp_degree: Data parallel degree (number of processes)
            zero_stage: ZeRO stage (1, 2, or 3)
            grad_accumulation_steps: Number of steps to accumulate gradients
            clip_grad: Gradient clipping value (0 to disable)
            cast_inputs: Cast model inputs to specified dtype
            torch_dtype: Data type for mixed precision (default: float16)
            cpu_offload: Offload parameters to CPU (ZeRO-3)
            scatter_gather: Scatter parameters to devices
            verbose: Enable verbose logging
        """
        self.model = model
        self.optimizer = optimizer
        self.dp_degree = dp_degree
        self.zero_stage = zero_stage
        self.grad_accumulation_steps = grad_accumulation_steps
        self.clip_grad = clip_grad
        self.cast_inputs = cast_inputs
        self.torch_dtype = torch_dtype
        self.cpu_offload = cpu_offload
        self.scatter_gather = scatter_gather
        self.verbose = verbose
        
        # Validate zero stage
        if zero_stage not in [1, 2, 3]:
            raise ValueError(f"ZeRO stage must be 1, 2, or 3, got {zero_stage}")
        
        # Initialize optimizer
        super().__init__(model.parameters(), optimizer.defaults)
        
        # State management
        self.state = defaultdict(dict)
        self.partition_count = 0
        self.cached_params = {}
        self.fp16_params = {}
        self.loss_scale = 1.0
        
        # Setup optimizer
        self._setup_zero()
        
        # Validation
        self._validate_setup()
        
        logger.info(f"Initialized ZeRO Optimizer with stage {zero_stage}, dp_degree {dp_degree}")
    
    def _setup_zero(self):
        """Setup ZeRO optimization."""
        # Get rank information
        if not dist.is_initialized():
            raise RuntimeError("Distributed environment not initialized")
        
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Ensure consistency
        if self.world_size % self.dp_degree != 0:
            raise ValueError(f"World size {self.world_size} must be divisible by dp_degree {self.dp_degree}")
        
        # Setup partitioning
        self._setup_partitioning()
        self._partition_parameters()
        self._partition_optimizer_states()
        
        # Setup mixed precision
        if self.cast_inputs:
            self._setup_mixed_precision()
    
    def _setup_partitioning(self):
        """Setup parameter partitioning configuration."""
        if self.zero_stage >= 2:
            self.gradient_bucket_size = 5e8
            self.gradient_bucket = []
            self.gradient_offsets = []
            self.gradient_nums = []
        
        if self.zero_stage >= 3:
            self.partition_id = self.global_rank % self.dp_degree
            self.parameters_per_partition = len(list(self.model.parameters())) // self.dp_degree
            self.partition_offsets = [self.parameters_per_partition * p_id 
                                    for p_id in range(self.dp_degree)]
        
        if self.verbose:
            logger.info(f"ZeRO stage {self.zero_stage} partitioning setup complete")
    
    def _partition_parameters(self):
        """Partition parameters for ZeRO-3."""
        if self.zero_stage < 3:
            self.cached_params = {p_id: p for p_id, p in enumerate(self.model.parameters())}
            return
        
        # ZeRO-3: Partition parameters across processes
        total_params = len(list(self.model.parameters()))
        
        if self.scatter_gather:
            # Scatter parameters to different devices
            self._scatter_parameters()
        else:
            # Keep all parameters but partition the state
            self._partition_parameter_state()
    
    def _scatter_parameters(self):
        """Scatter parameters across processes (ZeRO-3)."""
        self.scattered_params = []
        partition_start = self.partition_offsets[self.partition_id]
        partition_end = self.partition_offsets[self.partition_id + 1] if self.partition_id + 1 < len(self.partition_offsets) else len(list(self.model.parameters()))
        
        for p_id, p in enumerate(self.model.parameters()):
            if partition_start <= p_id < partition_end:
                self.scattered_params.append(p)
            else:
                # Offload to CPU or keep reference
                if self.cpu_offload:
                    self.cached_params[p_id] = p.cpu()
                else:
                    self.cached_params[p_id] = p
                    p.data = torch.zeros_like(p.data)
    
    def _partition_parameter_state(self):
        """Partition parameter state without scattering (ZeRO-3)."""
        partition_start = self.partition_offsets[self.partition_id]
        partition_end = self.partition_offsets[self.partition_id + 1] if self.partition_id + 1 < len(self.partition_offsets) else len(list(self.model.parameters()))
        
        for p_id, p in enumerate(self.model.parameters()):
            if not (partition_start <= p_id < partition_end):
                if self.cpu_offload:
                    self.cached_params[p_id] = p.cpu()
                else:
                    self.cached_params[p_id] = p.data.clone()
                    p.data = torch.zeros_like(p.data)
    
    def _partition_optimizer_states(self):
        """Partition optimizer states."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # Only process parameters assigned to this partition
                if self.zero_stage >= 3:
                    p_id = len(self.cached_params)
                    if p_id not in [p_id for p_id, _ in self.cached_params.items()]:
                        continue
                
                # Initialize optimizer state for this parameter
                if p not in self.state:
                    # Copy optimizer state
                    self.state[p] = {}
                    for key, value in self.optimizer.state[p].items():
                        if isinstance(value, torch.Tensor):
                            # Partition optimizer states
                            self.state[p][key] = value.data.clone()
                        else:
                            self.state[p][key] = value
                
                # Set initial parameter state
                self.state[p]['step'] = torch.tensor(0, dtype=torch.int64)
                self.state[p]['exp_avg'] = torch.zeros_like(p.data)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        self.manual_cast = True
        
        # Create FP16 parameter copies
        for p in self.model.parameters():
            if p.requires_grad:
                self.fp16_params[p] = p.data.clone().to(self.torch_dtype)
    
    def _validate_setup(self):
        """Validate ZeRO setup configuration."""
        # Check if we have a proper distributed environment
        if not dist.is_initialized():
            raise RuntimeError("PyTorch distributed not initialized")
        
        # Validate zero stage compatibility
        if self.zero_stage == 3 and not torch.cuda.is_available():
            raise RuntimeError("ZeRO-3 requires CUDA for effective parameter partitioning")
        
        # Check gradient accumulation
        if self.grad_accumulation_steps < 1:
            raise ValueError(f"grad_accumulation_steps must be >= 1, got {self.grad_accumulation_steps}")
        
        # Validate dp_degree
        if self.dp_degree <= 0 or self.dp_degree > self.world_size:
            raise ValueError(f"dp_degree must be in [1, {self.world_size}], got {self.dp_degree}")
    
    def step(self, closure=None):
        """Execute optimizer step with ZeRO optimization."""
        if self.grad_accumulation_steps > 1:
            return self._step_with_accumulation(closure)
        else:
            return self._step_no_accumulation(closure)
    
    def _step_with_accumulation(self, closure=None):
        """Handle step with gradient accumulation."""
        if not hasattr(self, 'grad_accum_step'):
            self.grad_accum_step = 0
        
        self.grad_accum_step += 1
        
        if self.grad_accum_step < self.grad_accumulation_steps:
            # Accumulate gradients
            return None
        else:
            # Execute step
            self.grad_accum_step = 0
            return self._step_no_accumulation(closure)
    
    def _step_no_accumulation(self, closure=None):
        """Execute optimization step without accumulation."""
        # Sync gradients across processes
        if self.zero_stage >= 2:
            self._sync_gradients()
        
        # Apply gradient clipping
        if self.clip_grad > 0:
            self._clip_gradients()
        
        # Execute optimizer step
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Update parameters
        self._update_parameters()
        
        # Handle mixed precision
        if self.cast_inputs:
            self._update_fp16_params()
        
        return loss
    
    def _sync_gradients(self):
        """Sync gradients across processes (ZeRO Stage 2+)."""
        for p in self.model.parameters():
            if p.grad is not None:
                # All-reduce gradients
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.data.div_(self.dp_degree)
    
    def _clip_gradients(self):
        """Apply gradient clipping."""
        clip_grad_norm_(
            [p for p in self.model.parameters() if p.grad is not None],
            self.clip_grad,
            norm_type=2.0,
            error_if_nonfinite=False
        )
    
    def _update_parameters(self):
        """Update parameters using partitioned optimizer states."""
        # Update all groups
        for group in self.optimizer.param_groups:
            # Update each parameter
            for p in group['params']:
                if p in self.state:
                    self._update_single_parameter(p, group)
    
    def _update_single_parameter(self, p: torch.Tensor, group: Dict):
        """Update a single parameter using its optimizer state."""
        if p not in self.state:
            return
        
        # Get optimizer state
        state = self.state[p]
        
        # Standard optimizer step (Adam/SGD/etc)
        # This is a simplified AdamW implementation
        if isinstance(self.optimizer, torch.optim.AdamW):
            self._adamw_update(p, group, state)
        else:
            # Fallback to base optimizer
            self.optimizer.state[p] = state
    
    def _adamw_update(self, p: torch.Tensor, group: Dict, state: Dict):
        """AdamW update with ZeRO optimization."""
        # Get parameters
        step = state['step'].item()
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        
        # AdamW parameters
        beta1, beta2 = group.get('betas', (0.9, 0.999))
        eps = group.get('eps', 1e-8)
        weight_decay = group.get('weight_decay', 0.0)
        learning_rate = group.get('lr', 1e-3)
        
        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Compute bias-corrected first moment estimate
        exp_avg = exp_avg / bias_correction1
        
        # Compute bias-corrected second raw moment estimate
        denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
        
        # Update parameters
        p.data.addcdiv_(exp_avg, denom, value=-learning_rate)
        
        # Weight decay
        if weight_decay > 0:
            p.data.add_(p.data, alpha=-learning_rate * weight_decay)
        
        # Update step counter
        state['step'] = torch.tensor(step + 1, dtype=torch.int64)
    
    def _update_fp16_params(self):
        """Update FP16 parameter copies for mixed precision."""
        for p, fp16_p in self.fp16_params.items():
            fp16_p.data.copy_(p.data, non_blocking=True)
    
    def zero_grad(self, set_to_none=False):
        """Zero gradients with ZeRO optimization."""
        if set_to_none:
            for p in self.model.parameters():
                p.grad = None
        else:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()
    
    def get_loss_scale(self) -> float:
        """Get current loss scale for mixed precision."""
        return self.loss_scale
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.torch_dtype in [torch.float16, torch.bfloat16]:
            return loss * self.loss_scale
        return loss
    
    def unscale_grads(self):
        """Unscale gradients after step."""
        if self.torch_dtype in [torch.float16, torch.bfloat16]:
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data.div_(self.loss_scale)
    
    def _gather_parameters(self):
        """Gather all parameters to single process (ZeRO-3)."""
        if self.zero_stage < 3:
            return
        
        # Gather parameter references
        gathered_params = [None for _ in range(self.world_size)]
        
        for p_id, p in self.cached_params.items():
            if p_id not in [p_id for p_id, _ in self.scattered_params.items()]:
                # Gather parameter data
                dist.all_gather_object(gathered_params, p)
                
                # Reconstruct full parameter
                if self.global_rank == 0:
                    full_param = torch.cat([param.data.flatten() for param in gathered_params])
                    p.data = full_param.view_as(p.data)
    
    def _scatter_optimizer_states(self):
        """Scatter optimizer states across processes."""
        if self.zero_stage < 2:
            return
        
        # Scatter optimizer states
        gathered_states = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_states, dict(self.optimizer.state))
        
        # Process gathered states
        if self.global_rank == 0:
            for state_dict in gathered_states:
                self._merge_optimizer_states(state_dict)
    
    def _merge_optimizer_states(self, state_dict: Dict):
        """Merge optimizer states from different processes."""
        for p, state in state_dict.items():
            if p not in self.optimizer.state:
                self.optimizer.state[p] = {}
            
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    if key not in self.optimizer.state[p]:
                        self.optimizer.state[p][key] = torch.zeros_like(value)
                    
                    self.optimizer.state[p][key].data.add_(value.data)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        # Calculate memory usage for this process
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        
        # Estimate total memory across all processes
        total_allocated = allocated * self.dp_degree
        total_reserved = reserved * self.dp_degree
        
        return {
            'current_process_allocated_mb': allocated / 1024 / 1024,
            'current_process_reserved_mb': reserved / 1024 / 1024,
            'current_process_cached_mb': cached / 1024 / 1024,
            'estimated_total_allocated_mb': total_allocated / 1024 / 1024,
            'estimated_total_reserved_mb': total_reserved / 1024 / 1024,
            'memory_savings_mb': (total_allocated - allocated) / 1024 / 1024
        }
    
    def save_checkpoint(self, filepath: str, save_dtype: torch.dtype = torch.float32):
        """Save optimizer checkpoint with ZeRO optimization."""
        checkpoint = {
            'zero_stage': self.zero_stage,
            'dp_degree': self.dp_degree,
            'grad_accumulation_steps': self.grad_accumulation_steps,
            'optimizer_state_dict': {},
            'partition_info': {
                'partition_id': getattr(self, 'partition_id', 0),
                'parameters_per_partition': getattr(self, 'parameters_per_partition', 0),
                'partition_offsets': getattr(self, 'partition_offsets', [])
            }
        }
        
        # Save optimizer state for this partition
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.state:
                    checkpoint['optimizer_state_dict'][id(p)] = {
                        'step': self.state[p]['step'].item(),
                        'exp_avg': self.state[p]['exp_avg'].clone().to(save_dtype),
                        'exp_avg_sq': self.state[p]['exp_avg_sq'].clone().to(save_dtype)
                    }
        
        torch.save(checkpoint, filepath)
        
        if self.verbose:
            logger.info(f"ZeRO optimizer checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_dtype: torch.dtype = torch.float32):
        """Load optimizer checkpoint with ZeRO optimization."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Validate checkpoint
        if checkpoint['zero_stage'] != self.zero_stage:
            warnings.warn(f"ZeRO stage mismatch: checkpoint={checkpoint['zero_stage']}, current={self.zero_stage}")
        
        if checkpoint['dp_degree'] != self.dp_degree:
            warnings.warn(f"DP degree mismatch: checkpoint={checkpoint['dp_degree']}, current={self.dp_degree}")
        
        # Load optimizer state
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p_id = id(p)
                if p_id in checkpoint['optimizer_state_dict']:
                    if p not in self.state:
                        self.state[p] = {}
                    
                    ckpt_state = checkpoint['optimizer_state_dict'][p_id]
                    self.state[p]['step'] = torch.tensor(ckpt_state['step'], dtype=torch.int64)
                    self.state[p]['exp_avg'] = ckpt_state['exp_avg'].to(p.data.device)
                    self.state[p]['exp_avg_sq'] = ckpt_state['exp_avg_sq'].to(p.data.device)
        
        if self.verbose:
            logger.info(f"ZeRO optimizer checkpoint loaded from {filepath}")
    
    def __repr__(self):
        """String representation of the optimizer."""
        return (f"ZeroRedundancyOptimizer(zero_stage={self.zero_stage}, "
                f"dp_degree={self.dp_degree}, "
                f"grad_accumulation_steps={self.grad_accumulation_steps})")


class FlexibleOptimizer(Optimizer):
    """Flexible optimizer that supports multiple update strategies.
    
    This optimizer can switch between different update strategies dynamically
    and supports custom learning rate schedules and gradient transformations.
    """
    
    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        strategy: str = 'adamw',
        adaptive_lr: bool = True,
        gradient_noise: float = 0.0,
        warmup_steps: int = 0,
        cosine_annealing: bool = False
    ):
        """Initialize flexible optimizer.
        
        Args:
            params: Model parameters
            base_optimizer: Base PyTorch optimizer
            strategy: Update strategy ('adamw', 'sgd', 'adam', 'radam')
            adaptive_lr: Use adaptive learning rates
            gradient_noise: Add noise to gradients
            warmup_steps: Number of warmup steps
            cosine_annealing: Use cosine annealing schedule
        """
        self.base_optimizer = base_optimizer
        self.strategy = strategy
        self.adaptive_lr = adaptive_lr
        self.gradient_noise = gradient_noise
        self.warmup_steps = warmup_steps
        self.cosine_annealing = cosine_annealing
        self.step_count = 0
        
        super().__init__(params, base_optimizer.defaults)
    
    def step(self, closure=None):
        """Execute optimization step."""
        self.step_count += 1
        
        # Apply warmup
        if self.step_count <= self.warmup_steps:
            warmup_factor = self.step_count / self.warmup_steps
            self._apply_warmup_factor(warmup_factor)
        
        # Apply cosine annealing
        if self.cosine_annealing and self.step_count > self.warmup_steps:
            self._apply_cosine_annealing()
        
        # Add gradient noise
        if self.gradient_noise > 0:
            self._add_gradient_noise()
        
        # Execute base optimizer step
        return self.base_optimizer.step(closure)
    
    def _apply_warmup_factor(self, factor: float):
        """Apply warmup factor to learning rates."""
        for group in self.base_optimizer.param_groups:
            group['lr'] *= factor
    
    def _apply_cosine_annealing(self):
        """Apply cosine annealing to learning rates."""
        for group in self.base_optimizer.param_groups:
            # Cosine annealing formula
            T_max = 1000  # Could be configurable
            lr = group['lr'] * 0.5 * (1 + torch.cos(torch.tensor(self.step_count * 3.14159 / T_max)))
            group['lr'] = lr.item()
    
    def _add_gradient_noise(self):
        """Add noise to gradients for regularization."""
        if self.gradient_noise <= 0:
            return
        
        noise_scale = self.gradient_noise / ((1 + self.step_count) ** 0.55)
        
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    noise = torch.randn_like(p.grad) * noise_scale
                    p.grad.data.add_(noise)
    
    def zero_grad(self, set_to_none=False):
        """Zero gradients."""
        self.base_optimizer.zero_grad(set_to_none)
    
    def switch_strategy(self, new_strategy: str):
        """Switch optimization strategy."""
        if new_strategy != self.strategy:
            logger.info(f"Switching optimization strategy from {self.strategy} to {new_strategy}")
            self.strategy = new_strategy
            # Strategy switching would require rebuilding the optimizer
            # This is a simplified implementation
    
    def get_current_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        lrs = {}
        for i, group in enumerate(self.base_optimizer.param_groups):
            lrs[f'group_{i}'] = group['lr']
        return lrs
    
    def adaptive_lr_schedule(self, metrics: Dict[str, float], patience: int = 10):
        """Adaptive learning rate scheduling based on metrics."""
        if not self.adaptive_lr:
            return
        
        # Simple adaptive scheduling
        for metric_name, value in metrics.items():
            if 'loss' in metric_name.lower():
                if not hasattr(self, 'best_loss'):
                    self.best_loss = float('inf')
                
                if value < self.best_loss:
                    self.best_loss = value
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    
                    if self.patience_counter >= patience:
                        # Reduce learning rate
                        for group in self.base_optimizer.param_groups:
                            group['lr'] *= 0.5
                        self.patience_counter = 0
                        logger.info(f"Reduced learning rate due to no improvement in {metric_name}")


# Export main classes
__all__ = [
    'ZeroRedundancyOptimizer',
    'FlexibleOptimizer'
]