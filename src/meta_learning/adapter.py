import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import math
import json
from dataclasses import dataclass
import copy


class AdapterType(Enum):
    """Types of adapters supported."""
    
    # Basic adapters
    LINEAR = "linear"
    BOTTLENECK = "bottleneck"
    
    # Advanced adapters
    LORA = "lora"
    ADALORA = "adalora"
    
    # Prompt-based adapters
    PREFIX_TUNING = "prefix_tuning"
    P_TUNING = "p_tuning"
    PROMPT_TUNING = "prompt_tuning"
    
    # Layer-wise adapters
    LAYER_NORM_ADAPTER = "layer_norm_adapter"
    ADDITIVE_ADAPTER = "additive_adapter"
    
    # Multi-modal adapters
    CROSS_ATTENTION_ADAPTER = "cross_attention_adapter"
    
    # Specialized adapters
    COMPACTER = "compacter"
    HYPER_ADAPTER = "hyper_adapter"
    
    # Sequential adapters
    SEQUENTIAL_ADAPTER = "sequential_adapter"
    PARALLEL_ADAPTER = "parallel_adapter"


@dataclass
class AdapterConfig:
    """Configuration for adapter layers."""
    
    adapter_type: AdapterType
    input_dim: int
    hidden_dim: Optional[int] = None
    output_dim: Optional[int] = None
    
    # Common parameters
    dropout: float = 0.1
    activation: str = "relu"
    
    # LoRA specific
    rank: int = 8
    alpha: float = 32.0
    dropout_lora: float = 0.05
    
    # Prefix tuning specific
    num_virtual_tokens: int = 20
    
    # Prompt tuning specific
    prompt_length: int = 10
    
    # Sequential adapter specific
    num_layers: int = 2
    
    # General parameters
    bottleneck_ratio: float = 0.1  # For bottleneck adapters
    scaling_factor: float = 1.0
    
    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = max(1, int(self.input_dim * self.bottleneck_ratio))
        if self.output_dim is None:
            self.output_dim = self.input_dim


class LinearAdapter(nn.Module):
    """Simple linear adapter with optional non-linearity."""
    
    def __init__(self, config: AdapterConfig):
        super(LinearAdapter, self).__init__()
        
        self.config = config
        
        # Adapter layers
        self.down_project = nn.Linear(config.input_dim, config.hidden_dim)
        self.up_project = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Activation and dropout
        self.activation = self._get_activation(config.activation)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialization
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
            'elu': nn.ELU(inplace=True)
        }
        return activations.get(activation, nn.ReLU(inplace=True))
    
    def _init_weights(self):
        """Initialize adapter weights."""
        # Xavier initialization for linear layers
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.xavier_uniform_(self.up_project.weight)
        
        # Zero bias for better initial performance
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear adapter."""
        # Down projection
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Up projection
        x = self.up_project(x)
        
        return x


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation (LoRA) adapter."""
    
    def __init__(self, config: AdapterConfig):
        super(LoRAAdapter, self).__init__()
        
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        
        assert config.input_dim == config.output_dim, "LoRA requires input_dim == output_dim"
        
        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.empty(config.input_dim, self.rank, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.empty(self.rank, config.output_dim, dtype=torch.float32)
        )
        
        # Dropout for LoRA
        self.dropout = nn.Dropout(config.dropout_lora)
        
        # Initialize LoRA weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LoRA weights."""
        # Initialize A with small random values
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA adapter."""
        # Compute LoRA adaptation
        result = torch.matmul(x, self.lora_A)  # [batch, rank]
        result = torch.matmul(result, self.lora_B)  # [batch, output_dim]
        result = self.dropout(result)
        
        # Apply scaling
        result = result * self.scaling
        
        return result


class LoraAdapter(nn.Module):
    """LoRA implementation that can be applied to existing linear layers."""
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 32.0,
                 dropout: float = 0.05,
                 merge_weights: bool = False):
        super(LoraAdapter, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        
        # LoRA matrices
        self.lora_A = nn.Parameter(
            torch.empty(in_features, rank, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.empty(rank, out_features, dtype=torch.float32)
        )
        
        self.scaling = alpha / rank
        
        # Freeze the original weights
        self.frozen_weight = None
        
        # Dropout
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """Merge LoRA weights into base weights."""
        if self.frozen_weight is not None:
            self.frozen_weight += self.scaling * torch.matmul(self.lora_A, self.lora_B)
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA."""
        result = torch.matmul(x, self.lora_A)  # [batch, rank]
        result = torch.matmul(result, self.lora_B)  # [batch, out_features]
        result = self.lora_dropout(result)
        result = result * self.scaling
        
        return result


class PrefixTuningAdapter(nn.Module):
    """Prefix tuning adapter for language models."""
    
    def __init__(self, 
                 config: AdapterConfig,
                 num_layers: int = 12,
                 is_encoder_decoder: bool = False):
        super(PrefixTuningAdapter, self).__init__()
        
        self.config = config
        self.num_virtual_tokens = config.num_virtual_tokens
        self.num_layers = num_layers
        self.is_encoder_decoder = is_encoder_decoder
        
        # Prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(
                2 * num_layers if is_encoder_decoder else num_layers,
                num_virtual_tokens,
                config.input_dim
            )
        )
        
        # MLP for prefix processing
        self.prefix_mlp = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.input_dim)
        )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prefix tuning weights."""
        nn.init.normal_(self.prefix_embeddings, mean=0.0, std=0.02)
    
    def get_prompt(self, batch_size: int) -> torch.Tensor:
        """Get prefix prompts for a given batch size."""
        prompt = self.prefix_embeddings.expand(-1, batch_size, -1, -1)
        
        # Process through MLP
        prompt = prompt.transpose(0, 1)  # [batch, layers, tokens, dim]
        batch_size, num_layers, seq_len, hidden_dim = prompt.shape
        
        prompt = prompt.reshape(batch_size, num_layers * seq_len, hidden_dim)
        prompt = self.prefix_mlp(prompt)
        
        return prompt.reshape(batch_size, num_layers, seq_len, hidden_dim)


class PromptTuningAdapter(nn.Module):
    """Prompt tuning adapter for text generation."""
    
    def __init__(self, config: AdapterConfig, vocab_size: int = 30000):
        super(PromptTuningAdapter, self).__init__()
        
        self.config = config
        self.prompt_length = config.prompt_length
        self.vocab_size = vocab_size
        
        # Learnable prompts
        self.prompt_tokens = nn.Parameter(
            torch.randint(0, vocab_size, (1, self.prompt_length))
        )
        
        # Embedding layer for prompt tokens
        self.embedding = nn.Embedding(vocab_size, config.input_dim)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prompt tuning weights."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def get_prompt(self) -> torch.Tensor:
        """Get prompt embeddings."""
        return self.embedding(self.prompt_tokens)  # [1, prompt_length, dim]


class AdditiveAdapter(nn.Module):
    """Additive adapter that adds transformation to input."""
    
    def __init__(self, config: AdapterConfig):
        super(AdditiveAdapter, self).__init__()
        
        self.config = config
        
        # Additive transformation
        self.additive_net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.additive_net[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(self.additive_net[2].weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with additive transformation."""
        additive_output = self.additive_net(x)
        return x + additive_output  # Additive skip connection


class SequentialAdapter(nn.Module):
    """Sequential adapter combining multiple adapter types."""
    
    def __init__(self, configs: List[AdapterConfig]):
        super(SequentialAdapter, self).__init__()
        
        self.adapters = nn.ModuleList()
        
        for config in configs:
            adapter = self._create_adapter(config)
            self.adapters.append(adapter)
    
    def _create_adapter(self, config: AdapterConfig) -> nn.Module:
        """Create adapter based on configuration."""
        if config.adapter_type == AdapterType.LINEAR:
            return LinearAdapter(config)
        elif config.adapter_type == AdapterType.LORA:
            return LoraAdapter(config.input_dim, config.output_dim, config.rank)
        elif config.adapter_type == AdapterType.ADDITIVE_ADAPTER:
            return AdditiveAdapter(config)
        else:
            # Default to linear adapter
            return LinearAdapter(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequential adapters."""
        for adapter in self.adapters:
            x = adapter(x)
        return x


class LanguageAdapter(nn.Module):
    """Specialized adapter for language model fine-tuning."""
    
    def __init__(self, 
                 config: AdapterConfig,
                 use_prefix_tuning: bool = True,
                 use_lora: bool = True):
        super(LanguageAdapter, self).__init__()
        
        self.config = config
        self.use_prefix_tuning = use_prefix_tuning
        self.use_lora = use_lora
        
        # Create different adapter components
        self.adapters = nn.ModuleDict()
        
        if use_prefix_tuning:
            self.adapters['prefix'] = PrefixTuningAdapter(config)
        
        if use_lora:
            # For language models, LoRA is often applied to attention layers
            self.adapters['lora'] = LoraAdapter(
                config.input_dim, config.output_dim, config.rank
            )
        
        # Base adapter
        self.adapters['base'] = LinearAdapter(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through language adapter."""
        # Apply base adapter
        output = self.adapters['base'](x)
        
        # Apply additional adapters if enabled
        if self.use_lora and 'lora' in self.adapters:
            lora_output = self.adapters['lora'](x)
            output = output + lora_output
        
        return output


class VisionAdapter(nn.Module):
    """Specialized adapter for vision model fine-tuning."""
    
    def __init__(self, config: AdapterConfig, input_size: Tuple[int, int] = (224, 224)):
        super(VisionAdapter, self).__init__()
        
        self.config = config
        self.input_size = input_size
        
        # Vision-specific adapters
        self.adapters = nn.ModuleDict()
        
        # Feature adapter for different resolutions
        self.adapters['feature_adapter'] = LinearAdapter(config)
        
        # Spatial adapter for handling spatial information
        self.spatial_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through vision adapter."""
        # Apply feature adapter
        features = self.adapters['feature_adapter'](x)
        
        # Apply spatial adapter
        if x.dim() == 4:  # Image input
            spatial_features = self.spatial_adapter(x)
        else:
            spatial_features = features.mean(dim=1)  # Fallback for 1D features
        
        return features, spatial_features


class MultiModalAdapter(nn.Module):
    """Adapter for multi-modal models (text + vision)."""
    
    def __init__(self, 
                 text_config: AdapterConfig,
                 vision_config: AdapterConfig,
                 fusion_method: str = "attention"):
        super(MultiModalAdapter, self).__init__()
        
        self.fusion_method = fusion_method
        
        # Text and vision adapters
        self.text_adapter = LanguageAdapter(text_config)
        self.vision_adapter = VisionAdapter(vision_config)
        
        # Fusion layers
        if fusion_method == "attention":
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=text_config.input_dim,
                num_heads=8,
                dropout=text_config.dropout
            )
        elif fusion_method == "linear":
            self.fusion_layer = nn.Linear(
                text_config.input_dim + vision_config.input_dim,
                text_config.input_dim
            )
        else:
            self.fusion_layer = nn.Identity()
    
    def forward(self, 
                text_features: torch.Tensor,
                vision_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-modal adapter."""
        # Process text features
        text_output = self.text_adapter(text_features)
        
        # Process vision features
        if isinstance(vision_features, tuple):
            vision_output = vision_features[0]  # Use main features
        else:
            vision_output = self.vision_adapter(vision_features)
        
        # Fusion
        if self.fusion_method == "attention":
            # Cross-attention between text and vision
            fused_output, _ = self.fusion_layer(
                text_output, vision_output, vision_output
            )
        elif self.fusion_method == "linear":
            # Concatenate and linearly fuse
            combined = torch.cat([text_output, vision_output], dim=-1)
            fused_output = self.fusion_layer(combined)
        else:
            # Simple addition
            fused_output = text_output + vision_output
        
        return fused_output


class AdapterModel(nn.Module):
    """Main model class that incorporates adapters for parameter-efficient fine-tuning."""
    
    def __init__(self, 
                 base_model: nn.Module,
                 adapter_configs: List[AdapterConfig],
                 adapter_positions: List[str] = None,
                 enable_adapter_dropout: bool = True):
        """
        Initialize model with adapters.
        
        Args:
            base_model: Pre-trained base model
            adapter_configs: List of adapter configurations
            adapter_positions: Where to insert adapters (e.g., ['layer1', 'layer2'])
            enable_adapter_dropout: Whether to enable adapter dropout during training
        """
        super(AdapterModel, self).__init__()
        
        self.base_model = base_model
        self.adapter_configs = adapter_configs
        self.adapter_positions = adapter_positions or []
        self.enable_adapter_dropout = enable_adapter_dropout
        
        # Freeze base model parameters
        self.freeze_base_model()
        
        # Create adapters
        self.adapters = nn.ModuleList()
        for config in adapter_configs:
            self.adapters.append(self._create_adapter(config))
        
        # Adapter dropout probability
        self.adapter_dropout_p = 0.1
    
    def freeze_base_model(self):
        """Freeze base model parameters to enable parameter-efficient fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze base model parameters (full fine-tuning)."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def _create_adapter(self, config: AdapterConfig) -> nn.Module:
        """Create adapter based on configuration."""
        if config.adapter_type == AdapterType.LINEAR:
            return LinearAdapter(config)
        elif config.adapter_type == AdapterType.LORA:
            return LoraAdapter(config.input_dim, config.output_dim, config.rank)
        elif config.adapter_type == AdapterType.PREFIX_TUNING:
            return PrefixTuningAdapter(config)
        elif config.adapter_type == AdapterType.PROMPT_TUNING:
            return PromptTuningAdapter(config)
        elif config.adapter_type == AdapterType.ADDITIVE_ADAPTER:
            return AdditiveAdapter(config)
        elif config.adapter_type == AdapterType.SEQUENTIAL_ADAPTER:
            # For sequential, create multiple sub-adapters
            sub_configs = [config] * config.num_layers
            return SequentialAdapter(sub_configs)
        else:
            # Default to linear adapter
            return LinearAdapter(config)
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the model with adapters."""
        # Get base model output
        base_output = self.base_model(*args, **kwargs)
        
        # Apply adapters (simplified - in practice, you'd insert adapters at specific layers)
        if self.adapters and self.training and self.enable_adapter_dropout:
            # Apply adapter dropout during training
            for adapter in self.adapters:
                if isinstance(adapter, (LinearAdapter, LoraAdapter, AdditiveAdapter)):
                    if torch.rand(1).item() < self.adapter_dropout_p:
                        continue  # Skip this adapter
                    else:
                        # Apply adapter transformation
                        if base_output.dim() > 1:  # Skip for 0D tensors
                            base_output = base_output + adapter(base_output)
        
        return base_output
    
    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get parameters that are trainable (adapters only)."""
        trainable_params = []
        for adapter in self.adapters:
            trainable_params.extend(adapter.parameters())
        return trainable_params
    
    def get_parameter_count(self) -> Tuple[int, int]:
        """Get total parameters and trainable parameters count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.get_trainable_parameters())
        
        return total_params, trainable_params
    
    def save_adapters(self, path: str):
        """Save only adapter weights."""
        adapter_state_dict = {}
        for i, adapter in enumerate(self.adapters):
            adapter_state_dict[f'adapter_{i}'] = adapter.state_dict()
        
        torch.save(adapter_state_dict, path)
    
    def load_adapters(self, path: str, strict: bool = True):
        """Load adapter weights."""
        adapter_state_dict = torch.load(path, map_location='cpu')
        
        for i, adapter in enumerate(self.adapters):
            adapter_key = f'adapter_{i}'
            if adapter_key in adapter_state_dict:
                adapter.load_state_dict(adapter_state_dict[adapter_key], strict=strict)
    
    def get_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of adapter configuration and statistics."""
        total_params, trainable_params = self.get_parameter_count()
        
        summary = {
            'base_model_params': total_params - trainable_params,
            'adapter_params': trainable_params,
            'total_params': total_params,
            'trainable_percentage': (trainable_params / total_params) * 100,
            'num_adapters': len(self.adapters),
            'adapter_configs': [
                {
                    'type': config.adapter_type.value,
                    'input_dim': config.input_dim,
                    'hidden_dim': config.hidden_dim,
                    'output_dim': config.output_dim
                }
                for config in self.adapter_configs
            ]
        }
        
        return summary


# Utility functions for adapter training
class AdapterTrainer:
    """Trainer class for adapter-based fine-tuning."""
    
    def __init__(self, 
                 model: AdapterModel,
                 optimizer: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler._LRScheduler = None):
        self.model = model
        self.optimizer = optimizer or torch.optim.AdamW(
            model.get_trainable_parameters(), lr=1e-4
        )
        self.scheduler = scheduler
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, device: str = 'cpu') -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, device: str = 'cpu') -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # Update best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
        
        return avg_loss
    
    def train(self, 
              train_loader,
              val_loader = None,
              num_epochs: int = 10,
              device: str = 'cpu') -> Dict[str, List[float]]:
        """Complete training loop."""
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader, device)
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader, device)
                history['val_loss'].append(val_loss)
                
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}")
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        return history


# Example usage and testing
if __name__ == "__main__":
    # Create a simple base model for testing
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=512, output_dim=10):
            super(SimpleModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    # Create adapter configurations
    adapter_configs = [
        AdapterConfig(
            adapter_type=AdapterType.LINEAR,
            input_dim=512,
            hidden_dim=64,
            output_dim=512
        ),
        AdapterConfig(
            adapter_type=AdapterType.LORA,
            input_dim=512,
            output_dim=512,
            rank=8
        )
    ]
    
    # Create base model and adapter model
    base_model = SimpleModel()
    adapter_model = AdapterModel(base_model, adapter_configs)
    
    # Print parameter counts
    total_params, trainable_params = adapter_model.get_parameter_count()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {(trainable_params/total_params)*100:.2f}%")
    
    # Get adapter summary
    summary = adapter_model.get_adapter_summary()
    print("\nAdapter Summary:")
    print(json.dumps(summary, indent=2))
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 512)
    output = adapter_model(x)
    print(f"\nOutput shape: {output.shape}")
    
    # Test different adapter types
    print("\nTesting different adapter types...")
    
    # Test LoRA adapter
    lora_config = AdapterConfig(
        adapter_type=AdapterType.LORA,
        input_dim=128,
        output_dim=128,
        rank=4
    )
    lora_adapter = LoraAdapter(128, 128, rank=4)
    test_input = torch.randn(2, 128)
    lora_output = lora_adapter(test_input)
    print(f"LoRA output shape: {lora_output.shape}")
    
    # Test Prefix tuning
    prefix_config = AdapterConfig(
        adapter_type=AdapterType.PREFIX_TUNING,
        input_dim=256,
        num_virtual_tokens=10
    )
    prefix_adapter = PrefixTuningAdapter(prefix_config)
    prefix_prompt = prefix_adapter.get_prompt(batch_size=2)
    print(f"Prefix tuning prompt shape: {prefix_prompt.shape}")
    
    print("\nAdapter module testing completed successfully!")
