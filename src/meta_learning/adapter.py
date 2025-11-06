import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import math


@dataclass
class AdapterConfig:
    """Configuration for adapter modules."""
    adapter_size: int = 64
    hidden_size: int = 512
    bottleneck_size: int = 64
    dropout: float = 0.1
    activation: str = 'gelu'
    non_linearity: str = 'gelu'
    bias: bool = True
    layer_norm_epsilon: float = 1e-12
    pre_norm: bool = False
    post_norm: bool = True
    scaler: float = 1.0
    
    def __post_init__(self):
        if self.activation == 'gelu':
            self.activation_fn = F.gelu
        elif self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'tanh':
            self.activation_fn = torch.tanh
        elif self.activation == 'swish':
            def swish(x):
                return x * torch.sigmoid(x)
            self.activation_fn = swish
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


class BaseAdapter(nn.Module):
    """Base class for all adapter modules."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.output_size = config.hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter."""
        raise NotImplementedError
        
    def resize_adapter(self, new_size: int):
        """Resize the adapter to a new bottleneck size."""
        self.config.bottleneck_size = new_size
        
    def get_trainable_parameters(self) -> torch.Tensor:
        """Get all trainable parameters of the adapter."""
        return list(self.parameters())
        
    def freeze_non_adapter_parameters(self, model: nn.Module):
        """Freeze all parameters except adapter parameters."""
        for name, param in model.named_parameters():
            if 'adapter' not in name.lower():
                param.requires_grad = False
                
    def get_num_parameters(self) -> int:
        """Get the number of trainable parameters in the adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LinearAdapter(BaseAdapter):
    """Simple linear adapter with bottleneck design."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        
        # Down-projection to bottleneck
        self.down_projection = nn.Linear(
            config.hidden_size, 
            config.bottleneck_size, 
            bias=config.bias
        )
        
        # Up-projection back to hidden size
        self.up_projection = nn.Linear(
            config.bottleneck_size, 
            config.hidden_size, 
            bias=config.bias
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize adapter weights."""
        nn.init.xavier_uniform_(self.down_projection.weight)
        nn.init.xavier_uniform_(self.up_projection.weight)
        
        if self.down_projection.bias is not None:
            nn.init.zeros_(self.down_projection.bias)
        if self.up_projection.bias is not None:
            nn.init.zeros_(self.up_projection.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original shape for restoration
        original_shape = x.shape
        
        # Ensure input is 2D (batch_size, hidden_size)
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])
        
        # Pre-norm or post-norm processing
        if self.config.pre_norm:
            x = self.layer_norm(x)
        
        # Down projection
        x = self.down_projection(x)
        
        # Activation
        x = self.config.activation_fn(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Up projection
        x = self.up_projection(x)
        
        # Scale down (residual connection scaling)
        x = x * self.config.scaler
        
        # Post-norm
        if self.config.post_norm:
            x = self.layer_norm(x)
            
        # Restore original shape
        if len(original_shape) > 2:
            x = x.view(*original_shape[:-1], original_shape[-1])
            
        return x


class LoRAAdapter(BaseAdapter):
    """LoRA (Low-Rank Adaptation) adapter."""
    
    def __init__(self, config: AdapterConfig, r: int = 16, lora_alpha: int = 16, lora_dropout: float = 0.1):
        # Update config for LoRA
        config.adapter_size = r
        config.dropout = lora_dropout
        config.bottleneck_size = r
        
        super().__init__(config)
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty((config.hidden_size, r)))
        self.lora_B = nn.Parameter(torch.empty((r, config.hidden_size)))
        
        # Dropout for LoRA
        self.lora_dropout = nn.Dropout(lora_dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize LoRA weights."""
        # Kaiming initialization for A matrix
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Zero initialization for B matrix (so initial LoRA effect is zero)
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save original shape
        original_shape = x.shape
        
        # Ensure input is 2D
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])
        
        # LoRA computation: lora_A @ lora_B @ x
        # Equivalent to: x + lora_scaling * (A @ (B @ x))
        lora_output = torch.matmul(self.lora_A, self.lora_B)
        lora_output = torch.matmul(lora_output, x.t()).t()
        
        # Apply scaling
        lora_output = lora_output * self.scaling
        
        # Restore original shape
        if len(original_shape) > 2:
            lora_output = lora_output.view(*original_shape[:-1], original_shape[-1])
            
        return lora_output


class PrefixTuningAdapter(BaseAdapter):
    """Prefix tuning adapter that prepends learnable context."""
    
    def __init__(self, config: AdapterConfig, num_virtual_tokens: int = 5):
        super().__init__(config)
        
        self.num_virtual_tokens = num_virtual_tokens
        
        # Virtual token embeddings
        self.virtual_token_embeddings = nn.Embedding(
            num_virtual_tokens, 
            config.hidden_size
        )
        
        # Layer normalization for virtual tokens
        self.virtual_tokens_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize virtual token embeddings."""
        nn.init.normal_(self.virtual_token_embeddings.weight, std=0.02)
        
    def get_virtual_tokens(self, batch_size: int) -> torch.Tensor:
        """Get virtual tokens for a batch."""
        device = self.virtual_token_embeddings.weight.device
        
        # Create token indices [0, 1, 2, ..., num_virtual_tokens-1]
        token_indices = torch.arange(
            self.num_virtual_tokens, 
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        virtual_tokens = self.virtual_token_embeddings(token_indices)
        
        # Apply layer norm
        virtual_tokens = self.virtual_tokens_layer_norm(virtual_tokens)
        
        return virtual_tokens
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Get virtual tokens
        virtual_tokens = self.get_virtual_tokens(batch_size)
        
        # Concatenate virtual tokens with input
        prefixed_input = torch.cat([virtual_tokens, x], dim=1)
        
        return prefixed_input


class PTuningAdapter(BaseAdapter):
    """P-Tuning v2 adapter with continuous prompts."""
    
    def __init__(self, config: AdapterConfig, prompt_length: int = 10, pretrain_gpt_mlm: bool = True):
        super().__init__(config)
        
        self.prompt_length = prompt_length
        self.pretrain_gpt_mlm = pretrain_gpt_mlm
        
        # Continuous prompt tokens
        self.prompt_embeddings = nn.Embedding(
            prompt_length, 
            config.hidden_size
        )
        
        # MLP for prompt transformation
        self.prompt_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size)
        )
        
        # Layer norm
        self.prompt_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize P-Tuning weights."""
        nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
        
        # Initialize MLP weights
        for layer in self.prompt_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                    
    def get_prompt_tokens(self) -> torch.Tensor:
        """Get continuous prompt tokens."""
        device = self.prompt_embeddings.weight.device
        
        # Create prompt token indices
        prompt_indices = torch.arange(
            self.prompt_length, 
            device=device
        )
        
        # Get embeddings
        prompt_tokens = self.prompt_embeddings(prompt_indices)
        
        # Apply MLP transformation
        prompt_tokens = self.prompt_mlp(prompt_tokens)
        
        # Apply layer norm
        prompt_tokens = self.prompt_layer_norm(prompt_tokens)
        
        return prompt_tokens
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Get prompt tokens
        prompt_tokens = self.get_prompt_tokens()
        
        # Expand prompt tokens to batch size
        prompt_tokens = prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prompt_tokens


class BitFitAdapter(BaseAdapter):
    """BitFit: Simple, Efficient and Versatile Parameter Efficient Transfer Learning."""
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        
        # Only bias parameters are trainable
        # This is typically applied to existing linear layers
        
    def apply_to_layer(self, layer: nn.Module) -> nn.Module:
        """Apply BitFit to a linear layer by making only biases trainable."""
        if isinstance(layer, nn.Linear):
            # Make all bias parameters trainable
            if layer.bias is not None:
                layer.bias.requires_grad = True
            else:
                # If no bias, create one and make it trainable
                layer.bias = nn.Parameter(torch.zeros(layer.out_features))
                layer.bias.requires_grad = True
                
        return layer
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BitFit doesn't add new parameters, it just modifies existing bias terms
        # This method should be applied to specific layers, not as a standalone forward pass
        return x


class IA3Adapter(BaseAdapter):
    """IA3: Infusion Adapter for General Adaptation."""
    
    def __init__(self, config: AdapterConfig, target_modules: List[str] = None):
        super().__init__(config)
        
        self.target_modules = target_modules or []
        
        # Scale factors for infusion
        self.scale_factors = nn.ParameterDict({})
        
    def add_scale_factors(self, module_name: str, shape: torch.Size):
        """Add scale factors for a module."""
        # Initialize with ones (no scaling initially)
        self.scale_factors[module_name] = nn.Parameter(torch.ones(shape))
        
    def forward(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """Apply IA3 scaling."""
        if module_name in self.scale_factors:
            scale_factor = self.scale_factors[module_name]
            
            # Ensure compatible shapes
            if len(scale_factor.shape) == 1 and scale_factor.size(0) == x.size(-1):
                scale_factor = scale_factor.unsqueeze(0).unsqueeze(0)
            
            # Apply scaling
            x = x * scale_factor
            
        return x


class CompacterAdapter(BaseAdapter):
    """Compacter: Efficiently Adapting Pretrained Language Models via Structured Composable Adapter."""
    
    def __init__(self, config: AdapterConfig, groups: int = 4, share_top: bool = False):
        super().__init__(config)
        
        self.groups = groups
        self.share_top = share_top
        
        # Bottleneck projections
        self.bottleneck_down = nn.Linear(
            config.hidden_size, 
            config.bottleneck_size, 
            bias=False
        )
        
        self.bottleneck_up = nn.Linear(
            config.bottleneck_size, 
            config.hidden_size, 
            bias=False
        )
        
        # Group-specific projections
        self.group_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(groups)
        ])
        
        if self.share_top:
            self.shared_top = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        else:
            self.shared_top = None
            
        # Layer norm
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize Compacter weights."""
        nn.init.xavier_uniform_(self.bottleneck_down.weight)
        nn.init.xavier_uniform_(self.bottleneck_up.weight)
        
        for projection in self.group_projections:
            nn.init.xavier_uniform_(projection.weight)
            
        if self.shared_top is not None:
            nn.init.xavier_uniform_(self.shared_top.weight)
            
    def forward(self, x: torch.Tensor, group_id: int = 0) -> torch.Tensor:
        # Apply group-specific projection
        group_proj = self.group_projections[group_id % self.groups]
        x_group = group_proj(x)
        
        # Bottleneck processing
        x_bottleneck = self.bottleneck_down(x_group)
        x_bottleneck = self.config.activation_fn(x_bottleneck)
        x_bottleneck = self.dropout(x_bottleneck)
        
        # Up projection
        x_up = self.bottleneck_up(x_bottleneck)
        
        # Add to group projection
        x = x_group + x_up
        
        # Shared top layer if specified
        if self.shared_top is not None:
            x = x + self.shared_top(x)
            
        # Apply layer norm
        x = self.layer_norm(x)
        
        return x


class AdapterLayer(BaseAdapter):
    """Layer that can contain multiple adapters."""
    
    def __init__(self, config: AdapterConfig, adapters: Dict[str, BaseAdapter] = None):
        super().__init__(config)
        
        self.adapters = nn.ModuleDict()
        
        if adapters:
            self.adapters.update(adapters)
            
    def add_adapter(self, name: str, adapter: BaseAdapter):
        """Add an adapter to the layer."""
        self.adapters[name] = adapter
        
    def remove_adapter(self, name: str):
        """Remove an adapter from the layer."""
        if name in self.adapters:
            del self.adapters[name]
            
    def get_adapter(self, name: str) -> Optional[BaseAdapter]:
        """Get an adapter by name."""
        return self.adapters.get(name)
        
    def forward(self, x: torch.Tensor, adapter_name: str = None) -> torch.Tensor:
        """Forward pass through specified adapter or all adapters."""
        if adapter_name and adapter_name in self.adapters:
            return self.adapters[adapter_name](x)
        elif adapter_name is None and len(self.adapters) > 0:
            # Apply all adapters and sum
            adapter_outputs = [adapter(x) for adapter in self.adapters.values()]
            return torch.stack(adapter_outputs, dim=-1).sum(dim=-1)
        else:
            return x
            
    def enable_adapter(self, name: str):
        """Enable a specific adapter."""
        if name in self.adapters:
            for param in self.adapters[name].parameters():
                param.requires_grad = True
                
    def disable_adapter(self, name: str):
        """Disable a specific adapter (set requires_grad to False)."""
        if name in self.adapters:
            for param in self.adapters[name].parameters():
                param.requires_grad = False


class MultiModalAdapter(BaseAdapter):
    """Adapter for multi-modal data (text, image, audio)."""
    
    def __init__(self, config: AdapterConfig, 
                 text_adapter: Optional[BaseAdapter] = None,
                 image_adapter: Optional[BaseAdapter] = None,
                 audio_adapter: Optional[BaseAdapter] = None,
                 fusion_method: str = 'attention'):
        super().__init__(config)
        
        # Modality-specific adapters
        self.text_adapter = text_adapter or LinearAdapter(config)
        self.image_adapter = image_adapter or LinearAdapter(config)
        self.audio_adapter = audio_adapter or LinearAdapter(config)
        
        # Fusion method
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=8,
                dropout=config.dropout,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(config.hidden_size)
        elif fusion_method == 'concat':
            self.fusion_projection = nn.Linear(
                config.hidden_size * 3, 
                config.hidden_size
            )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize fusion weights."""
        if self.fusion_method == 'concat' and hasattr(self, 'fusion_projection'):
            nn.init.xavier_uniform_(self.fusion_projection.weight)
            if self.fusion_projection.bias is not None:
                nn.init.zeros_(self.fusion_projection.bias)
                
    def forward(self, text_features: torch.Tensor = None,
                image_features: torch.Tensor = None,
                audio_features: torch.Tensor = None) -> torch.Tensor:
        """Process multi-modal features through adapters."""
        processed_modalities = []
        
        # Process text features
        if text_features is not None:
            text_processed = self.text_adapter(text_features)
            processed_modalities.append(text_processed)
            
        # Process image features
        if image_features is not None:
            image_processed = self.image_adapter(image_features)
            processed_modalities.append(image_processed)
            
        # Process audio features
        if audio_features is not None:
            audio_processed = self.audio_adapter(audio_features)
            processed_modalities.append(audio_processed)
            
        if not processed_modalities:
            return torch.empty(0)
            
        # Fuse modalities
        if len(processed_modalities) == 1:
            return processed_modalities[0]
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            stacked_features = torch.stack(processed_modalities, dim=1)
            
            # Self-attention over modalities
            attended_features, _ = self.attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Apply norm and sum
            attended_features = self.fusion_norm(attended_features)
            fused_features = attended_features.sum(dim=1)
            
            return fused_features
            
        elif self.fusion_method == 'concat':
            # Concatenate and project
            concatenated = torch.cat(processed_modalities, dim=-1)
            fused_features = self.fusion_projection(concatenated)
            
            return fused_features
        else:
            # Default: sum
            return torch.stack(processed_modalities, dim=-1).sum(dim=-1)


class LanguageAdapter(BaseAdapter):
    """Adapter specialized for language model tasks."""
    
    def __init__(self, config: AdapterConfig, 
                 vocab_size: int = 30000,
                 max_position_embeddings: int = 512):
        super().__init__(config)
        
        # Position embeddings adapter
        self.position_embeddings_adapter = nn.Embedding(
            max_position_embeddings, 
            config.hidden_size
        )
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(
            2, 
            config.hidden_size
        )
        
        # Language-specific adapters
        self.language_adapter = LinearAdapter(config)
        self.task_adapter = LinearAdapter(config)
        
        # Layer norm for language features
        self.language_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize language adapter weights."""
        nn.init.normal_(self.position_embeddings_adapter.weight, std=0.02)
        nn.init.normal_(self.token_type_embeddings.weight, std=0.02)
        
    def get_position_embeddings(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Get position embeddings for a sequence."""
        position_indices = torch.arange(
            seq_length, 
            device=device
        )
        
        return self.position_embeddings_adapter(position_indices)
        
    def get_token_type_embeddings(self, token_type_ids: torch.Tensor) -> torch.Tensor:
        """Get token type embeddings."""
        return self.token_type_embeddings(token_type_ids)
        
    def forward(self, x: torch.Tensor, 
                token_type_ids: torch.Tensor = None,
                position_ids: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with language-specific processing."""
        
        # Add position embeddings if provided
        if position_ids is not None:
            position_embeds = self.get_position_embeddings(
                x.size(1), x.device
            )
            x = x + position_embeds
            
        # Add token type embeddings if provided
        if token_type_ids is not None:
            token_type_embeds = self.get_token_type_embeddings(token_type_ids)
            x = x + token_type_embeds
            
        # Apply language-specific processing
        x = self.language_layer_norm(x)
        x = self.language_adapter(x)
        x = self.task_adapter(x)
        
        return x


class VisionAdapter(BaseAdapter):
    """Adapter specialized for vision model tasks."""
    
    def __init__(self, config: AdapterConfig,
                 image_size: int = 224,
                 patch_size: int = 16,
                 num_channels: int = 3):
        super().__init__(config)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding adapter
        self.patch_embedding = nn.Conv2d(
            num_channels, config.hidden_size, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Learnable patch embeddings
        self.patch_embeddings = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02
        )
        
        # Class token
        self.class_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size) * 0.02
        )
        
        # Vision-specific adapters
        self.vision_adapter = LinearAdapter(config)
        self.spatial_adapter = LinearAdapter(config)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            self.num_patches + 1, 
            config.hidden_size
        )
        
        # Layer norm
        self.vision_layer_norm = nn.LayerNorm(
            config.hidden_size, 
            eps=config.layer_norm_epsilon
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize vision adapter weights."""
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)
            
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
        
    def create_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Create patches from input images."""
        batch_size = images.size(0)
        
        # Get patch embeddings
        patch_embeds = self.patch_embedding(images)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        
        # Add learnable patch embeddings
        patch_embeds = patch_embeds + self.patch_embeddings
        
        return patch_embeds
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with vision-specific processing."""
        batch_size = x.size(0)
        
        # Create patches
        patch_embeds = self.create_patches(x)
        
        # Add class token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat([class_tokens, patch_embeds], dim=1)
        
        # Add position embeddings
        position_ids = torch.arange(
            tokens.size(1), device=x.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        tokens = tokens + self.position_embeddings(position_ids)
        
        # Apply vision-specific processing
        tokens = self.vision_layer_norm(tokens)
        tokens = self.vision_adapter(tokens)
        tokens = self.spatial_adapter(tokens)
        
        return tokens


class AdapterModel(nn.Module):
    """Model wrapper for adapter-based parameter-efficient fine-tuning."""
    
    def __init__(self, base_model: nn.Module, config: AdapterConfig):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.adapters = nn.ModuleDict()
        
        # Add default adapter if specified
        if config.adapter_size > 0:
            self.add_adapter('default')
            
    def add_adapter(self, name: str, adapter: Optional[BaseAdapter] = None):
        """Add an adapter to the model."""
        if adapter is None:
            adapter = LinearAdapter(self.config)
            
        self.adapters[name] = adapter
        
    def remove_adapter(self, name: str):
        """Remove an adapter from the model."""
        if name in self.adapters:
            del self.adapters[name]
            
    def enable_adapters(self, names: List[str] = None):
        """Enable specific adapters (set requires_grad to True)."""
        adapters_to_enable = names or list(self.adapters.keys())
        
        for name in adapters_to_enable:
            if name in self.adapters:
                for param in self.adapters[name].parameters():
                    param.requires_grad = True
                    
    def disable_adapters(self, names: List[str] = None):
        """Disable specific adapters (set requires_grad to False)."""
        adapters_to_disable = names or list(self.adapters.keys())
        
        for name in adapters_to_disable:
            if name in self.adapters:
                for param in self.adapters[name].parameters():
                    param.requires_grad = False
                    
    def freeze_base_model(self):
        """Freeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        """Unfreeze the base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = True
            
    def get_num_adapter_parameters(self) -> Dict[str, int]:
        """Get number of parameters for each adapter."""
        counts = {}
        for name, adapter in self.adapters.items():
            counts[name] = adapter.get_num_parameters()
        return counts
        
    def get_total_parameters(self) -> Tuple[int, int]:
        """Get total and trainable parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, trainable_params
        
    def forward(self, *args, **kwargs):
        """Forward pass through base model with adapter processing."""
        # Get base model output
        base_output = self.base_model(*args, **kwargs)
        
        # Apply adapters if they exist
        if len(self.adapters) > 0:
            # This is a simplified approach - in practice, you'd need to integrate
            # adapters at specific points in the base model architecture
            adapter_outputs = []
            for adapter in self.adapters.values():
                if isinstance(base_output, torch.Tensor):
                    adapter_output = adapter(base_output)
                    adapter_outputs.append(adapter_output)
                    
            if adapter_outputs:
                # Combine adapter outputs with base output
                combined_output = base_output + sum(adapter_outputs)
                return combined_output
                
        return base_output