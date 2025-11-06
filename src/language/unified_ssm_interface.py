#!/usr/bin/env python3
"""
SSM/Linear-Attention Unified Interface

This module provides a unified interface for all SSM/Linear-Attention components,
making it easy to switch between different implementations and configurations.

Features:
- Unified API for all SSM variants (Traditional, Mamba-style)
- Linear attention mechanisms (Performer, Linear Transformer, etc.)
- Hybrid processing architectures
- Hardware-aware optimization
- Production-ready deployment interface

Author: mini-biai-1 Team
License: MIT
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

# Import all components with fallbacks
try:
    from .ssm_backbone import SSMBackbone, SSMConfig, SSMType, HardwareType, create_ssm_backbone
    SSM_BACKBONE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"SSM Backbone not available: {e}")
    SSM_BACKBONE_AVAILABLE = False

try:
    from .linear_attention import (
        MultiHeadLinearAttention, LinearAttentionConfig,
        LinearAttentionType, create_linear_attention
    )
    LINEAR_ATTENTION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Linear Attention not available: {e}")
    LINEAR_ATTENTION_AVAILABLE = False

try:
    from .hybrid_processor import (
        HybridProcessor, HybridProcessorConfig,
        HybridProcessingMode, create_hybrid_processor
    )
    HYBRID_PROCESSOR_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hybrid Processor not available: {e}")
    HYBRID_PROCESSOR_AVAILABLE = False

try:
    from .mamba_ssm import (
        MambaSSMBackbone, MambaSSMConfig, create_mamba_ssm
    )
    MAMBA_SSM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Mamba SSM not available: {e}")
    MAMBA_SSM_AVAILABLE = False


class SSMImplementationType(Enum):
    """Types of SSM implementations"""
    TRADITIONAL = "traditional"
    MAMBA = "mamba"
    HYBRID = "hybrid"


class ModelArchitecture(Enum):
    """Supported model architectures"""
    SSM_ONLY = "ssm_only"
    ATTENTION_ONLY = "attention_only"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_ADAPTIVE = "hybrid_adaptive"
    CASCADING = "cascading"


@dataclass
class UnifiedConfig:
    """Unified configuration for all SSM/Lattention components"""
    
    # Model architecture
    hidden_size: int = 512
    sequence_length: int = 512
    num_layers: int = 6
    
    # Implementation choices
    architecture: ModelArchitecture = ModelArchitecture.HYBRID_ADAPTIVE
    ssm_type: SSMImplementationType = SSMImplementationType.MAMBA
    
    # SSM parameters
    ssm_state_size: int = 256
    ssm_expand_factor: int = 2
    
    # Attention parameters
    num_attention_heads: int = 8
    attention_type: str = "performer"
    num_features: int = 64
    
    # Spiking integration
    spiking_enabled: bool = True
    spike_threshold: float = 0.5
    spike_decay: float = 0.95
    
    # Hardware and performance
    hardware_type: str = "auto"
    use_mixed_precision: bool = True
    memory_efficient: bool = True
    chunked_processing: bool = True
    
    # Memory integration
    memory_integration: bool = True
    external_memory_size: int = 64
    context_window_size: int = 128
    
    # Adaptive processing
    adaptive_processing: bool = True
    performance_threshold: float = 0.8
    
    # Monitoring and profiling
    enable_profiling: bool = True
    enable_performance_monitoring: bool = True


class UnifiedSSMInterface:
    """
    Unified interface for all SSM/Linear-Attention components
    
    This class provides a single interface to access and switch between
    different SSM and attention implementations seamlessly.
    """
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = self._setup_device()
        self.logger.info(f"Unified SSM Interface initialized on {self.device}")
        
        # Initialize components
        self._initialize_components()
        
        # Performance monitoring
        self.performance_history = []
        self.model_stats = {}
        
    def _setup_device(self) -> torch.device:
        """Setup optimal device based on configuration and availability"""
        hardware_type = self.config.hardware_type.lower()
        
        if hardware_type == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        elif hardware_type == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif hardware_type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _initialize_components(self):
        """Initialize all SSM and attention components"""
        # Initialize SSM component
        self.ssm_component = self._create_ssm_component()
        
        # Initialize attention component
        self.attention_component = self._create_attention_component()
        
        # Initialize hybrid processor if needed
        if self.config.architecture in [
            ModelArchitecture.HYBRID_SEQUENTIAL,
            ModelArchitecture.HYBRID_PARALLEL,
            ModelArchitecture.HYBRID_ADAPTIVE,
            ModelArchitecture.CASCADING
        ]:
            self.hybrid_component = self._create_hybrid_component()
        else:
            self.hybrid_component = None
        
        # Final processing layers
        self.output_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Move to device
        self._move_to_device()
        
        self.logger.info("All components initialized successfully")
    
    def _create_ssm_component(self):
        """Create SSM component based on configuration"""
        if self.config.ssm_type == SSMImplementationType.TRADITIONAL and SSM_BACKBONE_AVAILABLE:
            config = SSMConfig(
                hidden_size=self.config.hidden_size,
                state_size=self.config.ssm_state_size,
                num_layers=self.config.num_layers,
                ssm_type=SSMType.HIPPO,
                hardware_type=HardwareType.CUDA if self.device.type == 'cuda' else HardwareType.CPU,
                spiking_output=self.config.spiking_enabled,
                spike_threshold=self.config.spike_threshold,
                spike_decay=self.config.spike_decay
            )
            return create_ssm_backbone(config)
            
        elif self.config.ssm_type == SSMImplementationType.MAMBA and MAMBA_SSM_AVAILABLE:
            from .mamba_ssm import MambaActivationType
            config = MambaSSMConfig(
                hidden_size=self.config.hidden_size,
                state_size=self.config.ssm_state_size,
                num_layers=self.config.num_layers,
                expand_factor=self.config.ssm_expand_factor,
                spiking_output=self.config.spiking_enabled,
                spike_threshold=self.config.spike_threshold,
                spike_decay=self.config.spike_decay
            )
            return create_mamba_ssm(config)
        
        elif HYBRID_PROCESSOR_AVAILABLE:
            # Fallback to hybrid processor for SSM-like functionality
            config = HybridProcessorConfig(
                hidden_size=self.config.hidden_size,
                sequence_length=self.config.sequence_length,
                num_layers=self.config.num_layers,
                processing_mode=HybridProcessingMode.WEIGHTED,
                ssm_attention_ratio=1.0,  # Focus on SSM
                hardware_type=HardwareType.CUDA if self.device.type == 'cuda' else HardwareType.CPU,
                spiking_enabled=self.config.spiking_enabled
            )
            return create_hybrid_processor(config)
        
        else:
            # Minimal fallback
            self.logger.warning("No SSM component available, using minimal fallback")
            return MinimalSSMFallback(self.config.hidden_size)
    
    def _create_attention_component(self):
        """Create attention component based on configuration"""
        if not LINEAR_ATTENTION_AVAILABLE:
            self.logger.warning("Linear attention not available, using fallback")
            return MinimalAttentionFallback(self.config.hidden_size)
        
        # Map string attention type to enum
        attention_type_map = {
            "performer": LinearAttentionType.PERFORMER,
            "linear_transformer": LinearAttentionType.LINEAR_TRANSFORMER,
            "sliding_window": LinearAttentionType.SLIDING_WINDOW,
            "spiking_linear": LinearAttentionType.SPIKING_LINEAR
        }
        
        attention_type = attention_type_map.get(
            self.config.attention_type.lower(),
            LinearAttentionType.PERFORMER
        )
        
        config = LinearAttentionConfig(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_type=attention_type,
            num_features=self.config.num_features,
            spiking=self.config.spiking_enabled,
            spike_threshold=self.config.spike_threshold
        )
        
        return create_linear_attention(config)
    
    def _create_hybrid_component(self):
        """Create hybrid component"""
        if not HYBRID_PROCESSOR_AVAILABLE:
            return None
        
        # Map architecture to processing mode
        mode_map = {
            ModelArchitecture.HYBRID_SEQUENTIAL: HybridProcessingMode.SEQUENTIAL,
            ModelArchitecture.HYBRID_PARALLEL: HybridProcessingMode.PARALLEL,
            ModelArchitecture.HYBRID_ADAPTIVE: HybridProcessingMode.ADAPTIVE,
            ModelArchitecture.CASCADING: HybridProcessingMode.CASCADING
        }
        
        processing_mode = mode_map.get(
            self.config.architecture,
            HybridProcessingMode.ADAPTIVE
        )
        
        config = HybridProcessorConfig(
            hidden_size=self.config.hidden_size,
            sequence_length=self.config.sequence_length,
            num_layers=self.config.num_layers,
            num_attention_heads=self.config.num_attention_heads,
            processing_mode=processing_mode,
            ssm_attention_ratio=0.5,
            hardware_type=HardwareType.CUDA if self.device.type == 'cuda' else HardwareType.CPU,
            spiking_enabled=self.config.spiking_enabled,
            memory_integration=self.config.memory_integration,
            adaptive_switching=self.config.adaptive_processing
        )
        
        return create_hybrid_processor(config)
    
    def _move_to_device(self):
        """Move all components to the appropriate device"""
        if hasattr(self.ssm_component, 'to'):
            self.ssm_component.to(self.device)
        if hasattr(self.attention_component, 'to'):
            self.attention_component.to(self.device)
        if self.hybrid_component and hasattr(self.hybrid_component, 'to'):
            self.hybrid_component.to(self.device)
        if hasattr(self.output_projection, 'to'):
            self.output_projection.to(self.device)
        if hasattr(self.layer_norm, 'to'):
            self.layer_norm.to(self.device)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Unified forward pass through the selected architecture
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            **kwargs: Additional arguments (external_memory, context_window, etc.)
            
        Returns:
            Processed output tensor
        """
        start_time = time.time()
        
        try:
            # Ensure input is on correct device
            x = x.to(self.device)
            
            # Choose processing path based on architecture
            if self.config.architecture == ModelArchitecture.SSM_ONLY:
                output = self._process_ssm_only(x, **kwargs)
            elif self.config.architecture == ModelArchitecture.ATTENTION_ONLY:
                output = self._process_attention_only(x, **kwargs)
            elif self.hybrid_component:
                output = self._process_hybrid(x, **kwargs)
            else:
                # Fallback to simple combination
                output = self._process_fallback(x, **kwargs)
            
            # Apply final processing
            output = self.output_projection(output)
            output = self.layer_norm(output)
            output = self.dropout(output)
            
            # Update performance tracking
            if self.config.enable_performance_monitoring:
                forward_time = time.time() - start_time
                self._update_performance_stats(forward_time, x.shape)
            
            return output
            
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            # Fallback to simple processing
            return self._simple_forward(x)
    
    def _process_ssm_only(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process using only SSM component"""
        return self.ssm_component(x, **kwargs)
    
    def _process_attention_only(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process using only attention component"""
        return self.attention_component(x, x, x, **kwargs)
    
    def _process_hybrid(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Process using hybrid component"""
        return self.hybrid_component(x, **kwargs)
    
    def _process_fallback(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Fallback processing combining SSM and attention"""
        ssm_output = self.ssm_component(x, **kwargs)
        attention_output = self.attention_component(x, x, x, **kwargs)
        
        # Simple combination
        combined = 0.5 * ssm_output + 0.5 * attention_output
        return combined
    
    def _simple_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple fallback forward pass"""
        # Just apply layer norm and projection
        output = self.layer_norm(x)
        output = self.output_projection(output)
        return output
    
    def _update_performance_stats(self, forward_time: float, input_shape: tuple):
        """Update performance statistics"""
        stats = {
            'timestamp': time.time(),
            'forward_time': forward_time,
            'input_shape': input_shape,
            'device': str(self.device)
        }
        
        self.performance_history.append(stats)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {}
        
        forward_times = [s['forward_time'] for s in self.performance_history]
        
        summary = {
            'avg_forward_time': sum(forward_times) / len(forward_times),
            'min_forward_time': min(forward_times),
            'max_forward_time': max(forward_times),
            'total_inferences': len(self.performance_history),
            'current_device': str(self.device),
            'architecture': self.config.architecture.value,
            'ssm_type': self.config.ssm_type.value
        }
        
        # Component-specific summaries
        if hasattr(self.ssm_component, 'get_performance_summary'):
            summary['ssm_component'] = self.ssm_component.get_performance_summary()
        
        if hasattr(self.attention_component, 'get_performance_summary'):
            summary['attention_component'] = self.attention_component.get_performance_summary()
        
        if self.hybrid_component and hasattr(self.hybrid_component, 'get_performance_summary'):
            summary['hybrid_component'] = self.hybrid_component.get_performance_summary()
        
        return summary
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        info = {
            'architecture': {
                'type': self.config.architecture.value,
                'ssm_type': self.config.ssm_type.value,
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers
            },
            'components': {
                'ssm_available': self.ssm_component is not None,
                'attention_available': self.attention_component is not None,
                'hybrid_available': self.hybrid_component is not None
            },
            'hardware': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            },
            'features': {
                'spiking_enabled': self.config.spiking_enabled,
                'memory_integration': self.config.memory_integration,
                'adaptive_processing': self.config.adaptive_processing,
                'mixed_precision': self.config.use_mixed_precision
            }
        }
        
        return info
    
    def switch_architecture(self, new_architecture: ModelArchitecture):
        """Switch to a different architecture at runtime"""
        self.config.architecture = new_architecture
        self.logger.info(f"Switching to architecture: {new_architecture.value}")
        
        # Re-initialize components for new architecture
        self._initialize_components()
    
    def export_config(self, filepath: str):
        """Export configuration to file"""
        config_dict = {
            'hidden_size': self.config.hidden_size,
            'sequence_length': self.config.sequence_length,
            'num_layers': self.config.num_layers,
            'architecture': self.config.architecture.value,
            'ssm_type': self.config.ssm_type.value,
            'ssm_state_size': self.config.ssm_state_size,
            'num_attention_heads': self.config.num_attention_heads,
            'attention_type': self.config.attention_type,
            'spiking_enabled': self.config.spiking_enabled,
            'hardware_type': self.config.hardware_type,
            'memory_integration': self.config.memory_integration,
            'adaptive_processing': self.config.adaptive_processing
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration exported to: {filepath}")


# Fallback implementations for missing components
class MinimalSSMFallback(nn.Module):
    """Minimal SSM fallback when main implementation is not available"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.norm(self.linear(x))


class MinimalAttentionFallback(nn.Module):
    """Minimal attention fallback when linear attention is not available"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, query, key, value):
        output, _ = self.attention(query, key, value)
        return self.norm(output)


# Factory function
def create_unified_ssm(config: UnifiedConfig = None) -> UnifiedSSMInterface:
    """Create unified SSM interface"""
    if config is None:
        config = UnifiedConfig()
    
    return UnifiedSSMInterface(config)


# Predefined configurations
def get_production_config() -> UnifiedConfig:
    """Get production-ready configuration"""
    return UnifiedConfig(
        hidden_size=512,
        num_layers=12,
        architecture=ModelArchitecture.HYBRID_ADAPTIVE,
        ssm_type=SSMImplementationType.MAMBA,
        spiking_enabled=True,
        memory_integration=True,
        adaptive_processing=True,
        hardware_type="auto",
        enable_performance_monitoring=True
    )


def get_benchmark_config() -> UnifiedConfig:
    """Get configuration optimized for benchmarking"""
    return UnifiedConfig(
        hidden_size=256,
        num_layers=6,
        architecture=ModelArchitecture.HYBRID_PARALLEL,
        ssm_type=SSMImplementationType.MAMBA,
        spiking_enabled=False,  # Disable for pure performance testing
        memory_integration=False,
        adaptive_processing=False,
        hardware_type="auto",
        enable_performance_monitoring=True
    )


def get_memory_efficient_config() -> UnifiedConfig:
    """Get configuration optimized for memory efficiency"""
    return UnifiedConfig(
        hidden_size=256,
        num_layers=6,
        architecture=ModelArchitecture.SSM_ONLY,
        ssm_type=SSMImplementationType.MAMBA,
        spiking_enabled=True,
        memory_integration=False,
        adaptive_processing=False,
        hardware_type="auto",
        memory_efficient=True,
        chunked_processing=True
    )


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def demo_unified_interface():
        """Demonstrate unified interface functionality"""
        logging.basicConfig(level=logging.INFO)
        
        print("🚀 Unified SSM/Linear-Attention Interface Demo")
        print("=" * 60)
        
        # Test different configurations
        configs = [
            ("Production", get_production_config()),
            ("Benchmark", get_benchmark_config()),
            ("Memory Efficient", get_memory_efficient_config())
        ]
        
        for config_name, config in configs:
            print(f"\n🔧 Testing {config_name} Configuration")
            print("-" * 40)
            
            # Create unified interface
            unified = create_unified_ssm(config)
            
            # Test forward pass
            batch_size, seq_len = 2, 64
            test_input = torch.randn(batch_size, seq_len, config.hidden_size)
            
            print(f"Device: {unified.device}")
            print(f"Architecture: {config.architecture.value}")
            print(f"SSM Type: {config.ssm_type.value}")
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                output = unified(test_input)
                forward_time = time.time() - start_time
            
            print(f"Input shape: {test_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Forward time: {forward_time:.4f}s")
            
            # Model info
            model_info = unified.get_model_info()
            print(f"Components: SSM={model_info['components']['ssm_available']}, "
                  f"Attention={model_info['components']['attention_available']}, "
                  f"Hybrid={model_info['components']['hybrid_available']}")
        
        print("\n🎉 Unified Interface Demo Complete!")
    
    # Run demo
    asyncio.run(demo_unified_interface())