"""
Language Processing Module for mini-biai-1 - SSM/Linear-Attention Upgrade

This module provides comprehensive text processing capabilities with SSM/linear-attention
backbone, transformer-based embeddings, and spiking neural networks. The upgraded module
implements a hybrid architecture combining State Space Models and linear attention for
optimal sequence processing efficiency.

The module implements a modular architecture supporting:
- SSM-based text encoding with O(N) complexity
- Linear attention mechanisms for adaptive processing
- Spiking output layers for biological realism
- Memory integration for persistent text representations
- Hardware-aware optimization (CUDA, MPS, CPU fallbacks)
- Performance monitoring and adaptive mechanisms
- Comprehensive error handling with graceful fallbacks

Core Components (SSM/Linear-Attention Upgrade):
    SSMBackbone: State Space Model implementation for efficient sequence processing
    LinearAttentionMechanism: Linear attention with multiple variants
    HybridProcessor: Combined SSM-attention architecture with adaptive processing
    SSMTextProcessor: Advanced SSM-based text processor with integrated features
    SpikingOutputLayer: Biological spiking neuron integration
    MemoryGate: Memory integration with gating mechanisms
    AdaptiveProcessingController: Dynamic switching between processing modes

Architecture:
    ┌─────────────────────────────────────────────┐
    │              Input Text                     │
    └─────────────────────┬───────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │             Token Embedding                 │
    └─────────────────────┬───────────────────────┘
                         │
                         ▼
    ┌─────────────────────┬───────────────────────┐
    │   SSM Backbone      │  Linear Attention     │
    │  (O(N) complexity)  │   (Adaptive focus)    │
    └─────────┬───────────┴───────────┬───────────┘
              │                       │
              ▼                       ▼
    ┌─────────────────────┬───────────────────────┐
    │   Spiking Layer     │   Memory Integration   │
    │ (Biological realism)│   (Persistent state)   │
    └─────────┬───────────┴───────────┬───────────┘
              │                       │
              ▼                       ▼
    ┌─────────────────────────────────────────────┐
    │        Hybrid Output & Integration          │
    └─────────────────────────────────────────────┘

Processing Modes:
    1. SEQUENTIAL: SSM → Attention (cascaded processing)
    2. PARALLEL: Concurrent SSM and attention
    3. ADAPTIVE: Dynamic switching based on input characteristics
    4. WEIGHTED: Fixed combination of SSM and attention
    5. CASCADING: Multi-stage processing with feedback

Key Features:
    - Modular design with replaceable implementations
    - Hardware-aware optimization (CPU/CUDA/MPS)
    - Memory integration for persistent text representations
    - Async/await support for concurrent processing
    - Batch processing for improved throughput
    - Comprehensive similarity calculations
    - Fallback mechanisms for missing dependencies

Version: 2.0.0
Author: mini-biai-1 Team
License: MIT
"""

# SSM/Linear-Attention Components (with fallback support)
try:
    from .hybrid_processor import (
        HybridProcessor,
        HybridProcessorConfig,
        HybridProcessingMode,
        ProcessingModeRouter,
        AdaptiveProcessingController,
        SequenceAnalyzer,
        MemoryGate,
        SpikingProcessorState,
        StandardAttentionFallback,
        HybridPerformanceMonitor,
        create_hybrid_processor,
    )
    
    # Re-export fallback classes for compatibility
    from .hybrid_processor import SSMConfig, SSMType, SSMBackbone, SSMHardwareType
    from .hybrid_processor import SpikingOutputLayer, MultiHeadLinearAttention, LinearAttentionConfig, LinearAttentionType
    
    # Create minimal performance monitor and creator functions as fallbacks
    class SSMPerformanceMonitor:
        def __init__(self):
            pass
        def record_processing_time(self, time_ms):
            pass
    
    class PerformerAttention:
        def __init__(self, config):
            self.config = config
    
    class LinearTransformerAttention:
        def __init__(self, config):
            self.config = config
    
    class SlidingWindowAttention:
        def __init__(self, config):
            self.config = config
    
    class SpikingLinearAttention:
        def __init__(self, config):
            self.config = config
    
    def create_ssm_backbone(config):
        return SSMBackbone(config)
    
    def create_linear_attention(config):
        return MultiHeadLinearAttention(config)

except ImportError as e:
    print(f"Warning: Could not import Step 2 components: {e}")

# Import legacy components (always available)
from .hybrid_processor import SSMConfig, SSMType, SSMBackbone, SSMHardwareType
from .hybrid_processor import SpikingOutputLayer, MultiHeadLinearAttention, LinearAttentionConfig, LinearAttentionType

from .ssm_text_processor import (
    SSMTextProcessor,
    SSMTextConfig,
    SSMTextEncoding,
    TextEncodingType,
    create_ssm_text_processor,
)

# Legacy Components (Maintained for Compatibility)
from .linear_text import (
    # Core interfaces
    TextProcessorInterface,
    MemoryIntegrationInterface,
    
    # Concrete implementations
    LinearTextProcessor,
    LinearTextProcessorWithMemory,
    SSMTextProcessor as LegacySSMTextProcessor,  # Keep original for compatibility
    
    # Data structures
    ProcessingConfig,
    TextEncoding,
    HardwareType,
    
    # Utility functions
    create_text_processor,
    validate_text_input,
    calculate_similarity,
)

__all__ = [
    # SSM/Linear-Attention Components
    "SSMBackbone",
    "SSMConfig", 
    "SSMType",
    "SSMHardwareType",
    "SpikingOutputLayer",
    "SSMPerformanceMonitor",
    "create_ssm_backbone",
    
    # Linear Attention Components
    "MultiHeadLinearAttention",
    "LinearAttentionConfig",
    "LinearAttentionType",
    "PerformerAttention",
    "LinearTransformerAttention", 
    "SlidingWindowAttention",
    "SpikingLinearAttention",
    "create_linear_attention",
    
    # Hybrid Processor Components
    "HybridProcessor",
    "HybridProcessorConfig",
    "HybridProcessingMode",
    "ProcessingModeRouter",
    "AdaptiveProcessingController",
    "SequenceAnalyzer",
    "MemoryGate",
    "SpikingProcessorState",
    "StandardAttentionFallback",
    "HybridPerformanceMonitor",
    "create_hybrid_processor",
    
    # SSM Text Processor Components
    "SSMTextProcessor",
    "SSMTextConfig",
    "SSMTextEncoding",
    "TextEncodingType",
    "create_ssm_text_processor",
    
    # Legacy Components (Maintained for Compatibility)
    "TextProcessorInterface",
    "MemoryIntegrationInterface", 
    "LinearTextProcessor",
    "LinearTextProcessorWithMemory",
    "LegacySSMTextProcessor",  # Original placeholder
    "ProcessingConfig",
    "TextEncoding",
    "HardwareType",
    "create_text_processor",
    "validate_text_input",
    "calculate_similarity",
]

__version__ = "2.0.0"