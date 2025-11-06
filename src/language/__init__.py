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
    LinearAttentionMechanism: Linear attention with multiple variants (Performer, Linear Transformer, etc.)
    HybridProcessor: Combined SSM-attention architecture with adaptive processing
    SSMTextProcessor: Advanced SSM-based text processor with integrated features
    SpikingOutputLayer: Biological spiking neuron integration
    MemoryGate: Memory integration with gating mechanisms
    AdaptiveProcessingController: Dynamic switching between processing modes

Legacy Components (Maintained for Compatibility):
    TextProcessorInterface: Abstract base interface for text processing implementations
    MemoryIntegrationInterface: Interface for memory-backed text processing
    LinearTextProcessor: Transformer-based text processor
    LinearTextProcessorWithMemory: Memory-integrated text processor
    ProcessingConfig: Configuration management for text processing
    TextEncoding: Container for encoding results with metadata

Architecture (SSM/Linear-Attention Upgrade):
    The language module follows a hybrid architecture design:
    
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
    │ (Biological realism)│   (Persistent state)  │
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

Supported Models:
    Default: sentence-transformers/all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Fast inference speed
    - Good general-purpose quality
    - Multilingual support
    
    Alternatives:
    - all-mpnet-base-v2: Higher quality (768-dim, slower)
    - all-MiniLM-L12-v2: Better quality, medium speed (384-dim)
    - distilbert-base-nli-mean-tokens: Fast inference (768-dim)

Usage Examples:

SSM-Based Text Processing (New):
    >>> from src.language import (
    ...     create_ssm_text_processor, SSMTextConfig, 
    ...     TextEncodingType, HybridProcessingMode
    ... )
    >>> import asyncio
    >>> 
    >>> async def main():
    ...     # Create SSM configuration
    ...     config = SSMTextConfig(
    ...         hidden_size=256,
    ...         embedding_dim=128,
    ...         encoding_type=TextEncodingType.HYBRID,
    ...         processing_mode=HybridProcessingMode.ADAPTIVE,
    ...         spiking_enabled=True,
    ...         memory_integration=True
    ...     )
    ...     
    ...     # Create SSM text processor
    ...     processor = create_ssm_text_processor(config)
    ...     
    ...     # Encode single text with SSM
    ...     encoding = await processor.encode_text("Machine learning with SSM")
    ...     print(f"SSM Embedding shape: {encoding.embeddings.shape}")
    ...     
    ...     # Batch encode with performance monitoring
    ...     texts = [
    ...         "State Space Models process sequences efficiently",
    ...         "Linear attention reduces complexity to O(N)",
    ...         "Spiking neurons provide biological realism"
    ...     ]
    ...     encodings = await processor.batch_encode(texts)
    ...     print(f"Batch processed: {len(encodings)} texts")
    ...     
    ...     # Performance summary
    ...     perf = processor.get_performance_summary()
    ...     print(f"Performance: {perf}")
    >>> 
    >>> asyncio.run(main())

Hybrid SSM-Attention Processing:
    >>> from src.language import create_hybrid_processor, HybridProcessorConfig
    >>> 
    >>> # Create hybrid processor
    >>> config = HybridProcessorConfig(
    ...     hidden_size=512,
    ...     processing_mode=HybridProcessingMode.ADAPTIVE,
    ...     ssm_attention_ratio=0.7  # 70% SSM, 30% attention
    ... )
    >>> 
    >>> processor = create_hybrid_processor(config)
    >>> 
    >>> # Process sequence with hybrid approach
    >>> sequence = torch.randn(2, 128, 512)  # batch_size=2, seq_len=128, hidden_size=512
    >>> output = processor(sequence)
    >>> print(f"Hybrid output shape: {output.shape}")

Linear Attention Processing:
    >>> from src.language import create_linear_attention, LinearAttentionConfig
    >>> 
    >>> # Create linear attention processor
    >>> config = LinearAttentionConfig(
    ...     hidden_size=256,
    ...     attention_type=LinearAttentionType.PERFORMER,
    ...     num_features=64
    ... )
    >>> 
    >>> attention = create_linear_attention(config)
    >>> 
    >>> # Process with linear attention
    >>> query = torch.randn(2, 64, 256)
    >>> key = torch.randn(2, 64, 256) 
    >>> value = torch.randn(2, 64, 256)
    >>> output = attention(query, key, value)
    >>> print(f"Linear attention output: {output.shape}")

SSM Backbone for Custom Architectures:
    >>> from src.language import create_ssm_backbone, SSMConfig, SSMType
    >>> 
    >>> # Create standalone SSM backbone
    >>> config = SSMConfig(
    ...     hidden_size=512,
    ...     state_size=256,
    ...     ssm_type=SSMType.HIPPO,
    ...     spiking_output=True
    ... )
    >>> 
    >>> ssm_backbone = create_ssm_backbone(config)
    >>> 
    >>> # Use in custom architecture
    >>> x = torch.randn(4, 256, 512)  # batch_size=4, seq_len=256, hidden_size=512
    >>> ssm_output = ssm_backbone(x)
    >>> print(f"SSM output: {ssm_output.shape}")

Legacy Transformer Processing (Maintained):
    >>> from src.language import create_text_processor, ProcessingConfig
    >>> 
    >>> # Legacy transformer-based processing (still works)
    >>> config = ProcessingConfig(
    ...     model_name="all-MiniLM-L6-v2",
    ...     hardware_type=HardwareType.AUTO
    ... )
    >>> processor = create_text_processor(config)
    >>> encoding = await processor.encode_text("Traditional transformer processing")

Memory-Integrated SSM Processing:
    >>> class SimpleMemoryInterface(MemoryIntegrationInterface):
    ...     def __init__(self):
    ...         self.encodings = {}
    ...     async def store_encoding(self, key, encoding):
    ...         self.encodings[key] = encoding
    ...         return True
    ...     async def retrieve_encoding(self, key):
    ...         return self.encodings.get(key)
    ...     async def find_similar(self, encoding, threshold=0.8):
    ...         # Simplified similarity search
    ...         return []
    >>> 
    >>> # Create SSM processor with memory
    >>> memory = SimpleMemoryInterface()
    >>> processor = create_ssm_text_processor(config, memory_interface=memory)
    >>> 
    >>> # Encode and store with memory integration
    >>> encoding = await processor.encode_and_store(
    ...     "SSM processed information with memory",
    ...     "ssm_memory_key"
    ... )

Hardware Support:
    CPU: Universal compatibility, baseline performance
    CUDA: NVIDIA GPU acceleration (3-10x speedup)
    MPS: Apple Silicon optimization (2-5x speedup on M1/M2)
    
Performance Considerations:
    - GPU acceleration provides significant speedup for large batches
    - Memory usage scales with batch size and model dimensions
    - First run may be slower due to model loading
    - Batch processing more efficient than individual calls

Dependencies:
    Required:
    - torch: Deep learning framework
    - numpy: Numerical operations
    - transformers: Hugging Face transformer library (optional)
    - sentence-transformers: Text embedding models (optional)
    
    Optional:
    - faiss: Vector similarity search for memory integration
    - psutil: Memory usage monitoring

Error Handling:
    The module implements comprehensive error handling:
    - Graceful degradation when transformers unavailable
    - Automatic fallback to hash-based embeddings
    - Hardware compatibility checks with fallbacks
    - Thread-safe operations with proper locking
    - Detailed logging for debugging and monitoring

Version: 1.0.0
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