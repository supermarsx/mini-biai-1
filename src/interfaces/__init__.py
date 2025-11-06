"""
Core Interfaces for mini-biai-1 System

This module provides comprehensive interfaces and data structures that define
the communication contracts between all components in the brain-inspired
modular AI system. The interface system ensures proper separation of concerns,
type safety, and compatibility across all modules with enhanced capabilities.

The interface hierarchy includes:

Communication Layer:
    - Request: Standard request structure for system interactions
    - Reply: Response structure for system feedback
    - RouteDecision: Routing decisions from spiking neural networks

Memory System Interfaces:
    - STMState: Short-term memory state representation
    - Retrieval: Long-term memory retrieval results
    - EmbeddingVector: Vector representation for similarity operations

Audio Processing Interfaces:
    - AudioData: Core audio data structure
    - SpeechRecognition: ASR interface with transcript and confidence
    - TextToSpeech: TTS interface with audio output
    - AudioClassification: Audio classification results
    - MusicAnalysis: Music-specific analysis results

Multimodal Interfaces:
    - MultimodalInput: Unified multimodal input structure
    - CrossModalAttention: Cross-modal attention weights
    - ModalityFusion: Feature fusion operations
    - UnifiedRepresentation: Combined multimodal embeddings

Data Collection Interfaces:
    - SocialMediaPost: Social media content structure
    - IoTSensorData: IoT sensor reading interface
    - StreamingData: Real-time streaming data interface
    - CollectionMetrics: Data collection performance metrics

Optimization Interfaces:
    - FederatedClient: Federated learning client interface
    - ArchitectureSearchSpace: Neural architecture search space
    - OptimizationObjective: Multi-objective optimization
    - ModelCompression: Model compression parameters

Enhanced Memory Interfaces:
    - HierarchicalMemory: Multi-tier memory system
    - SemanticMemory: Knowledge graph-based memory
    - WorkingMemory: Active working memory buffer
    - KnowledgeGraph: Semantic relationship graph

Learning Interfaces:
    - RLEnvironment: Reinforcement learning environment
    - RLAgent: Learning agent interface
    - ExperienceReplay: Experience replay buffer
    - PolicyGradient: Policy gradient operations

Message Flow:
    Request → RouteDecision → Retrieval → Reply

Each interface is designed to be:
- Type-safe with comprehensive validation
- Serializable for distributed processing
- Versioned for backward compatibility
- Performance-optimized for real-time operations
- Biological plausibility constraints
- Thread-safe concurrent access patterns

Key Features:
    - Comprehensive type hints for IDE support
    - Automatic validation and error handling
    - Support for streaming and batch operations
    - Hardware-aware optimizations
    - Biological plausibility constraints
    - Multi-modal data support
    - Real-time processing capabilities
    - Distributed system compatibility

Usage Examples:
    Basic Request Processing:
    >>> from src.interfaces import Request, Reply, RouteDecision
    >>> 
    >>> # Create a request
    >>> request = Request(
    ...     query="What is neural network?",
    ...     context={"domain": "education"}
    ... )
    >>> 
    >>> # Process through pipeline
    >>> route = RouteDecision(expert="language", confidence=0.9)
    >>> response = Reply(result="Neural networks are computational models...", confidence=0.95)
    
    Audio Processing:
    >>> from src.interfaces import AudioData, SpeechRecognition
    >>> 
    >>> # Process speech recognition
    >>> audio = AudioData(waveform=np.array([...]), sample_rate=16000, 
    ...                   channels=1, format="wav", duration=2.5, metadata={})
    
    Multimodal Fusion:
    >>> from src.interfaces import MultimodalInput, UnifiedRepresentation
    >>> 
    >>> # Create multimodal input
    >>> multimodal_input = MultimodalInput(
    ...     text="A cat sitting on a table",
    ...     image=np.array([...]),
    ...     audio=np.array([...])
    ... )
    
    Reinforcement Learning:
    >>> from src.interfaces import RLEnvironment, RLAgent, Experience
    >>> 
    >>> # Setup RL environment
    >>> env = RLEnvironment(env_id="cart_pole", action_space={...}, 
    ...                     observation_space={...}, max_steps=1000)
    
    Memory Integration:
    >>> from src.interfaces import STMState, Retrieval, EmbeddingVector, HierarchicalMemory
    >>> 
    >>> # Update short-term memory
    >>> stm_state = STMState(current_items=100, max_items=1000)
    >>> 
    >>> # Retrieve from long-term memory
    >>> retrieval = Retrieval(
    ...     embeddings=[EmbeddingVector(values=[...])],
    ...     similarities=[0.95],
    ...     metadata=[{"source": "corpus"}]
    ... )
    
    Federated Learning:
    >>> from src.interfaces import FederatedClient, FederatedMetrics
    >>> 
    >>> # Setup federated client
    >>> client = FederatedClient(client_id="client_001", data_size=1000, ...)

Architecture Benefits:
    - Decouples implementation from interface contracts
    - Enables plugin-based architecture
    - Facilitates testing with mock implementations
    - Supports distributed system deployment
    - Enables performance monitoring and optimization
    - Biological plausibility constraints
    - Multi-modal AI integration
    - Real-time processing capabilities
    - Scalable architecture patterns

Dependencies:
    - torch: Tensor operations and neural network components
    - numpy: Numerical computations and array operations
    - typing: Type hints for modern Python versions
    - dataclasses: Automatic method generation for data structures
    - networkx: Graph data structures for knowledge graphs
    - asyncio: Asynchronous operations support

Version: 2.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .messages import (
    Request,
    STMState,
    Retrieval,
    RouteDecision,
    Reply,
    EmbeddingVector
)

from .audio_interfaces import (
    AudioFormat,
    AudioSampleRate,
    AudioData,
    SpeechRecognition,
    TextToSpeech,
    AudioClassification,
    MusicAnalysis,
    AudioEnhancement,
    AudioStream,
    AudioConfig
)

from .multimodal_interfaces import (
    ModalityType,
    FusionMethod,
    AttentionType,
    ModalityEmbedding,
    CrossModalAttention,
    ModalityFusion,
    UnifiedRepresentation,
    MultimodalInput,
    AttentionWeights,
    ModalityAlignment,
    MultimodalConfig,
    BiologicalConstraints,
    MultimodalOutput
)

from .data_collection_interfaces import (
    PlatformType,
    SensorType,
    DataFormat,
    SocialMediaPost,
    IoTSensorData,
    StreamingData,
    CollectionMetrics,
    RateLimitConfig,
    DataSource,
    WebContent,
    FileData,
    CollectionConfig
)

from .optimization_interfaces import (
    OptimizationAlgorithm,
    CompressionType,
    SearchStrategy,
    HardwareTarget,
    FederatedClient,
    ArchitectureSearchSpace,
    OptimizationObjective,
    ModelCompression,
    FederatedMetrics,
    SearchResult,
    DistributedOptimizer,
    NASConfig,
    HardwareProfile,
    OptimizationResult
)

from .memory_interfaces import (
    MemoryTier,
    ConsolidationType,
    KnowledgeRelation,
    ReasoningType,
    MemoryItem,
    WorkingMemory,
    EpisodicMemory,
    KnowledgeConcept,
    KnowledgeRelation,
    KnowledgeGraph,
    SemanticMemory,
    MemoryConsolidation,
    HierarchicalMemory,
    ReasoningQuery,
    MemoryRetrieval,
    MemoryConfig
)

from .learning_interfaces import (
    RLAlgorithm,
    ActionSpace,
    ObservationSpace,
    LearningRate,
    RLEnvironment,
    RLState,
    Experience,
    ExperienceReplay,
    PolicyNetwork,
    ValueFunction,
    PolicyGradient,
    BiologicalLearning,
    RLAgent,
    ValueNetwork,
    RLMetrics,
    LearningConfig,
    TrainingResult
)

__all__ = [
    # Core interfaces
    'Request',
    'STMState', 
    'Retrieval',
    'RouteDecision',
    'Reply',
    'EmbeddingVector',
    
    # Audio processing interfaces
    'AudioFormat',
    'AudioSampleRate',
    'AudioData',
    'SpeechRecognition',
    'TextToSpeech',
    'AudioClassification',
    'MusicAnalysis',
    'AudioEnhancement',
    'AudioStream',
    'AudioConfig',
    
    # Multimodal interfaces
    'ModalityType',
    'FusionMethod',
    'AttentionType',
    'ModalityEmbedding',
    'CrossModalAttention',
    'ModalityFusion',
    'UnifiedRepresentation',
    'MultimodalInput',
    'AttentionWeights',
    'ModalityAlignment',
    'MultimodalConfig',
    'BiologicalConstraints',
    'MultimodalOutput',
    
    # Data collection interfaces
    'PlatformType',
    'SensorType',
    'DataFormat',
    'SocialMediaPost',
    'IoTSensorData',
    'StreamingData',
    'CollectionMetrics',
    'RateLimitConfig',
    'DataSource',
    'WebContent',
    'FileData',
    'CollectionConfig',
    
    # Optimization interfaces
    'OptimizationAlgorithm',
    'CompressionType',
    'SearchStrategy',
    'HardwareTarget',
    'FederatedClient',
    'ArchitectureSearchSpace',
    'OptimizationObjective',
    'ModelCompression',
    'FederatedMetrics',
    'SearchResult',
    'DistributedOptimizer',
    'NASConfig',
    'HardwareProfile',
    'OptimizationResult',
    
    # Enhanced memory interfaces
    'MemoryTier',
    'ConsolidationType',
    'KnowledgeRelation',
    'ReasoningType',
    'MemoryItem',
    'WorkingMemory',
    'EpisodicMemory',
    'KnowledgeConcept',
    'KnowledgeGraph',
    'SemanticMemory',
    'MemoryConsolidation',
    'HierarchicalMemory',
    'ReasoningQuery',
    'MemoryRetrieval',
    'MemoryConfig',
    
    # Learning interfaces
    'RLAlgorithm',
    'ActionSpace',
    'ObservationSpace',
    'LearningRate',
    'RLEnvironment',
    'RLState',
    'Experience',
    'ExperienceReplay',
    'PolicyNetwork',
    'ValueFunction',
    'PolicyGradient',
    'BiologicalLearning',
    'RLAgent',
    'ValueNetwork',
    'RLMetrics',
    'LearningConfig',
    'TrainingResult'
]

__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"