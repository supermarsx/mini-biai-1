"""
Advanced Cross-Modal Attention System

A comprehensive multi-modal attention and fusion system for vision-language-audio integration.
"""

__version__ = "4.0.0"
__author__ = "mini-biai-1 Team"
__license__ = "MIT"

# Core components
from .attention import (
    AttentionMechanism,
    SelfAttention,
    CrossModalAttention,
    CoAttention,
    MultiHeadCrossModalAttention,
    BiologicalAttention,
    create_attention_mechanism,
    combine_attention_outputs
)

from .fusion import (
    FusionStrategy,
    EarlyFusion,
    LateFusion,
    HierarchicalFusion,
    BilinearFusion,
    AdaptiveFusion,
    create_fusion_strategy,
    evaluate_fusion_quality
)

from .embeddings import (
    ModalityEncoder,
    TextModalityEncoder,
    ImageModalityEncoder,
    AudioModalityEncoder,
    SharedEmbeddingSpace,
    CrossModalProjection,
    EmbeddingAlignment,
    UnifiedRepresentation,
    create_modality_encoder,
    evaluate_embedding_quality
)

from .retrieval import (
    CrossModalRetriever,
    SemanticSimilarityRetriever,
    VisionLanguageRetrieval,
    AudioVisualRetrieval,
    MultimodalGenerator,
    TextToImageGenerator,
    ImageToAudioGenerator,
    create_retriever,
    create_generator,
    evaluate_retrieval_quality
)

from .visualization import (
    AttentionVisualizer,
    CrossModalAttentionMapper,
    AttentionHeatmap,
    ModalityContributionAnalysis,
    AttentionVisualization,
    create_visualization_report,
    analyze_attention_patterns
)

from .integration import (
    IntegrationConfig,
    BaseIntegration,
    VisionLanguageIntegration,
    AudioVisualIntegration,
    UnifiedMultimodalProcessor,
    create_vision_language_integration,
    create_audio_visual_integration,
    create_unified_multimodal_processor
)

# Main exports
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Attention mechanisms
    'AttentionMechanism',
    'SelfAttention',
    'CrossModalAttention', 
    'CoAttention',
    'MultiHeadCrossModalAttention',
    'BiologicalAttention',
    'create_attention_mechanism',
    'combine_attention_outputs',
    
    # Fusion strategies
    'FusionStrategy',
    'EarlyFusion',
    'LateFusion',
    'HierarchicalFusion',
    'BilinearFusion',
    'AdaptiveFusion',
    'create_fusion_strategy',
    'evaluate_fusion_quality',
    
    # Embedding systems
    'ModalityEncoder',
    'TextModalityEncoder',
    'ImageModalityEncoder',
    'AudioModalityEncoder',
    'SharedEmbeddingSpace',
    'CrossModalProjection',
    'EmbeddingAlignment',
    'UnifiedRepresentation',
    'create_modality_encoder',
    'evaluate_embedding_quality',
    
    # Retrieval systems
    'CrossModalRetriever',
    'SemanticSimilarityRetriever',
    'VisionLanguageRetrieval',
    'AudioVisualRetrieval',
    'MultimodalGenerator',
    'TextToImageGenerator',
    'ImageToAudioGenerator',
    'create_retriever',
    'create_generator',
    'evaluate_retrieval_quality',
    
    # Visualization
    'AttentionVisualizer',
    'CrossModalAttentionMapper',
    'AttentionHeatmap',
    'ModalityContributionAnalysis',
    'AttentionVisualization',
    'create_visualization_report',
    'analyze_attention_patterns',
    
    # Integration
    'IntegrationConfig',
    'BaseIntegration',
    'VisionLanguageIntegration',
    'AudioVisualIntegration',
    'UnifiedMultimodalProcessor',
    'create_vision_language_integration',
    'create_audio_visual_integration',
    'create_unified_multimodal_processor'
]

# Check for required dependencies
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Warning: Some dependencies are missing. {e}")
    print("Please install required packages: torch, numpy, matplotlib, seaborn")

# Initialize logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Advanced Cross-Modal Attention System v{__version__} initialized")