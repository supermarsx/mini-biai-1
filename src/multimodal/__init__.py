"""
Advanced Cross-Modal Attention System for Brain-Inspired AI

This module implements sophisticated multi-modal attention mechanisms for vision-language-audio
integration, providing advanced cross-modal fusion capabilities and biological plausibility.

Key Components:
- Cross-Modal Attention Mechanisms: Co-attention, self-attention, cross-attention
- Multi-Modal Fusion Strategies: Early, late, hierarchical, and bilinear fusion
- Multi-Modal Embedding Spaces: Shared representations across modalities
- Cross-Modal Retrieval: Vision-language-audio retrieval and generation
- Attention Visualization: Interpretability and attention analysis
- Integration: Seamless integration with existing vision and language modules

Architecture Benefits:
- Brain-inspired attention mechanisms
- Real-time multi-modal processing
- Cross-modal alignment and fusion
- Attention weight visualization
- Production-ready optimization

Version: 4.0.0
Author: mini-biai-1 Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings

# Core attention mechanisms
from .attention import (
    CrossModalAttention,
    SelfAttention,
    CoAttention,
    MultiHeadCrossModalAttention,
    BiologicalAttention,
    AttentionMechanism
)

# Multi-modal fusion strategies
from .fusion import (
    EarlyFusion,
    LateFusion,
    HierarchicalFusion,
    BilinearFusion,
    AdaptiveFusion,
    FusionStrategy
)

# Multi-modal embeddings
from .embeddings import (
    ModalityEmbedding,
    SharedEmbeddingSpace,
    CrossModalProjection,
    EmbeddingAlignment,
    UnifiedRepresentation
)

# Cross-modal retrieval
from .retrieval import (
    CrossModalRetriever,
    VisionLanguageRetrieval,
    AudioVisualRetrieval,
    MultimodalGenerator
)

# Attention visualization
from .visualization import (
    AttentionVisualizer,
    CrossModalAttentionMapper,
    AttentionHeatmap,
    ModalityContributionAnalysis
)

# Integration with existing modules
from .integration import (
    MultimodalIntegration,
    VisionLanguageIntegration,
    AudioVisualIntegration,
    UnifiedMultimodalProcessor
)

# Version information
__version__ = "4.0.0"
__author__ = "mini-biai-1 Team"

# Set up logging
logger = logging.getLogger(__name__)

# Export all components
__all__ = [
    # Core attention mechanisms
    'CrossModalAttention',
    'SelfAttention',
    'CoAttention', 
    'MultiHeadCrossModalAttention',
    'BiologicalAttention',
    'AttentionMechanism',
    
    # Multi-modal fusion strategies
    'EarlyFusion',
    'LateFusion',
    'HierarchicalFusion',
    'BilinearFusion',
    'AdaptiveFusion',
    'FusionStrategy',
    
    # Multi-modal embeddings
    'ModalityEmbedding',
    'SharedEmbeddingSpace',
    'CrossModalProjection',
    'EmbeddingAlignment',
    'UnifiedRepresentation',
    
    # Cross-modal retrieval
    'CrossModalRetriever',
    'VisionLanguageRetrieval',
    'AudioVisualRetrieval',
    'MultimodalGenerator',
    
    # Attention visualization
    'AttentionVisualizer',
    'CrossModalAttentionMapper',
    'AttentionHeatmap',
    'ModalityContributionAnalysis',
    
    # Integration modules
    'MultimodalIntegration',
    'VisionLanguageIntegration',
    'AudioVisualIntegration',
    'UnifiedMultimodalProcessor',
    
    # Version info
    '__version__'
]

# Check for required dependencies
def _check_dependencies():
    """Check if required dependencies are available."""
    required_packages = ['torch', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        warnings.warn(
            f"Missing recommended packages: {missing_packages}. "
            "Some visualization features may not work properly.",
            ImportWarning
        )

# Initialize the module
_check_dependencies()
logger.info(f"Advanced Cross-Modal Attention System v{__version__} initialized")