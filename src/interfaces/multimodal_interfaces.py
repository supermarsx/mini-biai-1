"""
Multimodal interfaces for the brain-inspired AI system.

This module defines interfaces for multimodal processing capabilities including
vision-language fusion, cross-modal attention, and unified representation learning.

The interfaces support seamless integration of vision, audio, and text modalities
with attention-based fusion mechanisms and biological plausibility constraints.

Key Components:
    - MultimodalInput: Unified multimodal input structure
    - CrossModalAttention: Cross-modal attention weights
    - ModalityFusion: Feature fusion operations
    - UnifiedRepresentation: Combined multimodal embeddings
    - AttentionWeights: Attention mechanism weights
    - ModalityAlignment: Cross-modal alignment results

Architecture Benefits:
    - Attention-based modality fusion
    - Biological plausibility constraints
    - Efficient cross-modal processing
    - Unified representation learning
    - Scalable multimodal architectures

Version: 1.0.0
Author: mini-biai-1 Team
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple
from enum import Enum
import numpy as np
import torch


class ModalityType(Enum):
    """Supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    SENSOR = "sensor"


class FusionMethod(Enum):
    """Multimodal fusion methods."""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    CROSS_ATTENTION = "cross_attention"
    HIERARCHICAL = "hierarchical"
    BILINEAR = "bilinear"


class AttentionType(Enum):
    """Attention mechanism types."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    MULTI_HEAD = "multi_head"
    SPARSE_ATTENTION = "sparse_attention"
    BIOLOGICAL_ATTENTION = "biological_attention"


@dataclass
class ModalityEmbedding:
    """
    Modality-specific embedding representation.
    
    Attributes:
        modality: Type of modality
        embedding: Vector representation
        features: Raw feature values
        metadata: Modality-specific metadata
        quality_score: Quality assessment score
    """
    modality: ModalityType
    embedding: np.ndarray
    features: np.ndarray
    metadata: Dict[str, Any]
    quality_score: float


@dataclass
class CrossModalAttention:
    """
    Cross-modal attention weights and operations.
    
    Attributes:
        query_modality: Source modality for attention
        key_modality: Target modality for attention
        attention_weights: Attention weight matrix
        attention_scores: Attention score values
        attended_features: Attended feature vectors
    """
    query_modality: ModalityType
    key_modality: ModalityType
    attention_weights: np.ndarray
    attention_scores: np.ndarray
    attended_features: np.ndarray


@dataclass
class ModalityFusion:
    """
    Multimodal feature fusion operation.
    
    Attributes:
        method: Fusion method used
        input_embeddings: Input modality embeddings
        fused_features: Combined feature representation
        fusion_weights: Fusion weights per modality
        attention_maps: Attention maps if applicable
    """
    method: FusionMethod
    input_embeddings: List[ModalityEmbedding]
    fused_features: np.ndarray
    fusion_weights: Dict[ModalityType, float]
    attention_maps: Optional[List[CrossModalAttention]] = None


@dataclass
class UnifiedRepresentation:
    """
    Unified multimodal representation.
    
    Attributes:
        representation: Combined multimodal vector
        modality_contributions: Contribution of each modality
        alignment_scores: Cross-modal alignment scores
        coherence_metrics: Representation coherence measures
        semantic_consistency: Semantic consistency scores
    """
    representation: np.ndarray
    modality_contributions: Dict[ModalityType, float]
    alignment_scores: Dict[Tuple[ModalityType, ModalityType], float]
    coherence_metrics: Dict[str, float]
    semantic_consistency: float


@dataclass
class MultimodalInput:
    """
    Unified multimodal input structure.
    
    Attributes:
        text: Text input data
        image: Image input data
        audio: Audio input data
        video: Video input data
        sensor: Sensor input data
        modality_weights: Relative importance weights
        sequence_order: Temporal sequence information
    """
    text: Optional[Union[str, np.ndarray]] = None
    image: Optional[np.ndarray] = None
    audio: Optional[np.ndarray] = None
    video: Optional[np.ndarray] = None
    sensor: Optional[Dict[str, Any]] = None
    modality_weights: Optional[Dict[ModalityType, float]] = None
    sequence_order: Optional[List[int]] = None


@dataclass
class AttentionWeights:
    """
    Attention mechanism weights and parameters.
    
    Attributes:
        attention_type: Type of attention mechanism
        weights_matrix: Attention weight matrix
        query_features: Query feature vectors
        key_features: Key feature vectors
        value_features: Value feature vectors
        attention_mask: Attention mask for padding
    """
    attention_type: AttentionType
    weights_matrix: np.ndarray
    query_features: np.ndarray
    key_features: np.ndarray
    value_features: np.ndarray
    attention_mask: Optional[np.ndarray] = None


@dataclass
class ModalityAlignment:
    """
    Cross-modal alignment results.
    
    Attributes:
        aligned_pairs: Aligned modality pairs
        alignment_scores: Alignment quality scores
        transformation_matrices: Alignment transformations
        semantic_matches: Semantic correspondence matches
        temporal_alignment: Temporal alignment results
    """
    aligned_pairs: List[Tuple[ModalityType, ModalityType]]
    alignment_scores: Dict[Tuple[ModalityType, ModalityType], float]
    transformation_matrices: Dict[Tuple[ModalityType, ModalityType], np.ndarray]
    semantic_matches: List[Dict[str, Any]]
    temporal_alignment: Optional[Dict[str, Any]] = None


@dataclass
class MultimodalConfig:
    """
    Multimodal processing configuration.
    
    Attributes:
        fusion_method: Preferred fusion method
        attention_config: Attention mechanism parameters
        embedding_dim: Dimension of embeddings
        num_heads: Number of attention heads
        dropout_rate: Dropout probability
        hidden_layers: Hidden layer configuration
        output_dim: Output representation dimension
    """
    fusion_method: FusionMethod
    attention_config: Dict[str, Any]
    embedding_dim: int
    num_heads: int
    dropout_rate: float
    hidden_layers: List[int]
    output_dim: int


@dataclass
class BiologicalConstraints:
    """
    Biological plausibility constraints for multimodal processing.
    
    Attributes:
        attention_sparsity: Sparsity constraints on attention
        temporal_dependencies: Temporal dependency requirements
        modality_integration: Biological modality integration rules
        neural_plasticity: Neural plasticity parameters
        cognitive_load: Cognitive load limitations
    """
    attention_sparsity: float
    temporal_dependencies: Dict[str, Any]
    modality_integration: Dict[str, Any]
    neural_plasticity: Dict[str, float]
    cognitive_load: Dict[str, float]


@dataclass
class MultimodalOutput:
    """
    Multimodal processing output structure.
    
    Attributes:
        unified_representation: Combined multimodal representation
        modality_specific_outputs: Individual modality outputs
        attention_analysis: Attention mechanism analysis
        alignment_results: Cross-modal alignment results
        confidence_scores: Output confidence measures
        processing_metadata: Processing metadata and statistics
    """
    unified_representation: UnifiedRepresentation
    modality_specific_outputs: Dict[ModalityType, Any]
    attention_analysis: AttentionWeights
    alignment_results: ModalityAlignment
    confidence_scores: Dict[str, float]
    processing_metadata: Dict[str, Any]


# Export all interfaces
__all__ = [
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
    'MultimodalOutput'
]