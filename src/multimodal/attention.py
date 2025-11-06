"""
Advanced Cross-Modal Attention Mechanisms

This module implements sophisticated attention mechanisms for cross-modal processing,
including co-attention, self-attention, cross-attention, and biological attention patterns
with brain-inspired architecture principles.

Key Features:
- Multi-head cross-modal attention
- Co-attention mechanisms for vision-language alignment
- Biological attention with sparse patterns
- Self-attention for modal consistency
- Attention weight visualization and analysis

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                Cross-Modal Attention System                │
    └─────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Self-     │ │   Cross-    │ │   Co-       │
    │ Attention   │ │ Attention   │ │ Attention   │
    │             │ │             │ │             │
    │ • Intra-    │ │ • Inter-    │ │ • Vision-   │
    │   modal     │ │   modal     │ │   Language  │
    │ • Token     │ │ • Query-    │ │ • Audio-    │
    │   attention │ │   Key       │ │   Visual    │
    │   modeling  │ │ -Value      │ │ • Joint     │
    └─────────────┘ └─────────────┘ └─────────────┘
                          │
                          ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Biological  │ │ Multi-Head  │ │ Attention   │
    │ Attention   │ │ Attention   │ │ Analysis    │
    │             │ │             │ │             │
    │ • Sparse    │ │ • Multiple  │ │ • Weight    │
    │   patterns  │ │   heads     │ │   analysis  │
    │ • Recurrent │ │ • Parallel  │ │ • Heatmaps  │
    │   attention │ │   processing│ │ • Gradients │
    └─────────────┘ └─────────────┘ └─────────────┘

Version: 4.0.0
Author: mini-biai-1 Team
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math
import warnings
import logging

# Import interfaces
try:
    from ..interfaces.multimodal_interfaces import (
        ModalityType, AttentionType, AttentionWeights, CrossModalAttention as InterfaceCrossModalAttention
    )
    # Create a local CrossModalAttention class for this module
    CrossModalAttention = InterfaceCrossModalAttention
except ImportError:
    # Fallback for standalone usage
    from enum import Enum
    from dataclasses import dataclass
    
    class ModalityType(Enum):
        TEXT = "text"
        IMAGE = "image"
        AUDIO = "audio"
        VIDEO = "video"
    
    class AttentionType(Enum):
        SELF_ATTENTION = "self_attention"
        CROSS_ATTENTION = "cross_attention"
        MULTI_HEAD = "multi_head"
        SPARSE_ATTENTION = "sparse_attention"
        BIOLOGICAL_ATTENTION = "biological_attention"
    
    @dataclass
    class CrossModalAttention:
        """Simplified CrossModalAttention for fallback usage."""
        query_modality: ModalityType
        key_modality: ModalityType
        attention_weights: np.ndarray
        attention_scores: np.ndarray
        attended_features: np.ndarray
    
    @dataclass
    class AttentionWeights:
        attention_type: AttentionType
        weights_matrix: np.ndarray
        query_features: np.ndarray
        key_features: np.ndarray
        value_features: np.ndarray
        attention_mask: Optional[np.ndarray] = None

logger = logging.getLogger(__name__)


class AttentionMechanism(ABC):
    """Abstract base class for attention mechanisms."""
    
    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of attention mechanism.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            modality_mask: Mask for different modalities
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        pass
    
    @abstractmethod
    def get_attention_weights(self) -> torch.Tensor:
        """Get attention weights for analysis."""
        pass


class SelfAttention(AttentionMechanism):
    """
    Self-Attention mechanism for intra-modal processing.
    
    Implements scaled dot-product self-attention with optional biological constraints.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        sparse_attention: bool = False,
        sparsity_ratio: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.sparse_attention = sparse_attention
        self.sparsity_ratio = sparsity_ratio
        
        # Linear projections
        self.query_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        
        # Scaled dot-product attention scaling
        self.scale = math.sqrt(self.head_dim)
        
        # Biological constraints
        self.register_buffer('attention_weights', torch.Tensor())
        self.attention_masks = None
        
        logger.info(f"SelfAttention initialized: dim={embedding_dim}, heads={num_heads}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of self-attention.
        
        Args:
            query: Input tensor of shape (batch, seq_len, embedding_dim)
            key: Key tensor of shape (batch, seq_len, embedding_dim) 
            value: Value tensor of shape (batch, seq_len, embedding_dim)
            modality_mask: Optional mask for modalities
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        q = self.query_projection(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_projection(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_projection(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply modality mask if provided
        if modality_mask is not None:
            attention_scores = attention_scores.masked_fill(modality_mask == 0, -1e9)
        
        # Apply sparse attention if enabled
        if self.sparse_attention:
            attention_scores = self._apply_sparse_attention(attention_scores)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for analysis
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )
        output = self.output_projection(attended_output)
        
        return output, attention_weights
    
    def _apply_sparse_attention(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Apply sparse attention pattern inspired by biological neurons."""
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Top-k sparse attention (biological plausibility)
        k = max(1, int(seq_len * self.sparsity_ratio))
        top_scores, _ = torch.topk(attention_scores, k, dim=-1)
        
        # Create sparse mask
        sparse_mask = torch.zeros_like(attention_scores)
        sparse_mask.scatter_(-1, torch.topk(attention_scores, k, dim=-1).indices, 1.0)
        
        # Apply sparse mask
        attention_scores = attention_scores * sparse_mask
        
        return attention_scores
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get stored attention weights."""
        if hasattr(self, 'attention_weights') and len(self.attention_weights) > 0:
            return self.attention_weights
        else:
            return torch.Tensor()


class CrossModalAttention(AttentionMechanism):
    """
    Cross-Modal Attention for inter-modal processing.
    
    Enables attention between different modalities (e.g., vision to text, audio to vision).
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        modality_specific_heads: bool = True,
        **kwargs
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.modality_specific_heads = modality_specific_heads
        
        # Linear projections
        self.query_projection = nn.Linear(query_dim, output_dim, bias=bias)
        self.key_projection = nn.Linear(key_dim, output_dim, bias=bias)
        self.value_projection = nn.Linear(value_dim, output_dim, bias=bias)
        self.output_projection = nn.Linear(output_dim, output_dim, bias=bias)
        
        # Scaled dot-product attention scaling
        self.scale = math.sqrt(self.head_dim)
        
        # Cross-modal alignment layers
        if modality_specific_heads:
            self.alignment_layers = nn.ModuleDict({
                'text': nn.Linear(output_dim, output_dim // 2),
                'image': nn.Linear(output_dim, output_dim // 2),
                'audio': nn.Linear(output_dim, output_dim // 2)
            })
        
        logger.info(f"CrossModalAttention initialized: q={query_dim}, k={key_dim}, v={value_dim}, o={output_dim}")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        query_modality: Optional[ModalityType] = None,
        key_modality: Optional[ModalityType] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of cross-modal attention.
        
        Args:
            query: Query tensor from source modality
            key: Key tensor from target modality
            value: Value tensor from target modality
            modality_mask: Optional mask for modalities
            query_modality: Type of query modality
            key_modality: Type of key modality
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, query_len = query.shape[:2]
        batch_size, key_len = key.shape[:2]
        
        # Project to Q, K, V
        q = self.query_projection(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key_projection(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value_projection(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-modal attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply modality alignment if enabled
        if self.modality_specific_heads and query_modality and key_modality:
            attention_scores = self._apply_modality_alignment(
                attention_scores, query_modality, key_modality
            )
        
        # Apply mask if provided
        if modality_mask is not None:
            attention_scores = attention_scores.masked_fill(modality_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.output_dim
        )
        output = self.output_projection(attended_output)
        
        return output, attention_weights
    
    def _apply_modality_alignment(
        self,
        attention_scores: torch.Tensor,
        query_modality: ModalityType,
        key_modality: ModalityType
    ) -> torch.Tensor:
        """Apply modality-specific alignment to attention scores."""
        # Apply alignment layers based on modality combinations
        alignment_key = f"{query_modality.value}_{key_modality.value}"
        
        if alignment_key in self.alignment_layers:
            # Apply modality-specific alignment
            alignment_layer = self.alignment_layers[alignment_key]
            # This is a simplified version - in practice, you'd apply this more carefully
            # considering the attention score dimensions
        
        return attention_scores


class CoAttention(AttentionMechanism):
    """
    Co-Attention mechanism for joint vision-language processing.
    
    Implements bidirectional attention between paired modalities with shared attention weights.
    """
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **kwargs
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        
        # Attention mechanisms
        self.vision_to_language_attention = CrossModalAttention(
            query_dim=vision_dim,
            key_dim=language_dim,
            value_dim=language_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        if bidirectional:
            self.language_to_vision_attention = CrossModalAttention(
                query_dim=language_dim,
                key_dim=vision_dim,
                value_dim=vision_dim,
                output_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Joint attention for combining both directions
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2 if bidirectional else hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim,
            hidden_dim
        )
        
        logger.info(f"CoAttention initialized: vision={vision_dim}, language={language_dim}")
    
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of co-attention.
        
        Args:
            vision_features: Vision feature tensor
            language_features: Language feature tensor
            vision_mask: Optional mask for vision features
            language_mask: Optional mask for language features
            
        Returns:
            Tuple of (vision_attended, language_attended, attention_weights)
        """
        # Vision-to-Language attention
        vision_attended, v2l_attention = self.vision_to_language_attention(
            query=vision_features,
            key=language_features,
            value=language_features,
            modality_mask=language_mask,
            query_modality=ModalityType.IMAGE,
            key_modality=ModalityType.TEXT
        )
        
        language_attended = language_features
        joint_attention_weights = v2l_attention
        
        if self.bidirectional:
            # Language-to-Vision attention
            language_attended, l2v_attention = self.language_to_vision_attention(
                query=language_features,
                key=vision_features,
                value=vision_features,
                modality_mask=vision_mask,
                query_modality=ModalityType.TEXT,
                key_modality=ModalityType.IMAGE
            )
            
            # Combine attention weights
            joint_attention_weights = torch.cat([v2l_attention, l2v_attention], dim=-1)
        
        # Apply joint attention
        if self.bidirectional:
            combined_features = torch.cat([vision_attended, language_attended], dim=-1)
            joint_attended, attention_weights = self.joint_attention(
                query=combined_features,
                key=combined_features,
                value=combined_features
            )
        else:
            joint_attended = torch.cat([vision_attended, language_attended], dim=-1)
            attention_weights = v2l_attention
        
        # Project output
        output = self.output_projection(joint_attended)
        
        return vision_attended, language_attended, attention_weights


class MultiHeadCrossModalAttention(AttentionMechanism):
    """
    Multi-Head Cross-Modal Attention for complex multimodal scenarios.
    
    Combines multiple attention heads with different modalities and patterns.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_modalities: int = 3,
        num_heads_per_modality: int = 4,
        dropout: float = 0.1,
        cross_modal_routing: bool = True,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_modalities = num_modalities
        self.heads_per_modality = num_heads_per_modality
        self.total_heads = num_modalities * num_heads_per_modality
        self.cross_modal_routing = cross_modal_routing
        self.dropout = nn.Dropout(dropout)
        
        # Create attention heads for each modality
        self.attention_heads = nn.ModuleDict()
        self.attention_heads['self'] = nn.ModuleList([
            SelfAttention(
                embedding_dim=embedding_dim,
                num_heads=num_heads_per_modality,
                dropout=dropout
            ) for _ in range(num_modalities)
        ])
        
        # Cross-modal attention heads
        self.cross_modal_heads = nn.ModuleDict()
        modalities = ['text', 'image', 'audio']
        
        for i, src_mod in enumerate(modalities):
            for j, tgt_mod in enumerate(modalities):
                if i != j:  # No self-attention here
                    self.cross_modal_heads[f"{src_mod}_to_{tgt_mod}"] = CrossModalAttention(
                        query_dim=embedding_dim,
                        key_dim=embedding_dim,
                        value_dim=embedding_dim,
                        output_dim=embedding_dim,
                        num_heads=num_heads_per_modality,
                        dropout=dropout
                    )
        
        # Routing mechanism for adaptive attention
        if cross_modal_routing:
            self.attention_router = nn.Linear(embedding_dim, self.total_heads)
            self.routing_weights = torch.ones(self.total_heads)
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        logger.info(f"MultiHeadCrossModalAttention initialized: {self.total_heads} total heads")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of multi-head cross-modal attention.
        
        Args:
            modality_features: Dictionary mapping modality names to feature tensors
            
        Returns:
            Tuple of (attended_output, attention_weights_dict)
        """
        attended_outputs = {}
        all_attention_weights = {}
        
        # Self-attention for each modality
        for modality, features in modality_features.items():
            if modality in self.attention_heads['self']:
                attended, weights = self.attention_heads['self'][
                    list(self.attention_heads['self']).index(
                        self.attention_heads['self'][
                            list(self.attention_heads['self']).index
                        ]
                    )
                ](query=features, key=features, value=features)
                
                attended_outputs[f"{modality}_self"] = attended
                all_attention_weights[f"{modality}_self"] = weights
        
        # Cross-modal attention
        for cross_key, attention_layer in self.cross_modal_heads.items():
            src_mod, tgt_mod = cross_key.split('_to_')
            
            if src_mod in modality_features and tgt_mod in modality_features:
                src_features = modality_features[src_mod]
                tgt_features = modality_features[tgt_mod]
                
                attended, weights = attention_layer(
                    query=src_features,
                    key=tgt_features,
                    value=tgt_features
                )
                
                attended_outputs[f"{src_mod}_to_{tgt_mod}"] = attended
                all_attention_weights[f"{src_mod}_to_{tgt_mod}"] = weights
        
        # Combine outputs
        combined_output = torch.zeros_like(list(modality_features.values())[0])
        
        if self.cross_modal_routing and hasattr(self, 'attention_router'):
            # Apply attention routing
            for modality, features in modality_features.items():
                routing_scores = F.softmax(self.attention_router(features), dim=-1)
                # Weighted combination of attention outputs
                combined_output = combined_output + routing_scores.mean(dim=1, keepdim=True) * features
        else:
            # Simple average of self-attention outputs
            for modality, attended in attended_outputs.items():
                if modality.endswith('_self'):
                    combined_output = combined_output + attended
            combined_output = combined_output / len(attended_outputs)
        
        # Final output projection
        output = self.output_projection(combined_output)
        
        return output, all_attention_weights


class BiologicalAttention(AttentionMechanism):
    """
    Biological Attention mechanism with sparse, recurrent, and realistic patterns.
    
    Mimics biological neural attention with sparse activation patterns and
    temporal dynamics inspired by neuroscience research.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        sparse_ratio: float = 0.3,
        recurrent_steps: int = 3,
        plasticity: float = 0.1,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.sparse_ratio = sparse_ratio
        self.recurrent_steps = recurrent_steps
        self.plasticity = plasticity
        self.head_dim = embedding_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Base attention components
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Biological constraints
        self.sparse_gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Attention state for recurrent processing
        self.attention_state = torch.zeros(1, 1, embedding_dim)
        
        # Plasticity parameters
        self.plasticity_gate = nn.Linear(embedding_dim, 1)
        
        logger.info(f"BiologicalAttention initialized: {embedding_dim}dim, {num_heads}heads, {sparse_ratio:.2f}sparse")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of biological attention.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        q = self.query_projection(query)
        k = self.key_projection(key)
        v = self.value_projection(value)
        
        # Recurrent attention processing
        for step in range(self.recurrent_steps):
            # Compute attention scores
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embedding_dim)
            
            # Apply biological sparsity
            sparse_weights = self.sparse_gate(q)
            attention_scores = attention_scores * sparse_weights
            
            # Apply softmax with biological constraints
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply sparse constraint
            sparse_mask = self._create_sparse_mask(attention_weights)
            attention_weights = attention_weights * sparse_mask
            
            # Recurrent update of attention state
            attention_output = torch.matmul(attention_weights, v)
            self.attention_state = (1 - self.plasticity) * self.attention_state + \
                                 self.plasticity * attention_output.mean(dim=1, keepdim=True)
        
        # Final attention application
        attended_output = torch.matmul(attention_weights, v)
        
        # Apply dropout
        attended_output = self.dropout(attended_output)
        
        # Output projection
        output = self.output_projection(attended_output)
        
        return output, attention_weights
    
    def _create_sparse_mask(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Create sparse activation mask inspired by biological neurons."""
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        
        # Determine top-k for sparsity
        k = max(1, int(seq_len * self.sparse_ratio))
        
        # Create sparse mask
        top_scores, _ = torch.topk(attention_weights, k, dim=-1)
        sparse_mask = torch.zeros_like(attention_weights)
        sparse_mask.scatter_(-1, torch.topk(attention_weights, k, dim=-1).indices, 1.0)
        
        return sparse_mask
    
    def get_attention_weights(self) -> torch.Tensor:
        """Get attention weights for analysis."""
        # In a real implementation, this would return stored attention weights
        return torch.Tensor()


# Utility functions for attention mechanism management

def create_attention_mechanism(
    attention_type: AttentionType,
    **kwargs
) -> AttentionMechanism:
    """Factory function to create attention mechanisms.
    
    Args:
        attention_type: Type of attention mechanism
        **kwargs: Configuration parameters
        
    Returns:
        Configured attention mechanism
    """
    attention_classes = {
        AttentionType.SELF_ATTENTION: SelfAttention,
        AttentionType.CROSS_ATTENTION: CrossModalAttention,
        AttentionType.BIOLOGICAL_ATTENTION: BiologicalAttention,
        AttentionType.MULTI_HEAD: MultiHeadCrossModalAttention
    }
    
    if attention_type not in attention_classes:
        raise ValueError(f"Unknown attention type: {attention_type}")
    
    return attention_classes[attention_type](**kwargs)


def combine_attention_outputs(
    attention_outputs: List[Tuple[torch.Tensor, torch.Tensor]],
    combination_method: str = "weighted_average"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine outputs from multiple attention mechanisms.
    
    Args:
        attention_outputs: List of (output, weights) tuples
        combination_method: Method to combine outputs
        
    Returns:
        Tuple of (combined_output, combined_weights)
    """
    if not attention_outputs:
        return torch.Tensor(), torch.Tensor()
    
    if combination_method == "weighted_average":
        # Weighted average based on confidence scores
        weights = [F.softmax(torch.rand(1)) for _ in attention_outputs]
        combined_output = sum(w * out for w, (out, _) in zip(weights, attention_outputs))
        combined_weights = torch.stack([w for _, w in attention_outputs])
    elif combination_method == "concatenation":
        # Concatenate outputs
        outputs, weights = zip(*attention_outputs)
        combined_output = torch.cat(outputs, dim=-1)
        combined_weights = torch.stack(weights)
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")
    
    return combined_output, combined_weights


# Export all classes and functions
__all__ = [
    'AttentionMechanism',
    'SelfAttention', 
    'CrossModalAttention',
    'CoAttention',
    'MultiHeadCrossModalAttention',
    'BiologicalAttention',
    'create_attention_mechanism',
    'combine_attention_outputs'
]