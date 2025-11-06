"""
Multi-Modal Embedding Spaces and Projections

This module implements sophisticated multi-modal embedding spaces and cross-modal
projection mechanisms for unified representation learning across vision, language,
and audio modalities with attention-based alignment and biological constraints.

Key Features:
- Shared embedding spaces with modality-specific encoders
- Cross-modal projection networks
- Embedding alignment and consistency mechanisms
- Unified representation learning
- Quality-aware embeddings with confidence scores
- Biological plausibility constraints

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │            Multi-Modal Embedding System                    │
    └─────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Modality   │ │   Shared    │ │  Cross-     │
    │  Specific   │ │  Embedding  │ │   Modal     │
    │  Encoders   │ │   Space     │ │ Projection  │
    │             │ │             │ │             │
    │ • Text      │ │ • Unified   │ │ • Text->Vis │
    │   Token     │ │   vector    │ │ • Vis->Text │
    │ • Image     │ │ • Shared    │ │ • Audio->Vis│
    │   Patch     │ │   attention │ │ • Attention │
    │ • Audio     │ │ • Quality   │ │   routing   │
    │   Spectro   │ │   scoring   │ │ • Alignment │
    └─────────────┘ └─────────────┘ └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │ Embedding   │ │  Unified    │ │ Quality     │
    │  Alignment  │ │ Representa  │ │ Assessment  │
    │             │ │   tion      │ │             │
    │ • Cosine    │ │ • Multi-    │ │ • Confidence│
    │   similarity│ │   modal     │ │   scores    │
    │ • Orthogonal│ │   fusion    │ │ • Quality   │
    │   constraints│ │ • Attention│ │   metrics   │
    │ • Attention │ │   weights   │ │ • Consistency│
    │   maps      │ │ • Semantic  │ │ • Validity  │
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
import logging

# Import interfaces
try:
    from ..interfaces.multimodal_interfaces import (
        ModalityType, ModalityEmbedding, UnifiedRepresentation, 
        ModalityAlignment, BiologicalConstraints
    )
    from .attention import AttentionMechanism
    from .fusion import FusionStrategy
except ImportError:
    # Fallback for standalone usage
    from enum import Enum
    from dataclasses import dataclass
    
    class ModalityType(Enum):
        TEXT = "text"
        IMAGE = "image"
        AUDIO = "audio"
        VIDEO = "video"


logger = logging.getLogger(__name__)


@dataclass
class ModalityEmbedding:
    """Modality-specific embedding with metadata."""
    modality: ModalityType
    embedding: torch.Tensor
    features: torch.Tensor
    metadata: Dict[str, Any]
    quality_score: float
    confidence: float = 0.0


class ModalityEncoder(ABC):
    """Abstract base class for modality-specific encoders."""
    
    @abstractmethod
    def forward(self, input_data: Any) -> ModalityEmbedding:
        """Encode input data into modality embedding.
        
        Args:
            input_data: Input data for the modality
            
        Returns:
            ModalityEmbedding object
        """
        pass


class TextModalityEncoder(ModalityEncoder):
    """Text modality encoder for language processing."""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 512,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # Text processing layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(6)
        ])
        
        # Quality assessment
        self.quality_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"TextModalityEncoder: {vocab_size} vocab, {embedding_dim}dim")
    
    def forward(self, input_data: Union[str, List[str], torch.Tensor]) -> ModalityEmbedding:
        """Encode text input into embedding.
        
        Args:
            input_data: Text input (string, list, or tokenized tensor)
            
        Returns:
            ModalityEmbedding with text representation
        """
        # Handle different input formats
        if isinstance(input_data, str):
            # Simple tokenization (in practice, use proper tokenizer)
            tokens = torch.tensor([ord(c) for c in input_data[:self.max_seq_length]])
            if len(tokens) < self.max_seq_length:
                padding = torch.zeros(self.max_seq_length - len(tokens), dtype=torch.long)
                tokens = torch.cat([tokens, padding])
        elif isinstance(input_data, list):
            # Convert list of strings to tokens
            tokens = torch.tensor([hash(item) % self.vocab_size for item in input_data])
            if len(tokens) < self.max_seq_length:
                padding = torch.zeros(self.max_seq_length - len(tokens), dtype=torch.long)
                tokens = torch.cat([tokens, padding])
        else:
            # Assume already tokenized
            tokens = input_data.long()
        
        # Ensure correct shape
        if len(tokens.shape) == 1:
            tokens = tokens.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Create position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        
        # Create token embeddings
        token_embeds = self.token_embedding(tokens)
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            embeddings = layer(embeddings)
        
        # Pool embeddings
        pooled_embedding = embeddings.mean(dim=1)  # Mean pooling
        
        # Assess quality
        quality_score = self.quality_predictor(pooled_embedding).squeeze(-1)
        
        # Metadata
        metadata = {
            'input_type': type(input_data).__name__,
            'seq_length': seq_len,
            'vocab_coverage': min(1.0, len(torch.unique(tokens)) / self.vocab_size),
            'embedding_method': 'transformer_mean_pool'
        }
        
        return ModalityEmbedding(
            modality=ModalityType.TEXT,
            embedding=pooled_embedding,
            features=embeddings,
            metadata=metadata,
            quality_score=quality_score.item(),
            confidence=quality_score.item()
        )


class ImageModalityEncoder(ModalityEncoder):
    """Image modality encoder for visual processing."""
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        embedding_dim: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        # Calculate number of patches
        self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embeddings
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embedding_dim) * 0.02
        )
        
        # Vision transformer layers
        self.vit_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(6)
        ])
        
        # Quality assessment
        self.quality_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"ImageModalityEncoder: {image_size} -> {self.num_patches} patches, {embedding_dim}dim")
    
    def forward(self, input_data: Union[np.ndarray, torch.Tensor]) -> ModalityEmbedding:
        """Encode image input into embedding.
        
        Args:
            input_data: Image input (numpy array or torch tensor)
            
        Returns:
            ModalityEmbedding with image representation
        """
        # Handle different input formats
        if isinstance(input_data, np.ndarray):
            if input_data.dtype != np.float32:
                input_data = input_data.astype(np.float32) / 255.0
            images = torch.from_numpy(input_data).permute(0, 3, 1, 2)  # NCHW
        else:
            # Assume torch tensor
            if len(input_data.shape) == 3:
                input_data = input_data.unsqueeze(0)  # Add batch dim
            images = input_data
        
        # Ensure correct size
        if images.shape[-2:] != self.image_size:
            images = F.interpolate(images, size=self.image_size, mode='bilinear')
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        # Create patch embeddings
        patch_embeds = self.patch_embedding(images).flatten(2).transpose(1, 2)
        
        # Add position embeddings
        batch_size = patch_embeds.shape[0]
        position_embeds = self.position_embedding.expand(batch_size, -1, -1)
        embeddings = patch_embeds + position_embeds
        
        # Apply transformer layers
        for layer in self.vit_layers:
            embeddings = layer(embeddings)
        
        # Pool embeddings
        pooled_embedding = embeddings.mean(dim=1)  # Mean pooling
        
        # Assess quality
        quality_score = self.quality_predictor(pooled_embedding).squeeze(-1)
        
        # Metadata
        metadata = {
            'input_shape': images.shape,
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'resolution': f"{self.image_size[0]}x{self.image_size[1]}",
            'embedding_method': 'vit_mean_pool'
        }
        
        return ModalityEmbedding(
            modality=ModalityType.IMAGE,
            embedding=pooled_embedding,
            features=embeddings,
            metadata=metadata,
            quality_score=quality_score.item(),
            confidence=quality_score.item()
        )


class AudioModalityEncoder(ModalityEncoder):
    """Audio modality encoder for sound processing."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        embedding_dim: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.embedding_dim = embedding_dim
        
        # Spectrogram parameters
        self.n_mels = 80
        
        # Mel-spectrogram conversion
        self.mel_conv = nn.Conv1d(1, embedding_dim, kernel_size=n_fft, stride=hop_length)
        
        # Audio transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 2,
                dropout=dropout,
                batch_first=True
            ) for _ in range(6)
        ])
        
        # Quality assessment
        self.quality_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"AudioModalityEncoder: {sample_rate}Hz, {n_fft}fft, {embedding_dim}dim")
    
    def forward(self, input_data: Union[np.ndarray, torch.Tensor]) -> ModalityEmbedding:
        """Encode audio input into embedding.
        
        Args:
            input_data: Audio input (numpy array or torch tensor)
            
        Returns:
            ModalityEmbedding with audio representation
        """
        # Handle different input formats
        if isinstance(input_data, np.ndarray):
            audio = torch.from_numpy(input_data).float().unsqueeze(0)  # Add channel dim
        else:
            if len(input_data.shape) == 1:
                audio = input_data.unsqueeze(0)  # Add channel dim
            else:
                audio = input_data
        
        # Convert to mel-spectrogram
        # In practice, use librosa, but we'll use a simple STFT approximation
        stft = torch.stft(
            audio.squeeze(1), 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True
        )
        magnitude = torch.abs(stft)
        
        # Convert to mel scale approximation
        mel_spec = magnitude[:, :self.n_mels, :]  # Take first n_mels frequency bins
        
        # Apply audio embedding
        audio_embeds = self.mel_conv(mel_spec).transpose(1, 2)  # (batch, time, embed_dim)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            audio_embeds = layer(audio_embeds)
        
        # Pool embeddings
        pooled_embedding = audio_embeds.mean(dim=1)  # Mean pooling
        
        # Assess quality
        quality_score = self.quality_predictor(pooled_embedding).squeeze(-1)
        
        # Metadata
        metadata = {
            'sample_rate': self.sample_rate,
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'n_mels': self.n_mels,
            'duration_samples': audio.shape[-1] if len(audio.shape) > 1 else None,
            'embedding_method': 'spectrogram_transformer_mean_pool'
        }
        
        return ModalityEmbedding(
            modality=ModalityType.AUDIO,
            embedding=pooled_embedding,
            features=audio_embeds,
            metadata=metadata,
            quality_score=quality_score.item(),
            confidence=quality_score.item()
        )


class SharedEmbeddingSpace(nn.Module):
    """
    Shared embedding space for unified multi-modal representation.
    
    Provides a common space where all modality embeddings can be projected
    for cross-modal comparison and fusion.
    """
    
    def __init__(
        self,
        input_dims: Dict[str, int],
        shared_dim: int = 512,
        use_modality_tokens: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_dims = input_dims
        self.shared_dim = shared_dim
        self.use_modality_tokens = use_modality_tokens
        self.dropout = nn.Dropout(dropout)
        
        # Modality-specific projections to shared space
        self.modality_projectors = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, shared_dim),
                nn.LayerNorm(shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for modality, dim in input_dims.items()
        })
        
        # Modality tokens for attention
        if use_modality_tokens:
            self.modality_tokens = nn.ParameterDict({
                modality: nn.Parameter(torch.randn(1, 1, shared_dim) * 0.02)
                for modality in input_dims.keys()
            })
        
        # Cross-modal attention for alignment
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            dropout=dropout
        )
        
        # Quality-based weighting
        self.quality_weights = nn.Parameter(torch.ones(len(input_dims)))
        
        logger.info(f"SharedEmbeddingSpace: {len(input_dims)} modalities -> {shared_dim} shared dim")
    
    def forward(
        self,
        modality_embeddings: Dict[str, ModalityEmbedding]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Project modality embeddings to shared space.
        
        Args:
            modality_embeddings: Dictionary of ModalityEmbedding objects
            
        Returns:
            Tuple of (shared_embeddings, metadata)
        """
        shared_embeddings = []
        alignment_weights = []
        
        # Project each modality to shared space
        for modality, embedding_obj in modality_embeddings.items():
            if modality not in self.modality_projectors:
                raise ValueError(f"Unknown modality: {modality}")
            
            # Project to shared space
            projected = self.modality_projectors[modality](embedding_obj.embedding)
            shared_embeddings.append(projected)
            
            # Apply quality-based weighting
            quality_weight = embedding_obj.quality_score
            weighted_projected = projected * quality_weight
            alignment_weights.append(quality_weight)
        
        # Stack embeddings
        stacked_embeddings = torch.stack(shared_embeddings, dim=1)  # (batch, num_modalities, shared_dim)
        
        # Apply cross-modal attention for alignment
        aligned_embeddings, attention_weights = self.alignment_attention(
            query=stacked_embeddings,
            key=stacked_embeddings,
            value=stacked_embeddings
        )
        
        # Compute unified representation
        quality_weights = F.softmax(self.quality_weights, dim=0)
        unified_representation = sum(
            weight * embed for weight, embed in zip(quality_weights, shared_embeddings)
        )
        
        # Add modality tokens if enabled
        if self.use_modality_tokens:
            modality_token_features = []
            for i, (modality, token) in enumerate(self.modality_tokens.items()):
                token_expanded = token.expand(stacked_embeddings.shape[0], -1, -1)
                modality_token_features.append(token_expanded)
            
            all_features = torch.cat([unified_representation.unsqueeze(1)] + modality_token_features, dim=1)
            unified_representation = all_features.mean(dim=1)
        
        metadata = {
            'modality_projections': shared_embeddings,
            'alignment_attention': attention_weights,
            'quality_weights': quality_weights,
            'unified_representation': unified_representation
        }
        
        return unified_representation, metadata


class CrossModalProjection(nn.Module):
    """
    Cross-modal projection networks for bidirectional mapping between modalities.
    
    Enables projection between different modality spaces with attention-based
    alignment and consistency constraints.
    """
    
    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        attention_heads: int = 8,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.dropout = nn.Dropout(dropout)
        
        # Projection layers
        self.projection_layers = nn.ModuleList()
        current_dim = source_dim
        
        for i in range(num_layers - 1):
            self.projection_layers.append(
                nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout)
                )
            )
            current_dim = hidden_dim
        
        # Final projection layer
        self.projection_layers.append(
            nn.Sequential(
                nn.Linear(current_dim, target_dim),
                nn.Dropout(dropout)
            )
        )
        
        # Cross-attention for alignment
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=attention_heads,
            dropout=dropout
        )
        
        # Consistency constraint
        self.consistency_network = nn.Sequential(
            nn.Linear(source_dim + target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"CrossModalProjection: {source_dim} -> {target_dim}, {num_layers} layers")
    
    def forward(
        self,
        source_embedding: torch.Tensor,
        target_space_features: Optional[torch.Tensor] = None,
        apply_consistency: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project source modality to target modality space.
        
        Args:
            source_embedding: Source modality embedding
            target_space_features: Optional target space features for attention
            apply_consistency: Whether to apply consistency constraints
            
        Returns:
            Tuple of (projected_embedding, consistency_score)
        """
        # Apply projection layers
        x = source_embedding
        for layer in self.projection_layers:
            x = layer(x)
        
        # Apply cross-attention if target features provided
        if target_space_features is not None:
            x = x.unsqueeze(0)  # Add sequence dimension
            attended, attention_weights = self.cross_attention(
                query=x,
                key=target_space_features.unsqueeze(0),
                value=target_space_features.unsqueeze(0)
            )
            x = attended.squeeze(0)
        
        # Apply consistency constraint
        consistency_score = 1.0
        if apply_consistency:
            concatenated = torch.cat([source_embedding, x], dim=-1)
            consistency_score = self.consistency_network(concatenated).squeeze(-1)
        
        return x, consistency_score
    
    def project_bidirectional(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project in both directions and compute consistency.
        
        Args:
            source_embedding: Source modality embedding
            target_embedding: Target modality embedding
            
        Returns:
            Tuple of (source_to_target, target_to_source, forward_consistency, backward_consistency)
        """
        # Forward projection
        source_to_target, forward_consistency = self.forward(
            source_embedding, target_embedding, apply_consistency=True
        )
        
        # Backward projection (assuming inverse dimensions)
        backward_projector = CrossModalProjection(
            source_dim=self.target_dim,
            target_dim=self.source_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers
        )
        target_to_source, backward_consistency = backward_projector.forward(
            target_embedding, source_embedding, apply_consistency=True
        )
        
        return source_to_target, target_to_source, forward_consistency, backward_consistency


class EmbeddingAlignment(nn.Module):
    """
    Embedding alignment module for ensuring consistency across modalities.
    
    Provides orthogonal constraints, attention-based alignment, and
    quality-aware consistency mechanisms.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        target_dim: int = 512,
        alignment_method: str = "orthogonal",
        use_attention: bool = True,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.target_dim = target_dim
        self.alignment_method = alignment_method
        self.use_attention = use_attention
        self.dropout = nn.Dropout(dropout)
        
        # Alignment projections
        self.alignment_projectors = nn.ModuleDict({
            modality: nn.Sequential(
                nn.Linear(dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.Dropout(dropout)
            ) for modality, dim in embedding_dims.items()
        })
        
        # Attention-based alignment
        if use_attention:
            self.alignment_attention = nn.MultiheadAttention(
                embed_dim=target_dim,
                num_heads=8,
                dropout=dropout
            )
        
        # Consistency constraints
        if alignment_method == "orthogonal":
            self.orthogonal_loss = nn.MSELoss()
        elif alignment_method == "cosine":
            self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # Quality-aware weighting
        self.quality_network = nn.Linear(target_dim, 1)
        
        logger.info(f"EmbeddingAlignment: {len(embedding_dims)} modalities, {alignment_method} method")
    
    def forward(
        self,
        modality_embeddings: Dict[str, ModalityEmbedding]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Align embeddings across modalities.
        
        Args:
            modality_embeddings: Dictionary of ModalityEmbedding objects
            
        Returns:
            Tuple of (aligned_embeddings, alignment_scores)
        """
        aligned_embeddings = {}
        alignment_scores = {}
        
        # Project all embeddings to target dimension
        for modality, embedding_obj in modality_embeddings.items():
            if modality not in self.alignment_projectors:
                raise ValueError(f"Unknown modality: {modality}")
            
            # Project to aligned space
            aligned = self.alignment_projectors[modality](embedding_obj.embedding)
            aligned_embeddings[modality] = aligned
        
        # Apply attention-based alignment
        if self.use_attention and len(aligned_embeddings) > 1:
            # Stack embeddings
            stacked = torch.stack(list(aligned_embeddings.values()), dim=1)
            
            # Apply cross-modal attention
            attended, attention_weights = self.alignment_attention(
                query=stacked,
                key=stacked,
                value=stacked
            )
            
            # Update aligned embeddings
            for i, (modality, aligned) in enumerate(aligned_embeddings.items()):
                aligned_embeddings[modality] = attended[:, i, :]
                
                # Compute alignment score
                quality_score = embedding_obj.quality_score
                attention_score = attention_weights[:, i, :].mean()
                alignment_scores[modality] = float(quality_score * attention_score.item())
        else:
            # Simple alignment without attention
            for modality, aligned in aligned_embeddings.items():
                quality_score = embedding_obj.quality_score
                alignment_scores[modality] = float(quality_score)
        
        return aligned_embeddings, alignment_scores
    
    def compute_alignment_loss(
        self,
        aligned_embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute alignment loss based on method.
        
        Args:
            aligned_embeddings: Aligned embeddings
            
        Returns:
            Alignment loss tensor
        """
        if self.alignment_method == "orthogonal":
            # Orthogonal constraint - embeddings should be diverse
            embeddings = list(aligned_embeddings.values())
            loss = 0.0
            
            for i, emb1 in enumerate(embeddings):
                for j, emb2 in enumerate(embeddings):
                    if i < j:
                        # Encourage orthogonality
                        dot_product = (emb1 * emb2).sum(dim=-1).mean()
                        loss = loss + dot_product**2
            
            return loss
            
        elif self.alignment_method == "cosine":
            # Cosine similarity constraint
            embeddings = list(aligned_embeddings.values())
            loss = 0.0
            
            for i, emb1 in enumerate(embeddings):
                for j, emb2 in enumerate(embeddings):
                    if i < j:
                        cosine_sim = F.cosine_similarity(emb1, emb2, dim=-1)
                        # Encourage moderate similarity (not too high, not too low)
                        target_sim = 0.5
                        loss = loss + (cosine_sim - target_sim)**2
            
            return loss
        
        return torch.tensor(0.0)


class UnifiedRepresentation(nn.Module):
    """
    Unified representation module combining all embedding and fusion components.
    
    Provides end-to-end processing from raw multi-modal inputs to unified
    representations with quality assessment and biological constraints.
    """
    
    def __init__(
        self,
        embedding_dims: Dict[str, int],
        shared_dim: int = 512,
        fusion_method: str = "adaptive",
        alignment_method: str = "orthogonal",
        use_biological_constraints: bool = True,
        **kwargs
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.shared_dim = shared_dim
        self.fusion_method = fusion_method
        self.alignment_method = alignment_method
        self.use_biological_constraints = use_biological_constraints
        
        # Initialize modality encoders
        self.modality_encoders = nn.ModuleDict({
            'text': TextModalityEncoder(embedding_dim=embedding_dims.get('text', 512)),
            'image': ImageModalityEncoder(embedding_dim=embedding_dims.get('image', 512)),
            'audio': AudioModalityEncoder(embedding_dim=embedding_dims.get('audio', 512))
        })
        
        # Initialize embedding space
        self.shared_space = SharedEmbeddingSpace(
            input_dims=embedding_dims,
            shared_dim=shared_dim
        )
        
        # Initialize cross-modal projections
        self.cross_projections = nn.ModuleDict()
        modalities = list(embedding_dims.keys())
        for i, src_mod in enumerate(modalities):
            for j, tgt_mod in enumerate(modalities):
                if i != j:
                    self.cross_projections[f"{src_mod}_to_{tgt_mod}"] = CrossModalProjection(
                        source_dim=embedding_dims[src_mod],
                        target_dim=embedding_dims[tgt_mod]
                    )
        
        # Initialize alignment module
        self.alignment_module = EmbeddingAlignment(
            embedding_dims=embedding_dims,
            target_dim=shared_dim,
            alignment_method=alignment_method
        )
        
        # Biological constraints
        if use_biological_constraints:
            self.biological_constraints = nn.ModuleDict({
                'sparsity_regularizer': nn.Linear(shared_dim, 1),
                'attention_regularizer': nn.Linear(shared_dim, 1),
                'plasticity_regularizer': nn.Linear(shared_dim, 1)
            })
        
        # Quality assessment
        self.quality_assessor = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"UnifiedRepresentation initialized: {len(embedding_dims)} modalities")
    
    def forward(
        self,
        modality_inputs: Dict[str, Any],
        apply_constraints: bool = True
    ) -> Tuple[UnifiedRepresentation, Dict[str, Any]]:
        """Forward pass of unified representation learning.
        
        Args:
            modality_inputs: Dictionary of raw modality inputs
            apply_constraints: Whether to apply biological constraints
            
        Returns:
            Tuple of (unified_representation, metadata)
        """
        # Encode each modality
        modality_embeddings = {}
        for modality, encoder in self.modality_encoders.items():
            if modality in modality_inputs:
                try:
                    embedding_obj = encoder(modality_inputs[modality])
                    modality_embeddings[modality] = embedding_obj
                except Exception as e:
                    logger.warning(f"Failed to encode {modality}: {e}")
        
        if not modality_embeddings:
            raise ValueError("No valid modality embeddings computed")
        
        # Align embeddings
        aligned_embeddings, alignment_scores = self.alignment_module(modality_embeddings)
        
        # Project to shared space
        shared_representation, shared_metadata = self.shared_space(aligned_embeddings)
        
        # Apply cross-modal projections
        cross_projections = {}
        for projection_name, projection_layer in self.cross_projections.items():
            src_mod, tgt_mod = projection_name.split('_to_')
            if src_mod in aligned_embeddings and tgt_mod in aligned_embeddings:
                src_embedding = aligned_embeddings[src_mod]
                tgt_embedding = aligned_embeddings[tgt_mod]
                
                projected, consistency = projection_layer.forward(
                    src_embedding, tgt_embedding, apply_consistency=True
                )
                cross_projections[projection_name] = {
                    'projected': projected,
                    'consistency': consistency
                }
        
        # Assess overall quality
        overall_quality = self.quality_assessor(shared_representation).squeeze(-1)
        
        # Apply biological constraints
        constraint_losses = {}
        if apply_constraints and self.use_biological_constraints:
            sparsity_loss = F.relu(self.biological_constraints['sparsity_regularizer'](shared_representation)).mean()
            attention_loss = F.relu(self.biological_constraints['attention_regularizer'](shared_representation)).mean()
            plasticity_loss = F.relu(self.biological_constraints['plasticity_regularizer'](shared_representation)).mean()
            
            constraint_losses = {
                'sparsity': sparsity_loss,
                'attention': attention_loss,
                'plasticity': plasticity_loss
            }
        
        # Create unified representation object
        unified_repr = UnifiedRepresentation(
            representation=shared_representation,
            modality_contributions={
                mod: float(emb.quality_score) for mod, emb in modality_embeddings.items()
            },
            alignment_scores=alignment_scores,
            coherence_metrics={
                'overall_quality': float(overall_quality.item()),
                'constraint_losses': {k: float(v.item()) for k, v in constraint_losses.items()}
            } if constraint_losses else {'overall_quality': float(overall_quality.item())},
            semantic_consistency=float(overall_quality.item())
        )
        
        # Compile metadata
        metadata = {
            'modality_embeddings': modality_embeddings,
            'aligned_embeddings': aligned_embeddings,
            'shared_metadata': shared_metadata,
            'cross_projections': cross_projections,
            'alignment_scores': alignment_scores,
            'overall_quality': float(overall_quality.item()),
            'constraint_losses': constraint_losses,
            'fusion_method': self.fusion_method,
            'alignment_method': self.alignment_method
        }
        
        return unified_repr, metadata


# Utility functions

def create_modality_encoder(
    modality: ModalityType,
    **kwargs
) -> ModalityEncoder:
    """Factory function to create modality encoders.
    
    Args:
        modality: Type of modality encoder
        **kwargs: Configuration parameters
        
    Returns:
        Configured modality encoder
    """
    encoder_classes = {
        ModalityType.TEXT: TextModalityEncoder,
        ModalityType.IMAGE: ImageModalityEncoder,
        ModalityType.AUDIO: AudioModalityEncoder
    }
    
    if modality not in encoder_classes:
        raise ValueError(f"Unknown modality: {modality}")
    
    return encoder_classes[modality](**kwargs)


def evaluate_embedding_quality(
    embeddings: Dict[str, ModalityEmbedding]
) -> Dict[str, float]:
    """Evaluate the quality of embeddings.
    
    Args:
        embeddings: Dictionary of modality embeddings
        
    Returns:
        Dictionary of quality metrics
    """
    quality_metrics = {}
    
    # Individual embedding quality
    for modality, embedding_obj in embeddings.items():
        quality_metrics[f"{modality}_quality"] = embedding_obj.quality_score
        quality_metrics[f"{modality}_confidence"] = embedding_obj.confidence
    
    # Cross-modal consistency
    embedding_vectors = [emb.embedding for emb in embeddings.values()]
    if len(embedding_vectors) > 1:
        # Compute pairwise similarities
        similarities = []
        for i, emb1 in enumerate(embedding_vectors):
            for j, emb2 in enumerate(embedding_vectors):
                if i < j:
                    sim = F.cosine_similarity(emb1, emb2, dim=-1).mean()
                    similarities.append(sim.item())
        
        quality_metrics['cross_modal_consistency'] = np.mean(similarities)
    
    return quality_metrics


# Export all classes and functions
__all__ = [
    'ModalityEncoder',
    'TextModalityEncoder',
    'ImageModalityEncoder', 
    'AudioModalityEncoder',
    'SharedEmbeddingSpace',
    'CrossModalProjection',
    'EmbeddingAlignment',
    'UnifiedRepresentation',
    'create_modality_encoder',
    'evaluate_embedding_quality'
]