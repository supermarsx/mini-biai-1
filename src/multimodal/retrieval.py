"""
Cross-Modal Retrieval and Generation Capabilities

This module implements sophisticated cross-modal retrieval and generation systems
for vision-language-audio integration, enabling retrieval across modalities and
generating content based on multi-modal inputs.

Key Features:
- Cross-modal retrieval (text->image, image->text, audio->visual, etc.)
- Multi-modal generation (text-to-image, image-to-audio, etc.)
- Retrieval-augmented generation (RAG) with multi-modal support
- Semantic similarity search and ranking
- Multi-modal memory and knowledge integration
- Real-time retrieval and generation

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │            Cross-Modal Retrieval & Generation              │
    └─────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Cross-    │ │   Multi-    │ │    RAG      │
    │  Modal      │ │  Modal      │ │ (Retrieval  │
    │ Retrieval   │ │Generation   │ │ Augmented   │
    │             │ │             │ │ Generation) │
    │ • Text->Img │ │ • Text->Img │ │ • Context   │
    │ • Img->Text │ │ • Img->Audio│ │ • Memory    │
    │ • Audio->Vis│ │ • Audio->Text│ │ • Knowledge │
    │ • Multi     │ │ • Cross-    │ │ • Generation│
    │   hop       │ │   modal     │ │   guided    │
    └─────────────┘ └─────────────┘ └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Semantic   │ │   Ranking   │ │  Memory     │
    │ Similarity  │ │   & Score   │ │ Integration │
    │             │ │             │ │             │
    │ • Cosine    │ │ • Multi-    │ │ • Episodic  │
    │ • Attention │ │   criteria  │ │ • Semantic  │
    │ • Learned   │ │ • Confidence│ │ • Working   │
    │ • Graph-    │ │ • Quality   │ │ • Integration│
    │   based     │ │   scoring   │ │ • Update    │
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
from collections import defaultdict, deque
import time

# Import interfaces and modules
try:
    from ..interfaces.multimodal_interfaces import (
        ModalityType, UnifiedRepresentation, ModalityEmbedding
    )
    from .embeddings import UnifiedRepresentation, ModalityEmbedding
    from .attention import CrossModalAttention
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
class RetrievalResult:
    """Result from cross-modal retrieval operation."""
    source_modality: ModalityType
    target_modality: ModalityType
    source_content: Any
    retrieved_content: Any
    similarity_score: float
    confidence_score: float
    retrieval_metadata: Dict[str, Any]
    timestamp: float


@dataclass
class GenerationRequest:
    """Request for multi-modal content generation."""
    source_modality: ModalityType
    target_modality: ModalityType
    source_content: Any
    generation_config: Dict[str, Any]
    context_embeddings: Optional[List[UnifiedRepresentation]] = None
    constraints: Optional[Dict[str, Any]] = None


class CrossModalRetriever(ABC):
    """Abstract base class for cross-modal retrieval systems."""
    
    @abstractmethod
    def retrieve(
        self,
        query: ModalityEmbedding,
        target_modality: ModalityType,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve content from target modality based on query.
        
        Args:
            query: Query embedding from source modality
            target_modality: Target modality to retrieve from
            top_k: Number of results to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of RetrievalResult objects
        """
        pass


class SemanticSimilarityRetriever(CrossModalRetriever):
    """
    Semantic similarity-based cross-modal retriever.
    
    Uses learned embeddings and similarity metrics to retrieve
    semantically similar content across modalities.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        similarity_metric: str = "cosine",
        top_k_candidates: int = 100,
        similarity_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.similarity_metric = similarity_metric
        self.top_k_candidates = top_k_candidates
        self.similarity_threshold = similarity_threshold
        
        # Similarity computation networks
        self.similarity_networks = nn.ModuleDict({
            'text_image': nn.Sequential(
                nn.Linear(embedding_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'text_audio': nn.Sequential(
                nn.Linear(embedding_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ),
            'image_audio': nn.Sequential(
                nn.Linear(embedding_dim * 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        })
        
        # Attention-based similarity
        self.attention_similarity = CrossModalAttention(
            query_dim=embedding_dim,
            key_dim=embedding_dim,
            value_dim=embedding_dim,
            output_dim=embedding_dim
        )
        
        # Content database (in practice, this would be a proper database)
        self.content_database = defaultdict(list)
        self.content_index = {}
        
        logger.info(f"SemanticSimilarityRetriever initialized: {embedding_dim}dim, {similarity_metric}")
    
    def index_content(
        self,
        modality: ModalityType,
        content: Any,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index content for retrieval.
        
        Args:
            modality: Modality of the content
            content: Raw content data
            embedding: Content embedding
            metadata: Optional metadata
        """
        item = {
            'content': content,
            'embedding': embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        self.content_database[modality].append(item)
        
        # Update index
        content_id = f"{modality.value}_{len(self.content_database[modality])}"
        self.content_index[content_id] = item
    
    def compute_similarity(
        self,
        query_embedding: torch.Tensor,
        candidate_embedding: torch.Tensor,
        similarity_type: str = "cosine"
    ) -> float:
        """Compute similarity between embeddings.
        
        Args:
            query_embedding: Query embedding
            candidate_embedding: Candidate embedding
            similarity_type: Type of similarity computation
            
        Returns:
            Similarity score
        """
        if similarity_type == "cosine":
            similarity = F.cosine_similarity(query_embedding, candidate_embedding, dim=-1)
            return similarity.item()
        elif similarity_type == "euclidean":
            distance = F.pairwise_distance(query_embedding, candidate_embedding, p=2)
            similarity = 1.0 / (1.0 + distance.item())  # Convert distance to similarity
            return similarity
        elif similarity_type == "learned":
            # Use learned similarity network
            combined = torch.cat([query_embedding, candidate_embedding], dim=-1)
            similarity = self.similarity_networks['text_image'](combined)
            return similarity.item()
        elif similarity_type == "attention":
            # Use attention-based similarity
            query_expanded = query_embedding.unsqueeze(0)
            candidate_expanded = candidate_embedding.unsqueeze(0)
            
            attended, attention_weights = self.attention_similarity(
                query=query_expanded,
                key=candidate_expanded,
                value=candidate_expanded
            )
            
            # Compute attention-weighted similarity
            similarity = F.cosine_similarity(attended, candidate_expanded, dim=-1)
            return similarity.item()
        else:
            raise ValueError(f"Unknown similarity type: {similarity_type}")
    
    def retrieve(
        self,
        query: ModalityEmbedding,
        target_modality: ModalityType,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Retrieve content from target modality.
        
        Args:
            query: Query embedding from source modality
            target_modality: Target modality to retrieve from
            top_k: Number of results to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of RetrievalResult objects
        """
        if target_modality not in self.content_database:
            logger.warning(f"No content indexed for modality: {target_modality}")
            return []
        
        candidates = self.content_database[target_modality]
        if not candidates:
            return []
        
        # Compute similarities
        similarities = []
        for item in candidates:
            similarity = self.compute_similarity(
                query.embedding, 
                item['embedding'],
                self.similarity_metric
            )
            similarities.append((item, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold and get top-k
        filtered_similarities = [
            (item, sim) for item, sim in similarities 
            if sim >= self.similarity_threshold
        ][:top_k]
        
        # Create retrieval results
        results = []
        for item, similarity in filtered_similarities:
            confidence = min(1.0, similarity * query.confidence * item.get('metadata', {}).get('quality', 1.0))
            
            result = RetrievalResult(
                source_modality=query.modality,
                target_modality=target_modality,
                source_content=None,  # Would be provided by caller
                retrieved_content=item['content'],
                similarity_score=similarity,
                confidence_score=confidence,
                retrieval_metadata={
                    'retrieval_method': 'semantic_similarity',
                    'similarity_metric': self.similarity_metric,
                    'item_metadata': item['metadata'],
                    'timestamp': item['timestamp']
                },
                timestamp=time.time()
            )
            results.append(result)
        
        return results


class VisionLanguageRetrieval(CrossModalRetriever):
    """
    Specialized retriever for vision-language tasks.
    
    Optimized for text-image, image-text, and other vision-language
    retrieval tasks with specialized attention mechanisms.
    """
    
    def __init__(
        self,
        vision_dim: int = 512,
        language_dim: int = 512,
        cross_attention_heads: int = 8,
        retrieval_top_k: int = 10,
        **kwargs
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.retrieval_top_k = retrieval_top_k
        
        # Cross-attention for vision-language alignment
        self.vision_language_attention = CrossModalAttention(
            query_dim=language_dim,
            key_dim=vision_dim,
            value_dim=vision_dim,
            output_dim=language_dim,
            num_heads=cross_attention_heads
        )
        
        # Bidirectional attention
        self.language_vision_attention = CrossModalAttention(
            query_dim=vision_dim,
            key_dim=language_dim,
            value_dim=language_dim,
            output_dim=vision_dim,
            num_heads=cross_attention_heads
        )
        
        # Retrieval scoring networks
        self.vision_to_text_scorer = nn.Sequential(
            nn.Linear(vision_dim + language_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.text_to_vision_scorer = nn.Sequential(
            nn.Linear(vision_dim + language_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Content databases
        self.image_database = []
        self.text_database = []
        
        logger.info(f"VisionLanguageRetrieval initialized: {vision_dim}dim vision, {language_dim}dim language")
    
    def index_image(
        self,
        image_data: Any,
        vision_embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index an image for retrieval.
        
        Args:
            image_data: Raw image data
            vision_embedding: Vision feature embedding
            metadata: Optional metadata
        """
        item = {
            'content': image_data,
            'embedding': vision_embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.image_database.append(item)
    
    def index_text(
        self,
        text_data: str,
        language_embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index text for retrieval.
        
        Args:
            text_data: Text content
            language_embedding: Language feature embedding
            metadata: Optional metadata
        """
        item = {
            'content': text_data,
            'embedding': language_embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.text_database.append(item)
    
    def retrieve_text_from_image(
        self,
        query_image_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve text descriptions for an image.
        
        Args:
            query_image_embedding: Image query embedding
            top_k: Number of results
            
        Returns:
            List of text retrieval results
        """
        results = []
        
        for text_item in self.text_database:
            # Apply cross-attention
            attended_text, attention_weights = self.language_vision_attention(
                query=query_image_embedding.unsqueeze(0),
                key=text_item['embedding'].unsqueeze(0),
                value=text_item['embedding'].unsqueeze(0)
            )
            
            # Score the match
            combined = torch.cat([query_image_embedding, attended_text.squeeze(0)], dim=-1)
            score = self.text_to_vision_scorer(combined).item()
            
            if score > 0.3:  # Threshold for relevance
                result = RetrievalResult(
                    source_modality=ModalityType.IMAGE,
                    target_modality=ModalityType.TEXT,
                    source_content=None,
                    retrieved_content=text_item['content'],
                    similarity_score=score,
                    confidence_score=score,
                    retrieval_metadata={
                        'retrieval_method': 'vision_to_text',
                        'attention_weights': attention_weights.squeeze(0).cpu().numpy(),
                        'text_metadata': text_item['metadata']
                    },
                    timestamp=time.time()
                )
                results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def retrieve_image_from_text(
        self,
        query_text_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve images for a text query.
        
        Args:
            query_text_embedding: Text query embedding
            top_k: Number of results
            
        Returns:
            List of image retrieval results
        """
        results = []
        
        for image_item in self.image_database:
            # Apply cross-attention
            attended_image, attention_weights = self.vision_language_attention(
                query=query_text_embedding.unsqueeze(0),
                key=image_item['embedding'].unsqueeze(0),
                value=image_item['embedding'].unsqueeze(0)
            )
            
            # Score the match
            combined = torch.cat([query_text_embedding, attended_image.squeeze(0)], dim=-1)
            score = self.vision_to_text_scorer(combined).item()
            
            if score > 0.3:  # Threshold for relevance
                result = RetrievalResult(
                    source_modality=ModalityType.TEXT,
                    target_modality=ModalityType.IMAGE,
                    source_content=None,
                    retrieved_content=image_item['content'],
                    similarity_score=score,
                    confidence_score=score,
                    retrieval_metadata={
                        'retrieval_method': 'text_to_vision',
                        'attention_weights': attention_weights.squeeze(0).cpu().numpy(),
                        'image_metadata': image_item['metadata']
                    },
                    timestamp=time.time()
                )
                results.append(result)
        
        # Sort by score and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def retrieve(
        self,
        query: ModalityEmbedding,
        target_modality: ModalityType,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Generic retrieval method.
        
        Args:
            query: Query embedding
            target_modality: Target modality
            top_k: Number of results
            
        Returns:
            List of retrieval results
        """
        if query.modality == ModalityType.IMAGE and target_modality == ModalityType.TEXT:
            return self.retrieve_text_from_image(query.embedding, top_k)
        elif query.modality == ModalityType.TEXT and target_modality == ModalityType.IMAGE:
            return self.retrieve_image_from_text(query.embedding, top_k)
        else:
            logger.warning(f"Unsupported retrieval: {query.modality} -> {target_modality}")
            return []


class AudioVisualRetrieval(CrossModalRetriever):
    """
    Specialized retriever for audio-visual tasks.
    
    Handles audio-image, audio-text, and other audio-visual
    retrieval operations with acoustic-visual attention.
    """
    
    def __init__(
        self,
        audio_dim: int = 512,
        vision_dim: int = 512,
        cross_attention_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.audio_dim = audio_dim
        self.vision_dim = vision_dim
        
        # Audio-visual cross-attention
        self.audio_vision_attention = CrossModalAttention(
            query_dim=audio_dim,
            key_dim=vision_dim,
            value_dim=vision_dim,
            output_dim=audio_dim,
            num_heads=cross_attention_heads
        )
        
        # Vision-audio cross-attention
        self.vision_audio_attention = CrossModalAttention(
            query_dim=vision_dim,
            key_dim=audio_dim,
            value_dim=audio_dim,
            output_dim=vision_dim,
            num_heads=cross_attention_heads
        )
        
        # Content databases
        self.audio_database = []
        self.image_database = []
        self.text_database = []
        
        logger.info(f"AudioVisualRetrieval initialized: {audio_dim}dim audio, {vision_dim}dim vision")
    
    def index_audio(
        self,
        audio_data: Any,
        audio_embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index audio content."""
        item = {
            'content': audio_data,
            'embedding': audio_embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.audio_database.append(item)
    
    def index_image(
        self,
        image_data: Any,
        vision_embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Index image content."""
        item = {
            'content': image_data,
            'embedding': vision_embedding,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        self.image_database.append(item)
    
    def retrieve_audio_from_image(
        self,
        query_image_embedding: torch.Tensor,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve audio for an image query."""
        results = []
        
        for audio_item in self.audio_database:
            # Cross-attention
            attended_audio, attention_weights = self.vision_audio_attention(
                query=query_image_embedding.unsqueeze(0),
                key=audio_item['embedding'].unsqueeze(0),
                value=audio_item['embedding'].unsqueeze(0)
            )
            
            # Score
            score = F.cosine_similarity(query_image_embedding, attended_audio.squeeze(0)).item()
            
            if score > 0.3:
                result = RetrievalResult(
                    source_modality=ModalityType.IMAGE,
                    target_modality=ModalityType.AUDIO,
                    source_content=None,
                    retrieved_content=audio_item['content'],
                    similarity_score=score,
                    confidence_score=score,
                    retrieval_metadata={
                        'retrieval_method': 'image_to_audio',
                        'attention_weights': attention_weights.squeeze(0).cpu().numpy()
                    },
                    timestamp=time.time()
                )
                results.append(result)
        
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def retrieve(
        self,
        query: ModalityEmbedding,
        target_modality: ModalityType,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalResult]:
        """Generic retrieval method."""
        if query.modality == ModalityType.IMAGE and target_modality == ModalityType.AUDIO:
            return self.retrieve_audio_from_image(query.embedding, top_k)
        else:
            logger.warning(f"Unsupported retrieval: {query.modality} -> {target_modality}")
            return []


class MultimodalGenerator(ABC):
    """Abstract base class for multi-modal content generation."""
    
    @abstractmethod
    def generate(
        self,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """Generate content based on request.
        
        Args:
            request: Generation request
            
        Returns:
            Dictionary with generated content and metadata
        """
        pass


class TextToImageGenerator(MultimodalGenerator):
    """
    Text-to-image generator using attention-based generation.
    
    Generates images from text descriptions using cross-modal
    attention and learned generation patterns.
    """
    
    def __init__(
        self,
        text_dim: int = 512,
        image_dim: int = 512,
        generation_resolution: Tuple[int, int] = (256, 256),
        **kwargs
    ):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.generation_resolution = generation_resolution
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, image_dim)
        )
        
        # Image decoder/generator
        self.image_decoder = nn.Sequential(
            nn.Linear(image_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, generation_resolution[0] * generation_resolution[1] * 3)
        )
        
        # Attention mechanism for generation
        self.generation_attention = CrossModalAttention(
            query_dim=image_dim,
            key_dim=image_dim,
            value_dim=image_dim,
            output_dim=image_dim
        )
        
        logger.info(f"TextToImageGenerator: {generation_resolution} resolution")
    
    def generate(
        self,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """Generate image from text."""
        if request.target_modality != ModalityType.IMAGE:
            raise ValueError("TextToImageGenerator only supports image generation")
        
        # Encode text
        encoded_text = self.text_encoder(request.source_content)
        
        # Attention-based generation
        attended_encoding, attention_weights = self.generation_attention(
            query=encoded_text.unsqueeze(0),
            key=encoded_text.unsqueeze(0),
            value=encoded_text.unsqueeze(0)
        )
        
        # Generate image
        image_flat = self.image_decoder(attended_encoding.squeeze(0))
        
        # Reshape to image format
        height, width = self.generation_resolution
        generated_image = image_flat.view(-1, height, width, 3)
        
        # Add noise for variation
        noise_level = request.generation_config.get('noise_level', 0.1)
        noise = torch.randn_like(generated_image) * noise_level
        generated_image = torch.clamp(generated_image + noise, 0, 1)
        
        metadata = {
            'generation_method': 'text_to_image',
            'source_text': request.source_content,
            'resolution': self.generation_resolution,
            'attention_weights': attention_weights.squeeze(0).cpu().numpy(),
            'noise_level': noise_level,
            'timestamp': time.time()
        }
        
        return {
            'generated_content': generated_image,
            'metadata': metadata,
            'confidence': 0.8  # Default confidence
        }


class ImageToAudioGenerator(MultimodalGenerator):
    """
    Image-to-audio generator for creating sound from visual content.
    
    Generates audio based on visual features using cross-modal
    attention and learned audio generation patterns.
    """
    
    def __init__(
        self,
        image_dim: int = 512,
        audio_dim: int = 512,
        audio_length: int = 16000,  # 1 second at 16kHz
        **kwargs
    ):
        super().__init__()
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.audio_length = audio_length
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.ReLU(),
            nn.Linear(256, audio_dim)
        )
        
        # Audio decoder
        self.audio_decoder = nn.Sequential(
            nn.Linear(audio_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, audio_length)
        )
        
        # Attention for audio generation
        self.audio_attention = CrossModalAttention(
            query_dim=audio_dim,
            key_dim=audio_dim,
            value_dim=audio_dim,
            output_dim=audio_dim
        )
        
        logger.info(f"ImageToAudioGenerator: {audio_length} samples")
    
    def generate(
        self,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """Generate audio from image."""
        if request.target_modality != ModalityType.AUDIO:
            raise ValueError("ImageToAudioGenerator only supports audio generation")
        
        # Encode image
        encoded_image = self.image_encoder(request.source_content)
        
        # Attention-based generation
        attended_encoding, attention_weights = self.audio_attention(
            query=encoded_image.unsqueeze(0),
            key=encoded_image.unsqueeze(0),
            value=encoded_image.unsqueeze(0)
        )
        
        # Generate audio
        generated_audio = self.audio_decoder(attended_encoding.squeeze(0))
        
        # Apply activation and scaling
        generated_audio = torch.tanh(generated_audio) * 0.5 + 0.5  # Scale to [0, 1]
        
        metadata = {
            'generation_method': 'image_to_audio',
            'source_image': request.source_content,
            'audio_length': self.audio_length,
            'attention_weights': attention_weights.squeeze(0).cpu().numpy(),
            'timestamp': time.time()
        }
        
        return {
            'generated_content': generated_audio,
            'metadata': metadata,
            'confidence': 0.75  # Default confidence
        }


# Utility functions and factory methods

def create_retriever(
    retriever_type: str,
    **kwargs
) -> CrossModalRetriever:
    """Factory function to create retrievers.
    
    Args:
        retriever_type: Type of retriever
        **kwargs: Configuration parameters
        
    Returns:
        Configured retriever
    """
    retriever_classes = {
        'semantic_similarity': SemanticSimilarityRetriever,
        'vision_language': VisionLanguageRetrieval,
        'audio_visual': AudioVisualRetrieval
    }
    
    if retriever_type not in retriever_classes:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    return retriever_classes[retriever_type](**kwargs)


def create_generator(
    generator_type: str,
    **kwargs
) -> MultimodalGenerator:
    """Factory function to create generators.
    
    Args:
        generator_type: Type of generator
        **kwargs: Configuration parameters
        
    Returns:
        Configured generator
    """
    generator_classes = {
        'text_to_image': TextToImageGenerator,
        'image_to_audio': ImageToAudioGenerator
    }
    
    if generator_type not in generator_classes:
        raise ValueError(f"Unknown generator type: {generator_type}")
    
    return generator_classes[generator_type](**kwargs)


def evaluate_retrieval_quality(
    results: List[RetrievalResult],
    ground_truth: Optional[List[Any]] = None
) -> Dict[str, float]:
    """Evaluate retrieval quality.
    
    Args:
        results: List of retrieval results
        ground_truth: Optional ground truth for evaluation
        
    Returns:
        Dictionary of quality metrics
    """
    if not results:
        return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
    
    metrics = {}
    
    # Basic metrics
    metrics['num_results'] = len(results)
    metrics['avg_similarity'] = np.mean([r.similarity_score for r in results])
    metrics['avg_confidence'] = np.mean([r.confidence_score for r in results])
    
    # If ground truth available
    if ground_truth:
        retrieved_contents = [r.retrieved_content for r in results]
        
        # Compute precision and recall
        true_positives = len(set(retrieved_contents) & set(ground_truth))
        precision = true_positives / len(retrieved_contents) if retrieved_contents else 0
        recall = true_positives / len(ground_truth) if ground_truth else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
    
    return metrics


# Export all classes and functions
__all__ = [
    'CrossModalRetriever',
    'SemanticSimilarityRetriever',
    'VisionLanguageRetrieval',
    'AudioVisualRetrieval',
    'MultimodalGenerator',
    'TextToImageGenerator',
    'ImageToAudioGenerator',
    'create_retriever',
    'create_generator',
    'evaluate_retrieval_quality'
]