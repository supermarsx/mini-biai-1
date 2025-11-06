"""
SSM Text Processor - Advanced Implementation

This module provides a complete SSM-based text processor that integrates
State Space Models, linear attention, and spiking dynamics for efficient
and biologically-inspired text processing.

Key Features:
- SSM-based text encoding with linear complexity
- Linear attention mechanisms for adaptive processing
- Spiking output layers for biological realism
- Memory integration for persistent text representations
- Hardware optimization (CUDA, MPS, CPU fallbacks)
- Performance monitoring and adaptive mechanisms
- Comprehensive error handling and fallbacks

Author: mini-biai-1 Team
License: MIT
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

# Import our custom components
from .ssm_backbone import SSMBackbone, SSMConfig, SSMType, HardwareType as SSMHardwareType
from .linear_attention import MultiHeadLinearAttention, LinearAttentionConfig, LinearAttentionType
from .hybrid_processor import HybridProcessor, HybridProcessorConfig, HybridProcessingMode


class TextEncodingType(Enum):
    """Types of text encoding supported"""
    SSM_ONLY = "ssm_only"
    LINEAR_ATTENTION = "linear_attention"
    HYBRID = "hybrid"
    SPIKING_HYBRID = "spiking_hybrid"


@dataclass
class SSMTextConfig:
    """Configuration for SSM text processor"""
    # Model architecture
    hidden_size: int = 256
    embedding_dim: int = 128  # Final embedding dimension
    vocab_size: int = 30000  # Vocabulary size for token embeddings
    
    # SSM configuration
    sequence_length: int = 512
    ssm_config: Optional[SSMConfig] = None
    ssm_state_size: int = 128
    ssm_type: SSMType = SSMType.HIPPO
    
    # Linear attention configuration
    attention_config: Optional[LinearAttentionConfig] = None
    attention_type: LinearAttentionType = LinearAttentionType.PERFORMER
    num_attention_heads: int = 8
    
    # Hybrid processing
    hybrid_config: Optional[HybridProcessorConfig] = None
    processing_mode: HybridProcessingMode = HybridProcessingMode.ADAPTIVE
    encoding_type: TextEncodingType = TextEncodingType.HYBRID
    
    # Hardware configuration
    hardware_type: SSMHardwareType = SSMHardwareType.AUTO
    use_mixed_precision: bool = True
    
    # Memory integration
    memory_integration: bool = True
    memory_keys: List[str] = None
    
    # Spiking integration
    spiking_enabled: bool = True
    spike_threshold: float = 0.5
    spike_decay: float = 0.95
    
    # Performance monitoring
    enable_profiling: bool = True
    adaptive_processing: bool = True
    
    # Tokenization (simplified for SSM)
    max_tokens: int = 512
    token_embedding_dim: int = 128


class MemoryIntegrationInterface(ABC):
    """Enhanced memory integration interface for SSM text processor"""
    
    @abstractmethod
    async def store_encoding(self, key: str, encoding: 'SSMTextEncoding') -> bool:
        """Store text encoding in memory"""
        pass
    
    @abstractmethod
    async def retrieve_encoding(self, key: str) -> Optional['SSMTextEncoding']:
        """Retrieve text encoding from memory"""
        pass
    
    @abstractmethod
    async def find_similar(self, encoding: 'SSMTextEncoding', threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar encodings in memory"""
        pass
    
    @abstractmethod
    async def update_memory_context(self, context: torch.Tensor):
        """Update global memory context"""
        pass
    
    @abstractmethod
    async def get_memory_context(self) -> torch.Tensor:
        """Get global memory context"""
        pass


@dataclass
class SSMTextEncoding:
    """Enhanced text encoding with SSM features"""
    embeddings: np.ndarray
    ssm_states: Optional[np.ndarray] = None  # SSM hidden states
    attention_weights: Optional[np.ndarray] = None  # Attention weights
    spiking_patterns: Optional[np.ndarray] = None  # Spiking patterns
    memory_context: Optional[np.ndarray] = None  # Memory context
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TokenEmbedding(nn.Module):
    """Token embedding layer for text processing"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(2048, embedding_dim)  # Max sequence length
        
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for token embeddings"""
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(token_ids)
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        # Combine embeddings
        embeddings = token_embeds + position_embeds
        
        return embeddings


class SSMTextProcessor:
    """
    Advanced SSM-based text processor with integrated memory and spiking features
    
    This processor provides:
    - Efficient SSM-based text encoding
    - Linear attention mechanisms
    - Spiking output layers
    - Memory integration
    - Hardware optimization
    - Performance monitoring
    """
    
    def __init__(self, config: SSMTextConfig, memory_interface: Optional[MemoryIntegrationInterface] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.memory = memory_interface
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize components
        self._initialize_components()
        
        # Performance monitoring
        self.performance_stats = {
            'encoding_time': [],
            'decoding_time': [],
            'memory_usage': [],
            'sequence_lengths': []
        }
        
        self.logger.info(f"SSM Text Processor initialized on {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup appropriate device"""
        # Convert SSMHardwareType to string for device creation
        hardware_type = self.config.hardware_type
        
        if hardware_type == SSMHardwareType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            # Map enum values to device strings
            device_map = {
                SSMHardwareType.CUDA: "cuda",
                SSMHardwareType.MPS: "mps",
                SSMHardwareType.CPU: "cpu"
            }
            return torch.device(device_map[hardware_type])
    
    def _initialize_components(self):
        """Initialize SSM and attention components"""
        # Token embedding layer
        self.token_embedding = TokenEmbedding(
            self.config.vocab_size, 
            self.config.token_embedding_dim
        )
        
        # Projection to hidden size
        self.input_projection = nn.Linear(
            self.config.token_embedding_dim, 
            self.config.hidden_size
        )
        
        # Initialize SSM configuration
        if self.config.ssm_config is None:
            self.config.ssm_config = SSMConfig(
                hidden_size=self.config.hidden_size,
                sequence_length=self.config.sequence_length,
                state_size=self.config.ssm_state_size,
                ssm_type=self.config.ssm_type,
                hardware_type=self.config.hardware_type,
                spiking_output=self.config.spiking_enabled,
                spike_threshold=self.config.spike_threshold,
                spike_decay=self.config.spike_decay
            )
        
        # Initialize attention configuration
        if self.config.attention_config is None:
            self.config.attention_config = LinearAttentionConfig(
                hidden_size=self.config.hidden_size,
                num_attention_heads=self.config.num_attention_heads,
                attention_type=self.config.attention_type,
                spiking=self.config.spiking_enabled,
                spike_threshold=self.config.spike_threshold,
                hardware_type=self.config.hardware_type
            )
        
        # Initialize hybrid configuration
        if self.config.hybrid_config is None:
            self.config.hybrid_config = HybridProcessorConfig(
                hidden_size=self.config.hidden_size,
                sequence_length=self.config.sequence_length,
                processing_mode=self.config.processing_mode,
                ssm_config=self.config.ssm_config,
                attention_config=self.config.attention_config,
                spiking_enabled=self.config.spiking_enabled,
                memory_integration=self.config.memory_integration,
                hardware_type=self.config.hardware_type
            )
        
        # Create hybrid processor
        self.hybrid_processor = HybridProcessor(self.config.hybrid_config)
        
        # Output projection to embedding dimension
        self.output_projection = nn.Linear(self.config.hidden_size, self.config.embedding_dim)
        
        # Normalization
        self.final_norm = nn.LayerNorm(self.config.embedding_dim)
        
        # Move to device
        self.to(self.device)
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """
        Simple tokenization for SSM processing
        Note: This is a simplified tokenization. In practice, you'd use
        a proper tokenizer like BPE, WordPiece, or SentencePiece
        """
        # Simple whitespace tokenization (for demonstration)
        tokens = text.lower().split()
        
        # Convert to token IDs (simplified hash-based mapping)
        token_ids = []
        for token in tokens:
            # Simple hash-based token ID (for demo purposes)
            token_id = hash(token) % self.config.vocab_size
            token_ids.append(token_id)
        
        # Pad or truncate to max_tokens
        if len(token_ids) > self.config.max_tokens:
            token_ids = token_ids[:self.config.max_tokens]
        else:
            token_ids.extend([0] * (self.config.max_tokens - len(token_ids)))  # Pad with 0
        
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)
    
    def _update_performance_stats(self, start_time: float, seq_len: int, operation: str):
        """Update performance statistics"""
        if not self.config.enable_profiling:
            return
        
        elapsed = time.time() - start_time
        
        if operation == 'encode':
            self.performance_stats['encoding_time'].append(elapsed)
        elif operation == 'decode':
            self.performance_stats['decoding_time'].append(elapsed)
        
        self.performance_stats['sequence_lengths'].append(seq_len)
        
        # Memory usage (GPU only)
        if self.device.type == 'cuda':
            memory_mb = torch.cuda.memory_allocated(self.device) / (1024**2)
            self.performance_stats['memory_usage'].append(memory_mb)
    
    async def encode_text(self, text: str, **kwargs) -> SSMTextEncoding:
        """
        Encode text using SSM-based processing
        
        Args:
            text: Input text to encode
            
        Returns:
            SSMTextEncoding with embeddings and metadata
        """
        start_time = time.time()
        
        try:
            # Tokenize input
            token_ids = self._tokenize_text(text)
            seq_len = len([t for t in token_ids if t != 0])  # Non-padded length
            
            # Embed tokens
            with torch.no_grad():
                if self.config.use_mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        embeddings = self._encode_sequence(token_ids.unsqueeze(0))
                else:
                    embeddings = self._encode_sequence(token_ids.unsqueeze(0))
            
            # Convert to numpy for storage
            embeddings_np = embeddings.cpu().numpy()
            
            # Create encoding object
            encoding = SSMTextEncoding(
                embeddings=embeddings_np,
                metadata={
                    'text': text,
                    'sequence_length': seq_len,
                    'device': str(self.device),
                    'encoding_type': self.config.encoding_type.value,
                    'timestamp': time.time()
                }
            )
            
            # Add memory context if available
            if self.memory and self.config.memory_integration:
                memory_context = await self.memory.get_memory_context()
                if memory_context is not None:
                    encoding.memory_context = memory_context.cpu().numpy()
            
            # Update performance stats
            self._update_performance_stats(start_time, seq_len, 'encode')
            
            self.logger.debug(f"Successfully encoded text of length {len(text)}")
            return encoding
            
        except Exception as e:
            self.logger.error(f"Text encoding failed: {e}")
            raise RuntimeError(f"Failed to encode text: {e}")
    
    def _encode_sequence(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode token sequence using hybrid processor"""
        batch_size, seq_len = token_ids.shape
        
        # Get token embeddings
        token_embeds = self.token_embedding(token_ids)
        
        # Project to hidden size
        hidden_states = self.input_projection(token_embeds)
        
        # Process through hybrid processor
        processed_states = self.hybrid_processor(hidden_states)
        
        # Global pooling to get sentence-level embedding
        # Use attention-weighted pooling instead of simple mean
        pooled_embedding = self._attention_pooling(processed_states)
        
        # Project to final embedding dimension
        final_embedding = self.output_projection(pooled_embedding)
        final_embedding = self.final_norm(final_embedding)
        
        return final_embedding
    
    def _attention_pooling(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Attention-based pooling for better sentence representations
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute attention weights
        attention_weights = torch.matmul(
            hidden_states, hidden_states.transpose(-2, -1)
        )
        attention_weights = F.softmax(attention_weights / math.sqrt(hidden_size), dim=-1)
        
        # Weighted sum
        pooled = torch.matmul(attention_weights, hidden_states)
        
        # Use the pooled representation for the entire sequence
        return pooled.mean(dim=1)
    
    async def decode_text(self, encoding: SSMTextEncoding) -> str:
        """
        Decode text representation (simplified implementation)
        Note: True text reconstruction from SSM encodings is non-trivial
        and would require a trained decoder/generative model
        """
        start_time = time.time()
        
        try:
            # For now, return a placeholder representation
            # In a full implementation, this would use a generative model
            embedding_info = {
                'embedding_shape': encoding.embeddings.shape,
                'has_ssm_states': encoding.ssm_states is not None,
                'has_attention_weights': encoding.attention_weights is not None,
                'has_spiking_patterns': encoding.spiking_patterns is not None,
                'encoding_type': encoding.metadata.get('encoding_type', 'unknown') if encoding.metadata else 'unknown'
            }
            
            decoded_info = f"[SSM DECODED] Embedding shape: {embedding_info['embedding_shape']}, " \
                          f"Type: {embedding_info['encoding_type']}, " \
                          f"Features: SSM={embedding_info['has_ssm_states']}, " \
                          f"Attention={embedding_info['has_attention_weights']}, " \
                          f"Spiking={embedding_info['has_spiking_patterns']}"
            
            # Update performance stats
            self._update_performance_stats(start_time, encoding.embeddings.shape[1], 'decode')
            
            return decoded_info
            
        except Exception as e:
            self.logger.error(f"Text decoding failed: {e}")
            raise RuntimeError(f"Failed to decode text: {e}")
    
    async def batch_encode(self, texts: List[str], **kwargs) -> List[SSMTextEncoding]:
        """Batch encode multiple texts"""
        try:
            if not texts:
                return []
            
            self.logger.info(f"Batch encoding {len(texts)} texts")
            
            encodings = []
            for text in texts:
                encoding = await self.encode_text(text, **kwargs)
                encodings.append(encoding)
            
            return encodings
            
        except Exception as e:
            self.logger.error(f"Batch encoding failed: {e}")
            raise RuntimeError(f"Failed to batch encode texts: {e}")
    
    async def encode_and_store(self, text: str, memory_key: str, **kwargs) -> SSMTextEncoding:
        """Encode text and store in memory"""
        try:
            encoding = await self.encode_text(text, **kwargs)
            
            if self.memory:
                success = await self.memory.store_encoding(memory_key, encoding)
                if not success:
                    self.logger.warning(f"Failed to store encoding with key: {memory_key}")
                
                # Update memory context
                if self.config.memory_integration:
                    await self.memory.update_memory_context(torch.from_numpy(encoding.embeddings))
            
            self.logger.info(f"Encoded and stored text with key: {memory_key}")
            return encoding
            
        except Exception as e:
            self.logger.error(f"Encode and store failed: {e}")
            raise
    
    async def retrieve_and_decode(self, memory_key: str) -> Optional[str]:
        """Retrieve encoding from memory and decode"""
        try:
            if not self.memory:
                self.logger.warning("No memory interface configured")
                return None
            
            encoding = await self.memory.retrieve_encoding(memory_key)
            if encoding is None:
                self.logger.info(f"No encoding found for key: {memory_key}")
                return None
            
            decoded_text = await self.decode_text(encoding)
            return decoded_text
            
        except Exception as e:
            self.logger.error(f"Retrieve and decode failed: {e}")
            raise
    
    async def encode_and_find_similar(self, text: str, threshold: float = 0.8, **kwargs) -> List[Tuple[str, float]]:
        """Encode text and find similar encodings in memory"""
        try:
            encoding = await self.encode_text(text, **kwargs)
            
            if not self.memory:
                self.logger.warning("No memory interface configured")
                return []
            
            similar_items = await self.memory.find_similar(encoding, threshold)
            return similar_items
            
        except Exception as e:
            self.logger.error(f"Find similar failed: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings"""
        return self.config.embedding_dim
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get performance statistics summary"""
        summary = {}
        
        for key, values in self.performance_stats.items():
            if values:
                summary[f'{key}_avg'] = np.mean(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
        
        # Add hybrid processor stats
        if hasattr(self, 'hybrid_processor'):
            hybrid_summary = self.hybrid_processor.get_performance_summary()
            summary.update({f'hybrid_{k}': v for k, v in hybrid_summary.items()})
        
        return summary
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'encoding_time': [],
            'decoding_time': [],
            'memory_usage': [],
            'sequence_lengths': []
        }
        
        # Reset hybrid processor stats
        if hasattr(self, 'hybrid_processor'):
            self.hybrid_processor.adaptive_reset()
    
    def get_memory_integration_status(self) -> Dict[str, Any]:
        """Get memory integration status"""
        if not self.config.memory_integration or not self.memory:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'memory_keys': self.config.memory_keys or [],
            'hybrid_memory_status': self.hybrid_processor.get_memory_integration_status()
        }


# Factory function
def create_ssm_text_processor(config: SSMTextConfig = None, 
                             memory_interface: Optional[MemoryIntegrationInterface] = None) -> SSMTextProcessor:
    """Create and initialize SSM text processor"""
    if config is None:
        config = SSMTextConfig()
    
    return SSMTextProcessor(config, memory_interface)


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    class SimpleMemoryInterface(MemoryIntegrationInterface):
        """Simple in-memory storage for testing"""
        
        def __init__(self):
            self.encodings = {}
            self.memory_context = None
        
        async def store_encoding(self, key: str, encoding: SSMTextEncoding) -> bool:
            self.encodings[key] = encoding
            return True
        
        async def retrieve_encoding(self, key: str) -> Optional[SSMTextEncoding]:
            return self.encodings.get(key)
        
        async def find_similar(self, encoding: SSMTextEncoding, threshold: float = 0.8) -> List[Tuple[str, float]]:
            # Simplified similarity calculation
            similar = []
            for key, stored_encoding in self.encodings.items():
                # Simple cosine similarity
                similarity = self._cosine_similarity(
                    encoding.embeddings.flatten(), 
                    stored_encoding.embeddings.flatten()
                )
                if similarity > threshold:
                    similar.append((key, similarity))
            return similar
        
        async def update_memory_context(self, context: torch.Tensor):
            self.memory_context = context
        
        async def get_memory_context(self) -> torch.Tensor:
            return self.memory_context
        
        def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    
    async def test_ssm_text_processor():
        """Test SSM text processor functionality"""
        logging.basicConfig(level=logging.INFO)
        
        # Create configuration
        config = SSMTextConfig(
            hidden_size=256,
            embedding_dim=128,
            sequence_length=128,
            processing_mode=HybridProcessingMode.ADAPTIVE,
            encoding_type=TextEncodingType.HYBRID,
            hardware_type=SSMHardwareType.CUDA if torch.cuda.is_available() else SSMHardwareType.CPU,
            spiking_enabled=True,
            memory_integration=True
        )
        
        # Create memory interface
        memory = SimpleMemoryInterface()
        
        # Create processor
        processor = create_ssm_text_processor(config, memory)
        
        # Test single text encoding
        text = "This is a test sentence for SSM-based encoding."
        print(f"Input text: {text}")
        
        encoding = await processor.encode_text(text)
        print(f"Embedding shape: {encoding.embeddings.shape}")
        print(f"Embedding dimension: {processor.get_embedding_dim()}")
        
        # Test encoding and storage
        stored_encoding = await processor.encode_and_store(text, "test_key_001")
        print(f"Stored encoding with key: test_key_001")
        
        # Test retrieval and decoding
        decoded = await processor.retrieve_and_decode("test_key_001")
        print(f"Decoded: {decoded}")
        
        # Test batch encoding
        texts = [
            "First SSM test sentence.",
            "Second SSM test sentence.",
            "Third SSM test sentence."
        ]
        
        batch_encodings = await processor.batch_encode(texts)
        print(f"Batch encoded {len(batch_encodings)} texts")
        
        # Performance summary
        perf_summary = processor.get_performance_summary()
        print(f"Performance summary: {perf_summary}")
    
    # Run test
    asyncio.run(test_ssm_text_processor())