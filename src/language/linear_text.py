# linear_text.py content (632 lines)
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. Install with: pip install transformers")


class HardwareType(Enum):
    """Supported hardware types for text processing"""
    AUTO = "auto"  # Automatic hardware detection
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    CPU_FALLBACK = "cpu_fallback"


@dataclass
class ProcessingConfig:
    """
    Configuration for text processing pipeline.
    
    This dataclass encapsulates all configuration parameters for the text
    processing pipeline, providing a centralized configuration management
    system that can be easily customized for different use cases.
    
    The configuration covers:
    - Model selection and parameters
    - Text processing constraints (length limits, batch sizes)
    - Hardware optimization settings
    - Output format preferences
    - Tokenization behavior control
    
    Key Configuration Areas:
        Model Configuration:
            - model_name: Transformer model identifier
            - hardware_type: Target compute device
            - return_tensors: Output tensor format preference
            
        Processing Constraints:
            - max_length: Maximum text length for processing
            - batch_size: Number of texts processed together
            - padding: Whether to pad sequences to same length
            - truncation: Whether to truncate long sequences
            
        Output Control:
            - return_attention_mask: Include attention masks in output
            - return_token_type_ids: Include token type IDs
            
    Example Usage:
        Basic Configuration:
        >>> config = ProcessingConfig()
        >>> print(f"Model: {config.model_name}")
        >>> print(f"Max length: {config.max_length}")
        
        GPU-Optimized Configuration:
        >>> gpu_config = ProcessingConfig(
        ...     model_name="all-mpnet-base-v2",  # Higher quality model
        ...     hardware_type=HardwareType.CUDA,
        ...     batch_size=64,                   # Large batch for GPU
        ...     max_length=512,                  # Longer sequences
        ...     return_tensors="pt"              # PyTorch tensors
        ... )
        
        Resource-Constrained Configuration:
        >>> lightweight_config = ProcessingConfig(
        ...     model_name="all-MiniLM-L6-v2",  # Small, fast model
        ...     hardware_type=HardwareType.CPU,
        ...     batch_size=8,                    # Small batches
        ...     max_length=256,                  # Shorter sequences
        ...     padding=False                    # Save memory
        ... )
        
        Apple Silicon Configuration:
        >>> mps_config = ProcessingConfig(
        ...     hardware_type=HardwareType.MPS,
        ...     batch_size=32,                   # Medium batches
        ...     return_tensors="pt"              # PyTorch format
        ... )
    
    Performance Impact:
        Model Selection:
            - Larger models: Higher quality, slower inference
            - Smaller models: Faster inference, lower quality
            
        Batch Size:
            - Larger batches: Better GPU utilization, higher memory usage
            - Smaller batches: Lower memory usage, potentially slower
            
        Hardware Type:
            - CPU: Universal compatibility, baseline performance
            - CUDA: GPU acceleration, 3-10x speedup
            - MPS: Apple Silicon optimization, 2-5x speedup
            
    Note:
        - All parameters have sensible defaults for general use
        - Hardware auto-detection overrides manual settings when AUTO is selected
        - Memory usage scales with batch_size × max_length
        - Text length limits prevent memory issues with very long inputs
    """
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 32
    hardware_type: HardwareType = HardwareType.AUTO
    return_tensors: str = "pt"
    padding: bool = True
    truncation: bool = True
    return_attention_mask: bool = True


@dataclass
class TextEncoding:
    """Container for text encoding results"""
    embeddings: np.ndarray
    attention_mask: Optional[np.ndarray] = None
    token_ids: Optional[List[int]] = None
    metadata: Optional[Dict[str, Any]] = None


class TextProcessorInterface(ABC):
    """Abstract interface for text processing implementations"""
    
    @abstractmethod
    async def encode_text(self, text: str, **kwargs) -> TextEncoding:
        """Encode text into embeddings"""
        pass
    
    @abstractmethod
    async def decode_text(self, encoding: TextEncoding) -> str:
        """Decode embeddings back to text"""
        pass
    
    @abstractmethod
    async def batch_encode(self, texts: List[str], **kwargs) -> List[TextEncoding]:
        """Batch encode multiple texts"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings"""
        pass


class LinearTextProcessor(TextProcessorInterface):
    """
    Linear text processor using transformer models.
    Designed for easy replacement with SSM/linear-attention models.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._model = None
        self._tokenizer = None
        self._device = self._setup_device()
        self._initialize_model()
    
    def _setup_device(self) -> str:
        """Setup appropriate device with hardware compatibility checking"""
        try:
            if self.config.hardware_type == HardwareType.AUTO:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            else:
                device_map = {
                    HardwareType.CUDA: "cuda",
                    HardwareType.MPS: "mps",
                    HardwareType.CPU: "cpu"
                }
                return device_map[self.config.hardware_type]
        except Exception as e:
            self.logger.warning(f"Device setup failed, falling back to CPU: {e}")
            return "cpu"
    
    def _initialize_model(self):
        """Initialize transformer model with error handling"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library is required but not installed")
        
        try:
            self.logger.info(f"Loading model {self.config.model_name} on {self._device}")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            # Load model with device mapping
            model_kwargs = {
                'device_map': self._device,
                'torch_dtype': torch.float16 if self._device == 'cuda' else torch.float32,
                'trust_remote_code': True
            }
            
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                **model_kwargs
            )
            
            self._model.eval()
            self.logger.info(f"Successfully loaded model on {self._device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def _encode_single_text(self, text: str) -> TextEncoding:
        """Encode a single text string"""
        try:
            # Tokenize input
            encoded = self._tokenizer(
                text,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=self.config.return_tensors,
                return_attention_mask=self.config.return_attention_mask,
                return_token_type_ids=False
            )
            
            # Get embeddings
            with torch.no_grad():
                if self._device != "cpu":
                    encoded = {k: v.to(self._device) for k, v in encoded.items()}
                
                outputs = self._model(**encoded)
                
                # Use [CLS] token embedding (first token) or mean pooling
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                    # Apply mean pooling with attention mask
                    attention_mask = encoded['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    pooled_embeddings = sum_embeddings / sum_mask
                else:
                    # Fallback for models without hidden states
                    pooled_embeddings = outputs.pooler_output
            
            # Convert to numpy
            embeddings_np = pooled_embeddings.cpu().numpy()
            
            return TextEncoding(
                embeddings=embeddings_np,
                attention_mask=encoded['attention_mask'].cpu().numpy() if self.config.return_attention_mask else None,
                token_ids=encoded['input_ids'].cpu().tolist() if hasattr(encoded['input_ids'], 'cpu') else [],
                metadata={
                    'model_name': self.config.model_name,
                    'device': self._device,
                    'text_length': len(text)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Text encoding failed: {e}")
            raise RuntimeError(f"Failed to encode text: {e}")
    
    async def encode_text(self, text: str, **kwargs) -> TextEncoding:
        """Encode text into embeddings (async interface)"""
        try:
            # Validate input
            if not isinstance(text, str):
                raise ValueError("Input must be a string")
            
            if not text.strip():
                raise ValueError("Input text cannot be empty")
            
            # Process text
            result = self._encode_single_text(text.strip())
            
            self.logger.debug(f"Successfully encoded text of length {len(text)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Text encoding error: {e}")
            raise
    
    async def decode_text(self, encoding: TextEncoding) -> str:
        """
        Decode embeddings back to text (stub implementation).
        Note: True text reconstruction from embeddings is not feasible.
        This is a placeholder for the interface.
        """
        try:
            # For now, this is a stub. True text reconstruction would require
            # a generative model and extensive training
            self.logger.warning("Text decoding is not implemented - this is a stub")
            
            # Placeholder: return a representation of the embedding
            embedding_info = f"Embedding shape: {encoding.embeddings.shape}"
            return f"[DECODED] {embedding_info}"
            
        except Exception as e:
            self.logger.error(f"Text decoding error: {e}")
            raise RuntimeError(f"Failed to decode text: {e}")
    
    async def batch_encode(self, texts: List[str], **kwargs) -> List[TextEncoding]:
        """Batch encode multiple texts"""
        try:
            if not texts:
                return []
            
            # Validate all inputs
            valid_texts = []
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    raise ValueError(f"Input {i} must be a string")
                if not text.strip():
                    raise ValueError(f"Input {i} cannot be empty")
                valid_texts.append(text.strip())
            
            # Batch process
            results = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                
                # Tokenize batch
                encoded_batch = self._tokenizer(
                    batch,
                    max_length=self.config.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors=self.config.return_tensors,
                    return_attention_mask=True,
                    return_token_type_ids=False
                )
                
                # Process batch
                with torch.no_grad():
                    if self._device != "cpu":
                        encoded_batch = {k: v.to(self._device) for k, v in encoded_batch.items()}
                    
                    outputs = self._model(**encoded_batch)
                    
                    # Extract embeddings with pooling
                    attention_mask = encoded_batch['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                    
                    # Convert to numpy and split by batch
                    batch_embeddings_np = batch_embeddings.cpu().numpy()
                    
                    for j in range(batch_embeddings_np.shape[0]):
                        results.append(TextEncoding(
                            embeddings=batch_embeddings_np[j:j+1],  # Keep batch dimension for consistency
                            attention_mask=attention_mask[j:j+1].cpu().numpy(),
                            token_ids=[],  # TODO: Handle batch token IDs if needed
                            metadata={
                                'model_name': self.config.model_name,
                                'device': self._device,
                                'batch_index': i + j,
                                'batch_size': len(batch)
                            }
                        ))
            
            self.logger.info(f"Successfully batch encoded {len(texts)} texts")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch encoding failed: {e}")
            raise RuntimeError(f"Failed to batch encode texts: {e}")
    
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of embeddings"""
        try:
            if self._model is None:
                raise RuntimeError("Model not initialized")
            
            # Get embedding dimension from model config
            if hasattr(self._model.config, 'hidden_size'):
                return self._model.config.hidden_size
            elif hasattr(self._model.config, 'd_model'):
                return self._model.config.d_model
            else:
                # Fallback: test with dummy input
                test_encoding = self._encode_single_text("test")
                return test_encoding.embeddings.shape[-1]
                
        except Exception as e:
            self.logger.error(f"Failed to get embedding dimension: {e}")
            return 768  # Common BERT dimension fallback


class SSMTextProcessor(TextProcessorInterface):
    """
    Placeholder for future SSM/linear-attention implementation.
    This class demonstrates the replaceable interface.
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("SSM Text Processor initialized (placeholder)")
    
    async def encode_text(self, text: str, **kwargs) -> TextEncoding:
        """SSM-based text encoding (placeholder)"""
        self.logger.info("SSM encoding not yet implemented")
        # TODO: Implement with SSM/linear-attention models
        raise NotImplementedError("SSM encoding not implemented yet")
    
    async def decode_text(self, encoding: TextEncoding) -> str:
        """SSM-based text decoding (placeholder)"""
        self.logger.info("SSM decoding not yet implemented")
        # TODO: Implement with SSM/linear-attention models
        raise NotImplementedError("SSM decoding not implemented yet")
    
    async def batch_encode(self, texts: List[str], **kwargs) -> List[TextEncoding]:
        """SSM-based batch encoding (placeholder)"""
        self.logger.info("SSM batch encoding not yet implemented")
        # TODO: Implement with SSM/linear-attention models
        raise NotImplementedError("SSM batch encoding not implemented yet")
    
    def get_embedding_dim(self) -> int:
        """Return SSM embedding dimension"""
        # TODO: Implement based on SSM model
        raise NotImplementedError("SSM embedding dimension not implemented yet")


class MemoryIntegrationInterface(ABC):
    """Interface for memory integration with text processing"""
    
    @abstractmethod
    async def store_encoding(self, key: str, encoding: TextEncoding) -> bool:
        """Store text encoding in memory"""
        pass
    
    @abstractmethod
    async def retrieve_encoding(self, key: str) -> Optional[TextEncoding]:
        """Retrieve text encoding from memory"""
        pass
    
    @abstractmethod
    async def find_similar(self, encoding: TextEncoding, threshold: float = 0.8) -> List[Tuple[str, float]]:
        """Find similar encodings in memory"""
        pass


class LinearTextProcessorWithMemory(LinearTextProcessor):
    """
    Text processor with integrated memory functionality.
    Provides function-calling interface for memory operations.
    """
    
    def __init__(self, config: ProcessingConfig, memory_interface: Optional[MemoryIntegrationInterface] = None):
        super().__init__(config)
        self.memory = memory_interface
        self.logger = logging.getLogger(__name__)
    
    async def encode_and_store(self, text: str, memory_key: str, **kwargs) -> TextEncoding:
        """Encode text and store in memory with a key"""
        try:
            encoding = await self.encode_text(text, **kwargs)
            
            if self.memory:
                success = await self.memory.store_encoding(memory_key, encoding)
                if not success:
                    self.logger.warning(f"Failed to store encoding with key: {memory_key}")
            
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


# Factory function for creating appropriate processor
def create_text_processor(config: ProcessingConfig = None, 
                         use_ssm: bool = False,
                         memory_interface: MemoryIntegrationInterface = None) -> TextProcessorInterface:
    """
    Factory function to create appropriate text processor
    
    Args:
        config: Processing configuration
        use_ssm: If True, use SSM processor (when available)
        memory_interface: Optional memory integration interface
    
    Returns:
        TextProcessorInterface: Configured text processor
    """
    if config is None:
        config = ProcessingConfig()
    
    if use_ssm:
        processor = SSMTextProcessor(config)
    else:
        processor = LinearTextProcessorWithMemory(config, memory_interface)
    
    return processor


# Utility functions
def validate_text_input(text: Union[str, List[str]]) -> bool:
    """Validate text input for processing"""
    if isinstance(text, str):
        return len(text.strip()) > 0
    elif isinstance(text, list):
        return all(isinstance(t, str) and len(t.strip()) > 0 for t in text)
    return False


def calculate_similarity(encoding1: TextEncoding, encoding2: TextEncoding) -> float:
    """Calculate cosine similarity between two text encodings"""
    try:
        emb1 = encoding1.embeddings.flatten()
        emb2 = encoding2.embeddings.flatten()
        
        # Ensure same length
        min_len = min(len(emb1), len(emb2))
        emb1 = emb1[:min_len]
        emb2 = emb2[:min_len]
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
        
    except Exception as e:
        logging.error(f"Similarity calculation failed: {e}")
        return 0.0


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Create processor configuration
        config = ProcessingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Better for embeddings
            hardware_type=HardwareType.AUTO,
            batch_size=16
        )
        
        try:
            # Create processor
            processor = create_text_processor(config)
            
            # Test single text encoding
            text = "This is a test sentence for encoding."
            encoding = await processor.encode_text(text)
            print(f"Embedding shape: {encoding.embeddings.shape}")
            print(f"Embedding dimension: {processor.get_embedding_dim()}")
            
            # Test batch encoding
            texts = [
                "First test sentence.",
                "Second test sentence.",
                "Third test sentence."
            ]
            batch_encodings = await processor.batch_encode(texts)
            print(f"Batch encoded {len(batch_encodings)} texts")
            
            # Test decoding (stub)
            decoded = await processor.decode_text(encoding)
            print(f"Decoded: {decoded}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Run async main
    asyncio.run(main())