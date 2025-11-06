# Linear Text Processing Module

A flexible text encoding/decoding module built with transformer-based embeddings, designed for future SSM/linear-attention upgrades with integrated memory functionality.

## Features

- **Text Encoding/Decoding**: Transformer-based text embeddings with batch processing
- **Hardware Compatibility**: Automatic GPU/CPU detection with fallbacks (CUDA, MPS for Apple Silicon)
- **Replaceable Interface**: Abstract base class allows easy swapping for SSM/linear-attention models
- **Memory Integration**: Function-calling interface for storing and retrieving embeddings
- **Error Handling**: Comprehensive error handling with graceful degradation
- **Batch Processing**: Efficient batch encoding for multiple texts
- **Similarity Search**: Built-in cosine similarity calculation for embeddings

## Installation

```bash
# Install required dependencies
pip install torch transformers numpy

# Optional: For CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Optional: For Apple Silicon (MPS) support
pip install torch torchvision torchaudio
```

## Quick Start

```python
import asyncio
from src.language import create_text_processor, ProcessingConfig, HardwareType

async def main():
    # Create configuration
    config = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        hardware_type=HardwareType.AUTO,
        batch_size=8
    )
    
    # Create processor
    processor = await create_text_processor(config)
    
    # Encode text
    text = "This is a sample sentence for encoding."
    encoding = await processor.encode_text(text)
    
    print(f"Embedding shape: {encoding.embeddings.shape}")
    print(f"Embedding dimension: {processor.get_embedding_dim()}")

# Run async function
asyncio.run(main())
```

## Architecture

### Core Components

1. **TextProcessorInterface**: Abstract base class defining the text processing contract
2. **LinearTextProcessor**: Main implementation using transformers library
3. **SSMTextProcessor**: Placeholder for future SSM/linear-attention models
4. **MemoryIntegrationInterface**: Abstract interface for memory storage/retrieval
5. **ProcessingConfig**: Configuration dataclass for processor settings

### Data Structures

- **TextEncoding**: Container for embeddings and metadata
- **ProcessingConfig**: Configuration options (model, hardware, batch size, etc.)
- **HardwareType**: Enum for supported hardware (CPU, CUDA, MPS)

## API Reference

### TextProcessorInterface

Abstract methods that all text processors must implement:

```python
async def encode_text(text: str, **kwargs) -> TextEncoding
async def decode_text(encoding: TextEncoding) -> str  
async def batch_encode(texts: List[str], **kwargs) -> List[TextEncoding]
def get_embedding_dim() -> int
```

### MemoryIntegrationInterface

Interface for memory storage integration:

```python
async def store_encoding(key: str, encoding: TextEncoding) -> bool
async def retrieve_encoding(key: str) -> Optional[TextEncoding]
async def find_similar(encoding: TextEncoding, threshold: float) -> List[Tuple[str, float]]
```

### Configuration Options

```python
ProcessingConfig(
    model_name="bert-base-uncased",      # Transformer model
    max_length=512,                      # Maximum sequence length
    batch_size=32,                       # Batch processing size
    hardware_type=HardwareType.AUTO,     # Hardware selection
    return_tensors="pt",                 # Output tensor format
    padding=True,                        # Enable padding
    truncation=True,                     # Enable truncation
    return_attention_mask=True           # Return attention masks
)
```

## Hardware Support

The module automatically detects and uses the best available hardware:

- **CUDA**: NVIDIA GPUs (with automatic fp16 optimization)
- **MPS**: Apple Silicon GPUs (Metal Performance Shaders)
- **CPU**: Fallback for systems without GPU support

Hardware selection can be configured:
- `HardwareType.AUTO`: Automatic detection (recommended)
- `HardwareType.CUDA`: Force CUDA usage
- `HardwareType.MPS`: Force Apple Silicon GPU
- `HardwareType.CPU`: Force CPU processing

## Memory Integration Example

```python
from src.language import SimpleMemoryInterface  # Your implementation

# Create memory interface
memory = SimpleMemoryInterface()

# Create processor with memory
processor = create_text_processor(config, memory_interface=memory)

# Encode and store
await processor.encode_and_store("sample text", "text_key")

# Retrieve and decode
text = await processor.retrieve_and_decode("text_key")

# Find similar texts
similar = await processor.encode_and_find_similar("new text", threshold=0.8)
```

## Future SSM/Linear-Attention Upgrade

The module is designed for easy replacement with SSM/linear-attention models:

```python
# Currently using transformers
processor = create_text_processor(config, use_ssm=False)

# Future SSM implementation
processor = create_text_processor(config, use_ssm=True)  # When available
```

Both implementations share the same interface, ensuring seamless upgrades.

## Error Handling

Comprehensive error handling includes:

- **Model Loading Errors**: Graceful fallback to CPU if GPU models fail
- **Input Validation**: Strict text input validation with clear error messages  
- **Hardware Compatibility**: Automatic fallback to supported hardware
- **Batch Processing Errors**: Individual failure handling within batches
- **Memory Integration Errors**: Non-blocking memory operation failures

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic text encoding/decoding
- Batch processing
- Memory integration
- Similarity calculations
- Error handling demonstrations

## Performance Considerations

- **Batch Processing**: Use batch_encode for multiple texts (much faster)
- **Hardware Selection**: Auto-detection usually optimal, but manual selection available
- **Memory Usage**: Large models may require significant GPU memory
- **Sequence Length**: Longer sequences increase computation linearly

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch_size or max_length
2. **Model Loading Fails**: Check internet connection for model download
3. **Apple Silicon Issues**: Ensure PyTorch with MPS support is installed
4. **Transformers Import Error**: Install with `pip install transformers`

### Hardware Compatibility

- **Windows**: CUDA support requires NVIDIA GPU + drivers
- **macOS**: MPS support requires Apple Silicon (M1/M2)
- **Linux**: Full CUDA support, good CPU fallback

## License

This module is designed for research and development purposes.

## Contributing

The module architecture supports easy extension:

1. Implement new processors by extending `TextProcessorInterface`
2. Add new memory backends by implementing `MemoryIntegrationInterface`
3. Extend configuration options via `ProcessingConfig`
4. Add new hardware support via `HardwareType` enum
