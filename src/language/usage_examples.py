#!/usr/bin/env python3
"""
Example usage of the Linear Text Processing Module

Demonstrates text encoding, decoding, batch processing, and memory integration.
"""

import asyncio
import logging
from typing import List, Optional

# Import from our module
from linear_text import (
    ProcessingConfig,
    HardwareType,
    create_text_processor,
    TextEncoding,
    MemoryIntegrationInterface,
    validate_text_input,
    calculate_similarity
)


class SimpleMemoryInterface(MemoryIntegrationInterface):
    """Simple in-memory implementation for demonstration"""
    
    def __init__(self):
        self.storage = {}
        self.logger = logging.getLogger(__name__)
    
    async def store_encoding(self, key: str, encoding: TextEncoding) -> bool:
        """Store text encoding in memory"""
        try:
            self.storage[key] = encoding
            self.logger.info(f"Stored encoding with key: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to store encoding: {e}")
            return False
    
    async def retrieve_encoding(self, key: str) -> Optional[TextEncoding]:
        """Retrieve text encoding from memory"""
        encoding = self.storage.get(key)
        if encoding:
            self.logger.info(f"Retrieved encoding with key: {key}")
        else:
            self.logger.info(f"No encoding found for key: {key}")
        return encoding
    
    async def find_similar(self, encoding: TextEncoding, threshold: float = 0.8) -> List[tuple]:
        """Find similar encodings in memory"""
        similar = []
        
        for key, stored_encoding in self.storage.items():
            similarity = calculate_similarity(encoding, stored_encoding)
            if similarity >= threshold:
                similar.append((key, similarity))
        
        # Sort by similarity (highest first)
        similar.sort(key=lambda x: x[1], reverse=True)
        self.logger.info(f"Found {len(similar)} similar encodings")
        return similar


async def demonstrate_basic_encoding():
    """Demonstrate basic text encoding functionality"""
    print("\n=== Basic Text Encoding Demo ===")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Good for semantic similarity
        hardware_type=HardwareType.AUTO,
        batch_size=8,
        max_length=256
    )
    
    try:
        # Create processor
        processor = create_text_processor(config)
        print(f"✓ Created processor on device: {processor._device}")
        print(f"✓ Embedding dimension: {processor.get_embedding_dim()}")
        
        # Test single text encoding
        text = "The quick brown fox jumps over the lazy dog."
        print(f"\nOriginal text: '{text}'")
        
        encoding = await processor.encode_text(text)
        print(f"✓ Encoded successfully")
        print(f"  - Embedding shape: {encoding.embeddings.shape}")
        print(f"  - Has attention mask: {encoding.attention_mask is not None}")
        print(f"  - Metadata: {encoding.metadata}")
        
        # Test decoding (stub)
        decoded = await processor.decode_text(encoding)
        print(f"✓ Decoded (stub): '{decoded}'")
        
    except Exception as e:
        print(f"✗ Error in basic encoding: {e}")


async def demonstrate_batch_processing():
    """Demonstrate batch text processing"""
    print("\n=== Batch Processing Demo ===")
    
    config = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        hardware_type=HardwareType.AUTO,
        batch_size=4
    )
    
    try:
        processor = create_text_processor(config)
        
        # Test batch encoding
        texts = [
            "Machine learning is fascinating.",
            "Natural language processing enables AI understanding.",
            "Neural networks process information hierarchically.",
            "Transformers revolutionized sequence modeling."
        ]
        
        print(f"Processing {len(texts)} texts in batch:")
        for i, text in enumerate(texts):
            print(f"  {i+1}. {text}")
        
        encodings = await processor.batch_encode(texts)
        
        print(f"✓ Batch processed {len(encodings)} texts")
        for i, encoding in enumerate(encodings):
            print(f"  - Text {i+1}: shape {encoding.embeddings.shape}")
            
    except Exception as e:
        print(f"✗ Error in batch processing: {e}")


async def demonstrate_memory_integration():
    """Demonstrate memory integration functionality"""
    print("\n=== Memory Integration Demo ===")
    
    config = ProcessingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        hardware_type=HardwareType.AUTO
    )
    
    # Create memory interface
    memory = SimpleMemoryInterface()
    
    try:
        # Create processor with memory
        processor = create_text_processor(config, memory_interface=memory)
        
        # Test texts with similar meanings
        texts_and_keys = [
            ("artificial intelligence is transforming technology", "ai_tech"),
            ("machine learning algorithms learn from data", "ml_algorithms"),
            ("deep learning uses neural networks", "deep_learning"),
            ("AI is revolutionizing the tech industry", "ai_tech_similar")  # Similar to first
        ]
        
        print("Encoding and storing texts:")
        for text, key in texts_and_keys:
            encoding = await processor.encode_and_store(text, key)
            print(f"  ✓ Stored '{text}' with key '{key}'")
        
        # Test retrieval
        retrieved = await processor.retrieve_and_decode("ai_tech")
        print(f"\n✓ Retrieved and decoded: '{retrieved}'")
        
        # Test similarity search
        query_text = "artificial intelligence changes technology"
        print(f"\nFinding similar texts to: '{query_text}'")
        
        similar = await processor.encode_and_find_similar(query_text, threshold=0.5)
        
        print(f"Found {len(similar)} similar encodings:")
        for key, similarity in similar:
            print(f"  - Key '{key}': similarity {similarity:.3f}")
            
    except Exception as e:
        print(f"✗ Error in memory integration: {e}")


async def demonstrate_utilities():
    """Demonstrate utility functions"""
    print("\n=== Utility Functions Demo ===")
    
    # Test input validation
    test_inputs = [
        "Valid text",
        "",
        None,
        123,
        ["Valid", "text", "list"],
        [],
        ["", "valid"]
    ]
    
    print("Testing input validation:")
    for i, inp in enumerate(test_inputs):
        try:
            result = validate_text_input(inp)
            print(f"  {i+1}. {type(inp).__name__}: {result}")
        except Exception as e:
            print(f"  {i+1}. {type(inp).__name__}: Error - {e}")
    
    # Test similarity calculation with dummy data
    print("\nTesting similarity calculation:")
    try:
        # Create dummy encodings
        import numpy as np
        
        emb1 = TextEncoding(embeddings=np.array([[0.1, 0.2, 0.3, 0.4]]))
        emb2 = TextEncoding(embeddings=np.array([[0.1, 0.2, 0.3, 0.4]]))  # Identical
        emb3 = TextEncoding(embeddings=np.array([[-0.1, -0.2, -0.3, -0.4]]))  # Opposite
        
        sim_identical = calculate_similarity(emb1, emb2)
        sim_opposite = calculate_similarity(emb1, emb3)
        
        print(f"  - Identical embeddings similarity: {sim_identical:.3f}")
        print(f"  - Opposite embeddings similarity: {sim_opposite:.3f}")
        
    except Exception as e:
        print(f"✗ Error in similarity calculation: {e}")


async def demonstrate_error_handling():
    """Demonstrate error handling capabilities"""
    print("\n=== Error Handling Demo ===")
    
    # Test with invalid configuration
    try:
        config = ProcessingConfig(model_name="nonexistent-model")
        processor = create_text_processor(config)
        print("✗ Should have failed with nonexistent model")
    except Exception as e:
        print(f"✓ Correctly handled invalid model: {type(e).__name__}")
    
    # Test with empty text
    try:
        config = ProcessingConfig()
        processor = create_text_processor(config)
        await processor.encode_text("")
        print("✗ Should have failed with empty text")
    except Exception as e:
        print(f"✓ Correctly handled empty text: {type(e).__name__}")
    
    # Test with invalid hardware type
    try:
        config = ProcessingConfig(hardware_type="invalid")
        print("✗ Should have failed with invalid hardware type")
    except Exception as e:
        print(f"✓ Correctly handled invalid hardware: {type(e).__name__}")


async def main():
    """Main demonstration function"""
    print("Linear Text Processing Module Demo")
    print("=" * 50)
    
    # Run all demonstrations
    await demonstrate_basic_encoding()
    await demonstrate_batch_processing()
    await demonstrate_memory_integration()
    await demonstrate_utilities()
    await demonstrate_error_handling()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())