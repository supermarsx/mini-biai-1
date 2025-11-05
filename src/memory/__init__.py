"""
Memory Systems Package for mini-biai-1

This package provides comprehensive memory management systems for AI systems,
implementing both short-term and long-term memory components with biological
inspiration and modern vector database technology.

Architecture Overview:
    The memory system follows a dual-storage architecture:
    
    Short-Term Memory (STM):
        - Ring buffer for temporal data with configurable capacity
        - Key-value scratchpad for quick access to frequently used data
        - TTL (Time-To-Live) support for automatic data expiration
        - Thread-safe operations with comprehensive locking
        - Real-time data cleanup and maintenance
    
    Long-Term Memory (LTM):
        - FAISS vector database for semantic similarity search
        - IVF-PQ indexing for efficient large-scale retrieval
        - Sentence transformer embeddings for text representation
        - Persistent storage with automatic save/load functionality
        - Adaptive indexing with automatic training data collection

Key Features:
    - Biologically-inspired memory hierarchies
    - High-performance vector similarity search
    - Thread-safe concurrent access
    - Automatic memory management and cleanup
    - Persistent storage capabilities
    - Configurable memory parameters
    - Comprehensive error handling and fallbacks
    - Performance monitoring and statistics

Integration:
    The memory systems are designed to integrate seamlessly with:
    - Inference pipelines for context management
    - Spiking neural networks for adaptive routing
    - Language models for semantic understanding
    - Data gathering systems for knowledge storage

Usage Examples:
    Basic Memory Operations:
    >>> from src.memory import ShortTermMemory, FAISSLongTermMemory
    >>> 
    >>> # Initialize memory systems
    >>> stm = ShortTermMemory(ring_buffer_capacity=1000, kv_max_size=10000)
    >>> ltm = FAISSLongTermMemory(dimension=384, max_size=100000)
    >>> 
    >>> # Add temporal memory
    >>> stm.add_temporal_memory("User asked about neural networks", {"category": "conversation"})
    >>> 
    >>> # Store long-term knowledge
    >>> memory_id = ltm.add_memory(
    ...     "Neural networks are computational models inspired by biological brains",
    ...     {"domain": "ai", "confidence": 0.9}
    ... )
    >>> 
    >>> # Retrieve relevant information
    >>> results = ltm.search("information about neural networks", k=5)
    >>> 
    >>> # Access recent conversations
    >>> recent = stm.get_recent_temporal(10)
    >>> 
    >>> # Key-value operations
    >>> stm.set_kv_memory("user_preference", "dark_mode", ttl=3600)
    >>> preference = stm.get_kv_memory("user_preference")

Performance Characteristics:
    Short-Term Memory:
        - O(1) average insertion and retrieval for ring buffer
        - O(1) average lookup for key-value operations
        - Memory usage: O(capacity) for ring buffer, O(max_size) for KV store
        - Cleanup overhead: O(n) for expired items, runs in background thread
    
    Long-Term Memory:
        - O(log n) average search time with FAISS IVF-PQ
        - O(n) insertion time for new embeddings
        - Memory usage: O(n × dimension) for embeddings
        - Index training: O(n × dimension²) for first 100+ items

Configuration:
    The memory systems support extensive configuration:
    
    STM Configuration:
        - ring_buffer_capacity: Size of temporal memory ring buffer (default: 1000)
        - kv_max_size: Maximum key-value store size (default: 10000)
        - cleanup_interval: Background cleanup frequency in seconds (default: 3600)
    
    LTM Configuration:
        - dimension: Vector embedding dimension (default: 384)
        - nlist: Number of IVF clusters (default: 100)
        - m: PQ subvector count (default: 16)
        - nbits: PQ bits per subvector (default: 8)
        - max_size: Maximum memory capacity (default: 1000000)
        - persist_path: Optional path for persistent storage

Dependencies:
    - torch: Tensor operations and neural network components
    - numpy: Numerical computations and array operations
    - faiss: Facebook AI Similarity Search for vector databases
    - sentence-transformers: Text embedding generation
    - threading: Concurrent access and thread safety
    - time: Timestamp management and TTL calculations

Error Handling:
    Both memory systems implement comprehensive error handling:
    - Graceful degradation on missing dependencies
    - Automatic fallback mechanisms for vector operations
    - Thread-safe error recovery
    - Comprehensive logging for debugging
    - Circuit breaker patterns for fault tolerance

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .stm import ShortTermMemory, MemoryItem
from .ltm_faiss import FAISSLongTermMemory, MemoryVector

__all__ = ['ShortTermMemory', 'MemoryItem', 'FAISSLongTermMemory', 'MemoryVector']

__version__ = "1.0.0"