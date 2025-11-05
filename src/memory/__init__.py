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

Performance Characteristics:
    Short-Term Memory:
        - O(1) average insertion and retrieval for ring buffer
        - O(1) average lookup for key-value operations
        - Memory usage: O(capacity) for ring buffer, O(max_size) for KV store
    
    Long-Term Memory:
        - O(log n) average search time with FAISS IVF-PQ
        - O(n) insertion time for new embeddings
        - Memory usage: O(n × dimension) for embeddings
        - Index training: O(n × dimension²) for first 100+ items

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .stm import ShortTermMemory, MemoryItem
from .ltm_faiss import FAISSLongTermMemory, MemoryVector

__all__ = ['ShortTermMemory', 'MemoryItem', 'FAISSLongTermMemory', 'MemoryVector']

__version__ = "1.0.0"