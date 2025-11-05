"""
Core Interfaces for mini-biai-1 System

This module provides fundamental interfaces and data structures that define
the communication contracts between all components in the brain-inspired
modular AI system. The interface system ensures proper separation of concerns,
type safety, and compatibility across all modules.

The interface hierarchy includes:

Communication Layer:
    - Request: Standard request structure for system interactions
    - Reply: Response structure for system feedback
    - RouteDecision: Routing decisions from spiking neural networks

Memory System Interfaces:
    - STMState: Short-term memory state representation
    - Retrieval: Long-term memory retrieval results
    - EmbeddingVector: Vector representation for similarity operations

Message Flow:
    Request → RouteDecision → Retrieval → Reply

Each interface is designed to be:
- Type-safe with comprehensive validation
- Serializable for distributed processing
- Versioned for backward compatibility
- Performance-optimized for real-time operations

Key Features:
    - Comprehensive type hints for IDE support
    - Automatic validation and error handling
    - Support for streaming and batch operations
    - Hardware-aware optimizations
    - Thread-safe concurrent access patterns

Usage Examples:
    Basic Request Processing:
    >>> from src.interfaces import Request, Reply, RouteDecision
    >>> 
    >>> # Create a request
    >>> request = Request(
    ...     query="What is neural network?",
    ...     context={"domain": "education"}
    ... )
    >>> 
    >>> # Process through pipeline
    >>> route = RouteDecision(expert="language", confidence=0.9)
    >>> response = Reply(result="Neural networks are computational models...", confidence=0.95)
    
    Memory Integration:
    >>> from src.interfaces import STMState, Retrieval, EmbeddingVector
    >>> 
    >>> # Update short-term memory
    >>> stm_state = STMState(current_items=100, max_items=1000)
    >>> 
    >>> # Retrieve from long-term memory
    >>> retrieval = Retrieval(
    ...     embeddings=[EmbeddingVector(values=[...])],
    ...     similarities=[0.95],
    ...     metadata=[{"source": "corpus"}]
    ... )

Architecture Benefits:
    - Decouples implementation from interface contracts
    - Enables plugin-based architecture
    - Facilitates testing with mock implementations
    - Supports distributed system deployment
    - Enables performance monitoring and optimization

Dependencies:
    - torch: Tensor operations and neural network components
    - numpy: Numerical computations and array operations
    - typing: Type hints for modern Python versions
    - dataclasses: Automatic method generation for data structures

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .messages import (
    Request,
    STMState,
    Retrieval,
    RouteDecision,
    Reply,
    EmbeddingVector
)

__all__ = [
    'Request',
    'STMState', 
    'Retrieval',
    'RouteDecision',
    'Reply',
    'EmbeddingVector'
]

__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"