"""
Core message interfaces for the brain-inspired AI system.

This module defines the fundamental data structures used throughout the system
for communication between components and state management.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import numpy as np


# Shared embedding format specification
# All embeddings in the system use this format for consistency
EmbeddingVector = Union[np.ndarray, List[float], List[int]]


@dataclass
class Request:
    """
    Represents a request message in the system.
    
    Attributes:
        text: The textual content of the request
        meta: Additional metadata associated with the request
    """
    text: str
    meta: Dict[str, Any]


@dataclass
class STMState:
    """
    Represents the Short-Term Memory state.
    
    Attributes:
        tokens: List of tokens representing the current context
        kv: Key-value store for associative memory
    """
    tokens: List[str]
    kv: Dict[str, Any]


@dataclass
class Retrieval:
    """
    Represents a retrieval operation with query and results.
    
    Attributes:
        query_vec: Vector representation of the query
        results: List of retrieved items with similarity scores
    """
    query_vec: EmbeddingVector
    results: List[Dict[str, Any]]


@dataclass
class RouteDecision:
    """
    Represents a routing decision based on weights and spiking rates.
    
    Attributes:
        weights: Weight distribution for routing choices
        spike_rate: Current spiking activity rate
    """
    weights: Dict[str, float]
    spike_rate: float


@dataclass
class Reply:
    """
    Represents a reply message with content and trace information.
    
    Attributes:
        text: The textual content of the reply
        trace: Execution or processing trace information
    """
    text: str
    trace: Dict[str, Any]