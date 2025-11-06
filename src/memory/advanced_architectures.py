#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Memory Architectures - Phase 4

This module implements three advanced memory architecture components:

1. **Efficient Attention** - Memory-efficient attention mechanisms for deep learning
2. **Hierarchical Memory** - Brain-inspired multi-level memory architecture
3. **Semantic Memory** - Knowledge representation with concept graphs

Author: MiniMax AI
Version: 1.0.0
Created: 2025-11-06

Attention Features:
- Chunked attention for long sequences
- Sparse attention patterns for efficiency
- Linear attention for reduced complexity
- Flash attention integration

Hierarchical Memory Features:
- Multi-level memory architecture (sensory, working, episodic, semantic)
- Automatic memory consolidation between levels
- SQLite persistence for long-term storage
- Importance-based memory management

Semantic Memory Features:
- NetworkX-based concept graphs
- Relationship modeling (is_a, has_a, part_of, etc.)
- Contextual knowledge representation
- Path finding and reasoning capabilities
"""

from .efficient_attention import (
    MemoryEfficientAttention,
    AttentionStats,
    get_attention_engine
)

from .hierarchical_memory import (
    HierarchicalMemorySystem,
    MemoryLevel,
    MemoryEntry,
    hierarchical_memory_demo
)

from .semantic_memory import (
    SemanticMemorySystem,
    ConceptNode,
    Relationship,
    semantic_memory_demo
)

__all__ = [
    # Attention
    'MemoryEfficientAttention',
    'AttentionStats', 
    'get_attention_engine',
    
    # Hierarchical Memory
    'HierarchicalMemorySystem',
    'MemoryLevel',
    'MemoryEntry',
    'hierarchical_memory_demo',
    
    # Semantic Memory
    'SemanticMemorySystem',
    'ConceptNode',
    'Relationship',
    'semantic_memory_demo'
]

__version__ = "1.0.0"
__author__ = "MiniMax AI"
__description__ = "Advanced Memory Architectures - Phase 4"
