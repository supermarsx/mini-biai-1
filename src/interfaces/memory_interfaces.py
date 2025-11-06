"""
Enhanced memory interfaces for the brain-inspired AI system.

This module defines interfaces for advanced memory systems including
hierarchical memory, semantic memory, and knowledge graph representations.

The interfaces support working memory, episodic memory, semantic memory,
knowledge graphs, and biological memory consolidation processes.

Key Components:
    - HierarchicalMemory: Multi-tier memory system
    - SemanticMemory: Knowledge graph-based memory
    - WorkingMemory: Active working memory buffer
    - EpisodicMemory: Event and experience memory
    - KnowledgeGraph: Semantic relationship graph
    - MemoryConsolidation: Memory consolidation process

Architecture Benefits:
    - Biological plausibility
    - Multi-tier memory hierarchy
    - Knowledge graph integration
    - Automatic consolidation
    - Semantic reasoning

Version: 1.0.0
Author: mini-biai-1 Team
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple, Set
from enum import Enum
from datetime import datetime
import numpy as np
import networkx as nx


class MemoryTier(Enum):
    """Memory hierarchy tiers."""
    WORKING = "working"
    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class ConsolidationType(Enum):
    """Memory consolidation types."""
    AUTO = "automatic"
    TRIGGERED = "triggered"
    SCHEDULED = "scheduled"
    EMERGENCY = "emergency"


class KnowledgeRelation(Enum):
    """Knowledge graph relation types."""
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    INFLUENCES = "influences"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    DERIVED_FROM = "derived_from"


class ReasoningType(Enum):
    """Reasoning operation types."""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    TEMPORAL = "temporal"


@dataclass
class MemoryItem:
    """
    Individual memory item structure.
    
    Attributes:
        item_id: Unique memory item identifier
        content: Memory content
        tier: Memory tier level
        strength: Memory strength (0.0-1.0)
        access_count: Number of times accessed
        last_access: Last access timestamp
        creation_time: Memory creation time
        decay_rate: Biological decay rate
        associations: Associated memory items
        importance: Importance score
        metadata: Additional memory metadata
    """
    item_id: str
    content: Any
    tier: MemoryTier
    strength: float
    access_count: int
    last_access: datetime
    creation_time: datetime
    decay_rate: float
    associations: List[str]
    importance: float
    metadata: Dict[str, Any]


@dataclass
class WorkingMemory:
    """
    Active working memory buffer.
    
    Attributes:
        capacity: Working memory capacity
        current_items: Currently held items
        attention_focus: Attention focus indicator
        processing_load: Cognitive processing load
        retention_time: Item retention time
        replacement_policy: Item replacement strategy
        attentional_control: Attention control mechanisms
        interference_patterns: Interference patterns
    """
    capacity: int
    current_items: List[MemoryItem]
    attention_focus: str
    processing_load: float
    retention_time: float
    replacement_policy: str
    attentional_control: Dict[str, Any]
    interference_patterns: List[str]


@dataclass
class EpisodicMemory:
    """
    Event and experience memory.
    
    Attributes:
        episodes: List of episodic memories
        temporal_organization: Temporal organization structure
        contextual_associations: Contextual associations
        emotional_valence: Emotional valence of episodes
        spatial_context: Spatial context information
        retrieval_cues: Memory retrieval cues
        reconstruction_fidelity: Reconstruction fidelity scores
    """
    episodes: List[Dict[str, Any]]
    temporal_organization: Dict[str, List[str]]
    contextual_associations: Dict[str, List[str]]
    emotional_valence: Dict[str, float]
    spatial_context: Dict[str, Dict[str, float]]
    retrieval_cues: List[str]
    reconstruction_fidelity: Dict[str, float]


@dataclass
class KnowledgeConcept:
    """
    Knowledge concept node.
    
    Attributes:
        concept_id: Unique concept identifier
        name: Concept name
        definition: Concept definition
        properties: Concept properties
        examples: Concept examples
        confidence: Concept confidence score
        source: Knowledge source
        creation_time: Concept creation time
        usage_count: Number of times used
    """
    concept_id: str
    name: str
    definition: str
    properties: Dict[str, Any]
    examples: List[str]
    confidence: float
    source: str
    creation_time: datetime
    usage_count: int


@dataclass
class KnowledgeRelation:
    """
    Knowledge relation between concepts.
    
    Attributes:
        relation_id: Unique relation identifier
        source_concept: Source concept identifier
        target_concept: Target concept identifier
        relation_type: Type of relation
        strength: Relation strength (0.0-1.0)
        confidence: Relation confidence
        evidence: Supporting evidence
        context: Relation context
        bidirectional: Whether relation is bidirectional
    """
    relation_id: str
    source_concept: str
    target_concept: str
    relation_type: KnowledgeRelation
    strength: float
    confidence: float
    evidence: List[str]
    context: str
    bidirectional: bool


@dataclass
class KnowledgeGraph:
    """
    Knowledge graph representation.
    
    Attributes:
        graph_id: Unique graph identifier
        concepts: Set of knowledge concepts
        relations: Set of knowledge relations
        graph_structure: NetworkX graph structure
        topological_properties: Graph topology properties
        clustering_coefficients: Clustering information
        centrality_measures: Node centrality measures
        communities: Graph communities/clusters
    """
    graph_id: str
    concepts: Set[KnowledgeConcept]
    relations: Set[KnowledgeRelation]
    graph_structure: nx.Graph
    topological_properties: Dict[str, Any]
    clustering_coefficients: Dict[str, float]
    centrality_measures: Dict[str, Dict[str, float]]
    communities: List[List[str]]


@dataclass
class SemanticMemory:
    """
    Semantic memory system.
    
    Attributes:
        knowledge_graphs: Collection of knowledge graphs
        concept_hierarchies: Concept hierarchy structures
        inferential_chains: Inferential reasoning chains
        similarity_networks: Concept similarity networks
        analogy_structures: Analogical reasoning structures
        semantic_networks: Semantic network representations
    """
    knowledge_graphs: Dict[str, KnowledgeGraph]
    concept_hierarchies: Dict[str, Dict[str, Any]]
    inferential_chains: List[List[str]]
    similarity_networks: Dict[str, Dict[str, float]]
    analogy_structures: List[Dict[str, Any]]
    semantic_networks: Dict[str, nx.Graph]


@dataclass
class MemoryConsolidation:
    """
    Memory consolidation process.
    
    Attributes:
        consolidation_id: Unique consolidation identifier
        source_items: Items to be consolidated
        target_tier: Target memory tier
        consolidation_strength: Consolidation strength
        interference_resistance: Interference resistance
        transfer_completeness: Transfer completeness
        time_requirements: Time requirements for consolidation
        metabolic_cost: Metabolic cost of consolidation
        success_probability: Consolidation success probability
    """
    consolidation_id: str
    source_items: List[str]
    target_tier: MemoryTier
    consolidation_strength: float
    interference_resistance: float
    transfer_completeness: float
    time_requirements: float
    metabolic_cost: float
    success_probability: float


@dataclass
class HierarchicalMemory:
    """
    Multi-tier hierarchical memory system.
    
    Attributes:
        working_memory: Working memory buffer
        short_term_memory: Short-term memory store
        episodic_memory: Episodic memory system
        semantic_memory: Semantic memory system
        procedural_memory: Procedural memory store
        consolidation_processes: Active consolidation processes
        memory_regulation: Memory regulation mechanisms
        capacity_limits: Tier capacity limitations
        transfer_patterns: Memory transfer patterns
    """
    working_memory: WorkingMemory
    short_term_memory: List[MemoryItem]
    episodic_memory: EpisodicMemory
    semantic_memory: SemanticMemory
    procedural_memory: Dict[str, Any]
    consolidation_processes: List[MemoryConsolidation]
    memory_regulation: Dict[str, Any]
    capacity_limits: Dict[MemoryTier, int]
    transfer_patterns: Dict[MemoryTier, Dict[str, float]]


@dataclass
class ReasoningQuery:
    """
    Semantic reasoning query.
    
    Attributes:
        query_id: Unique query identifier
        query_type: Type of reasoning query
        concepts: Concepts involved in reasoning
        relations: Relations to explore
        reasoning_path: Reasoning path taken
        conclusion: Reasoning conclusion
        confidence: Reasoning confidence
        evidence: Supporting evidence
        alternative_paths: Alternative reasoning paths
    """
    query_id: str
    query_type: ReasoningType
    concepts: List[str]
    relations: List[KnowledgeRelation]
    reasoning_path: List[str]
    conclusion: Any
    confidence: float
    evidence: List[str]
    alternative_paths: List[List[str]]


@dataclass
class MemoryRetrieval:
    """
    Memory retrieval operation.
    
    Attributes:
        retrieval_id: Unique retrieval identifier
        query: Retrieval query
        retrieved_items: Retrieved memory items
        retrieval_strength: Retrieval strength
        cue_effectiveness: Cue effectiveness
        interference_level: Interference level
        retrieval_time: Time taken for retrieval
        reconstruction_accuracy: Reconstruction accuracy
        associative_strengths: Associative connection strengths
    """
    retrieval_id: str
    query: Any
    retrieved_items: List[MemoryItem]
    retrieval_strength: float
    cue_effectiveness: float
    interference_level: float
    retrieval_time: float
    reconstruction_accuracy: float
    associative_strengths: Dict[str, float]


@dataclass
class MemoryConfig:
    """
    Memory system configuration.
    
    Attributes:
        tier_capacities: Capacity limits per memory tier
        consolidation_rates: Consolidation rates
        decay_parameters: Memory decay parameters
        attention_mechanisms: Attention control mechanisms
        consolidation_triggers: Consolidation trigger conditions
        interference_management: Interference management strategies
        biological_constraints: Biological plausibility constraints
    """
    tier_capacities: Dict[MemoryTier, int]
    consolidation_rates: Dict[Tuple[MemoryTier, MemoryTier], float]
    decay_parameters: Dict[MemoryTier, float]
    attention_mechanisms: Dict[str, Any]
    consolidation_triggers: Dict[MemoryTier, List[str]]
    interference_management: Dict[str, Any]
    biological_constraints: Dict[str, float]


# Export all interfaces
__all__ = [
    'MemoryTier',
    'ConsolidationType',
    'KnowledgeRelation',
    'ReasoningType',
    'MemoryItem',
    'WorkingMemory',
    'EpisodicMemory',
    'KnowledgeConcept',
    'KnowledgeRelation',
    'KnowledgeGraph',
    'SemanticMemory',
    'MemoryConsolidation',
    'HierarchicalMemory',
    'ReasoningQuery',
    'MemoryRetrieval',
    'MemoryConfig'
]