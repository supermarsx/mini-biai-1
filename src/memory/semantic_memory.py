#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic Memory System

A knowledge representation system that models concepts, relationships, 
and contextual understanding using graph-based structures.

Author: MiniMax AI
Version: 1.0.0
Created: 2025-11-06

Features:
- NetworkX-based concept graphs
- Multi-type relationships (is_a, has_a, part_of, etc.)
- Attribute knowledge representation
- Contextual reasoning and inference
- Path finding and knowledge traversal
- Similarity and association discovery
- Persistent storage with JSON serialization
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from pathlib import Path
import networkx as nx
from collections import defaultdict, deque
import hashlib
import time
import threading


class RelationshipType(Enum):
    """Types of semantic relationships between concepts."""
    # Hierarchical relationships
    IS_A = auto()          # "Dog is_a Mammal"
    PART_OF = auto()       # "Wheel part_of Car"
    MEMBER_OF = auto()     # "Wheel member_of Car"
    
    # Associative relationships  
    HAS_A = auto()         # "Car has_a Wheel"
    RELATED_TO = auto()    # "Dog related_to Cat"
    SIMILAR_TO = auto()    # "Dog similar_to Wolf"
    
    # Functional relationships
    CAUSES = auto()        # "Rain causes Wet"
    ENABLES = auto()       # "Wheel enables Movement"
    REQUIRES = auto()      # "Car requires Fuel"
    PRODUCES = auto()      # "Tree produces Oxygen"
    
    # Spatial relationships
    LOCATED_IN = auto()    # "Paris located_in France"
    CONTAINS = auto()      # "France contains Paris"
    
    # Temporal relationships
    OCCURS_BEFORE = auto() # "Birth occurs_before Death"
    OCCURS_AFTER = auto()  # "Death occurs_after Birth"
    OCCURS_DURING = auto() # "Dinner occurs_during Evening"
    
    # Conceptual relationships
    EXAMPLE_OF = auto()    # "Poodle example_of Dog"
    KIND_OF = auto()       # "Poodle kind_of Dog"
    
    @property
    def inverse(self) -> 'RelationshipType':
        """Get the inverse relationship type."""
        inverses = {
            RelationshipType.IS_A: RelationshipType.HAS_EXAMPLES,
            RelationshipType.PART_OF: RelationshipType.CONTAINS,
            RelationshipType.HAS_A: RelationshipType.IS_A,
            RelationshipType.CAUSES: RelationshipType.CAUSED_BY,
            RelationshipType.ENABLES: RelationshipType.ENABLED_BY,
            RelationshipType.REQUIRES: RelationshipType.REQUIRED_BY,
            RelationshipType.PRODUCES: RelationshipType.CONSUMED_BY,
            RelationshipType.LOCATED_IN: RelationshipType.CONTAINS,
            RelationshipType.OCCURS_BEFORE: RelationshipType.OCCURS_AFTER,
            RelationshipType.OCCURS_AFTER: RelationshipType.OCCURS_BEFORE,
            RelationshipType.EXAMPLE_OF: RelationshipType.HAS_EXAMPLES,
            RelationshipType.KIND_OF: RelationshipType.HAS_KINDS,
            
            # Self-inverse relationships
            RelationshipType.RELATED_TO: RelationshipType.RELATED_TO,
            RelationshipType.SIMILAR_TO: RelationshipType.SIMILAR_TO,
            RelationshipType.CONTAINS: RelationshipType.PART_OF,
            RelationshipType.HAS_EXAMPLES: RelationshipType.EXAMPLE_OF,
            RelationshipType.HAS_KINDS: RelationshipType.KIND_OF,
            RelationshipType.CAUSED_BY: RelationshipType.CAUSES,
            RelationshipType.ENABLED_BY: RelationshipType.ENABLES,
            RelationshipType.REQUIRED_BY: RelationshipType.REQUIRES,
            RelationshipType.CONSUMED_BY: RelationshipType.PRODUCES,
            RelationshipType.MEMBER_OF: RelationshipType.HAS_MEMBERS,
            RelationshipType.HAS_MEMBERS: RelationshipType.MEMBER_OF
        }
        return inverses.get(self, self)
    
    @property
    def is_hierarchical(self) -> bool:
        """Check if this is a hierarchical relationship."""
        return self in {
            RelationshipType.IS_A, 
            RelationshipType.PART_OF, 
            RelationshipType.MEMBER_OF,
            RelationshipType.KIND_OF,
            RelationshipType.EXAMPLE_OF
        }
    
    @property 
    def is_directional(self) -> bool:
        """Check if this relationship has directionality."""
        return self not in {
            RelationshipType.RELATED_TO,
            RelationshipType.SIMILAR_TO
        }


@dataclass
class ConceptNode:
    """Represents a concept in the semantic network."""
    id: str
    name: str
    concept_type: str = "entity"  # entity, attribute, event, abstract
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConceptNode':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Relationship:
    """Represents a relationship between concepts."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['relationship_type'] = self.relationship_type.name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Relationship':
        """Create from dictionary."""
        data['relationship_type'] = RelationshipType[data['relationship_type']]
        return cls(**data)


class SemanticMemorySystem:
    """
    Semantic memory system for knowledge representation and reasoning.
    
    Features:
    - NetworkX-based concept graphs with multiple relationship types
    - Hierarchical and associative knowledge modeling
    - Path finding and knowledge traversal algorithms
    - Similarity and association discovery
    - Contextual reasoning and inference
    - Persistent storage with JSON serialization
    - Thread-safe concurrent access
    - Performance optimization for large knowledge graphs
    """
    
    def __init__(
        self,
        storage_path: str = "semantic_memory.json",
        auto_save: bool = True,
        save_interval: int = 300  # 5 minutes
    ):
        self.storage_path = Path(storage_path)
        self.auto_save = auto_save
        self.save_interval = save_interval
        
        # Core graph structures
        self.concept_graph = nx.MultiDiGraph()  # Main knowledge graph
        self.attribute_graph = nx.Graph()       # Attribute relationships
        
        # Storage for nodes and relationships
        self.concepts: Dict[str, ConceptNode] = {}
        self.relationships: List[Relationship] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Performance statistics
        self._stats = {
            'concepts_added': 0,
            'relationships_added': 0,
            'queries_executed': 0,
            'paths_found': 0,
            'avg_query_time': 0.0
        }
        
        # Background save task
        self._save_task: Optional[asyncio.Task] = None
        
        if auto_save:
            self._start_background_save()
        
        # Load existing data
        self.load_from_disk()
    
    def _start_background_save(self):
        """Start background save task."""
        loop = asyncio.get_event_loop()
        self._save_task = loop.create_task(self._background_save_worker())
    
    def stop_background_save(self):
        """Stop background save task."""
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
    
    async def _background_save_worker(self):
        """Background worker for periodic saves."""
        while True:
            try:
                await asyncio.sleep(self.save_interval)
                await self.save_to_disk()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Background save error: {e}")
    
    def _generate_concept_id(self, name: str, concept_type: str = "entity") -> str:
        """Generate unique ID for a concept."""
        content = f"{name}:{concept_type}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    def add_concept(
        self, 
        name: str, 
        concept_type: str = "entity",
        description: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> str:
        """
        Add a new concept to the semantic memory.
        
        Args:
            name: Concept name/label
            concept_type: Type of concept (entity, attribute, event, abstract)
            description: Detailed description
            attributes: Key-value attributes
            confidence: Confidence level (0.0 to 1.0)
            
        Returns:
            Concept ID
        """
        with self._lock:
            concept_id = self._generate_concept_id(name, concept_type)
            
            # Check if concept already exists
            if concept_id in self.concepts:
                return concept_id
            
            # Create concept node
            concept = ConceptNode(
                id=concept_id,
                name=name,
                concept_type=concept_type,
                description=description,
                attributes=attributes or {},
                confidence=confidence
            )
            
            # Add to storage
            self.concepts[concept_id] = concept
            
            # Add to graph
            self.concept_graph.add_node(
                concept_id,
                name=name,
                type=concept_type,
                description=description,
                attributes=attributes or {},
                confidence=confidence
            )
            
            # Update statistics
            self._stats['concepts_added'] += 1
            
            return concept_id
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        strength: float = 1.0,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a relationship between concepts.
        
        Args:
            source_id: Source concept ID
            target_id: Target concept ID
            relationship_type: Type of relationship
            strength: Relationship strength (0.0 to 1.0)
            confidence: Confidence in relationship (0.0 to 1.0)
            metadata: Additional relationship metadata
            
        Returns:
            True if relationship was added successfully
        """
        with self._lock:
            # Verify concepts exist
            if source_id not in self.concepts or target_id not in self.concepts:
                return False
            
            # Create relationship
            relationship = Relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence,
                metadata=metadata or {}
            )
            
            # Add to storage
            self.relationships.append(relationship)
            
            # Add to graph with edge attributes
            self.concept_graph.add_edge(
                source_id,
                target_id,
                relationship_type=relationship_type,
                strength=strength,
                confidence=confidence,
                metadata=metadata or {},
                key=f"{source_id}_{target_id}_{relationship_type.name}"
            )
            
            # Also add inverse relationship if not self-inverse
            if relationship_type.inverse != relationship_type:
                inverse_relationship = Relationship(
                    source_id=target_id,
                    target_id=source_id,
                    relationship_type=relationship_type.inverse,
                    strength=strength,
                    confidence=confidence,
                    metadata=metadata or {}
                )
                self.relationships.append(inverse_relationship)
            
            # Update statistics
            self._stats['relationships_added'] += 1
            
            return True
    
    def get_concept(self, concept_id: str) -> Optional[ConceptNode]:
        """Get concept by ID."""
        with self._lock:
            return self.concepts.get(concept_id)
    
    def find_concepts(
        self, 
        name_pattern: Optional[str] = None,
        concept_type: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[ConceptNode]:
        """
        Find concepts matching criteria.
        
        Args:
            name_pattern: Name pattern to match (substring)
            concept_type: Filter by concept type
            attributes: Filter by attributes (all must match)
            limit: Maximum number of results
            
        Returns:
            List of matching ConceptNode objects
        """
        with self._lock:
            results = []
            
            for concept in self.concepts.values():
                # Filter by name pattern
                if name_pattern and name_pattern.lower() not in concept.name.lower():
                    continue
                
                # Filter by concept type
                if concept_type and concept.concept_type != concept_type:
                    continue
                
                # Filter by attributes
                if attributes:
                    match = True
                    for key, value in attributes.items():
                        if key not in concept.attributes or concept.attributes[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(concept)
                
                if len(results) >= limit:
                    break
            
            return results
    
    def get_relationships(
        self,
        concept_id: str,
        relationship_type: Optional[RelationshipType] = None,
        outgoing: bool = True,
        incoming: bool = True
    ) -> List[Tuple[Relationship, ConceptNode]]:
        """
        Get relationships for a concept.
        
        Args:
            concept_id: Source concept ID
            relationship_type: Filter by relationship type
            outgoing: Include outgoing relationships
            incoming: Include incoming relationships
            
        Returns:
            List of (Relationship, related_concept) tuples
        """
        with self._lock:
            results = []
            
            for relationship in self.relationships:
                # Check if relationship involves our concept
                is_outgoing = relationship.source_id == concept_id
                is_incoming = relationship.target_id == concept_id
                
                if not ((outgoing and is_outgoing) or (incoming and is_incoming)):
                    continue
                
                # Filter by relationship type
                if relationship_type and relationship.relationship_type != relationship_type:
                    continue
                
                # Get the related concept
                related_id = relationship.target_id if is_outgoing else relationship.source_id
                related_concept = self.concepts.get(related_id)
                
                if related_concept:
                    results.append((relationship, related_concept))
            
            return results
    
    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
        relationship_types: Optional[List[RelationshipType]] = None
    ) -> List[List[Tuple[Relationship, ConceptNode]]]:
        """
        Find paths between two concepts.
        
        Args:
            source_id: Starting concept ID
            target_id: Target concept ID
            max_depth: Maximum path length
            relationship_types: Allowed relationship types
            
        Returns:
            List of paths, each path is list of (relationship, concept) tuples
        """
        with self._lock:
            start_time = time.time()
            
            paths = []
            
            try:
                # Use NetworkX path finding with custom edge filter
                def edge_filter(u, v, key):
                    edge_data = self.concept_graph[u][v][key]
                    rel_type = edge_data['relationship_type']
                    
                    if relationship_types and rel_type not in relationship_types:
                        return False
                    
                    return True
                
                # Find all simple paths
                nx_paths = list(nx.all_simple_paths(
                    self.concept_graph,
                    source_id,
                    target_id,
                    cutoff=max_depth,
                    edge_filter=edge_filter
                ))
                
                # Convert to relationship sequences
                for path in nx_paths:
                    path_relationships = []
                    
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i + 1]
                        
                        # Find the edge between u and v
                        for key in self.concept_graph[u][v]:
                            edge_data = self.concept_graph[u][v][key]
                            if edge_filter(u, v, key):
                                # Find corresponding relationship
                                for rel in self.relationships:
                                    if (rel.source_id == u and rel.target_id == v and
                                        rel.relationship_type == edge_data['relationship_type']):
                                        related_concept = self.concepts[v]
                                        path_relationships.append((rel, related_concept))
                                        break
                                break
                    
                    if path_relationships:
                        paths.append(path_relationships)
                
                # Update statistics
                self._stats['paths_found'] += len(paths)
                
            finally:
                query_time = time.time() - start_time
                self._stats['queries_executed'] += 1
                self._stats['avg_query_time'] = (
                    (self._stats['avg_query_time'] * (self._stats['queries_executed'] - 1) + query_time) /
                    self._stats['queries_executed']
                )
            
            return paths
    
    def get_concept_context(
        self,
        concept_id: str,
        max_depth: int = 2,
        include_attributes: bool = True
    ) -> Dict[str, Any]:
        """
        Get the complete context of a concept including related concepts and attributes.
        
        Args:
            concept_id: Concept ID to analyze
            max_depth: Maximum depth for related concepts
            include_attributes: Whether to include attribute details
            
        Returns:
            Context dictionary with concept info and related concepts
        """
        with self._lock:
            concept = self.concepts.get(concept_id)
            if not concept:
                return {}
            
            context = {
                'concept': concept.to_dict(),
                'related_concepts': [],
                'attributes': {},
                'statistics': {
                    'total_relationships': 0,
                    'relationship_types': {}
                }
            }
            
            # Get direct relationships
            relationships = self.get_relationships(
                concept_id, 
                outgoing=True, 
                incoming=True
            )
            
            for relationship, related_concept in relationships:
                rel_info = {
                    'relationship_type': relationship.relationship_type.name,
                    'related_concept': related_concept.to_dict(),
                    'strength': relationship.strength,
                    'confidence': relationship.confidence
                }
                
                context['related_concepts'].append(rel_info)
                
                # Count relationship types
                rel_type = relationship.relationship_type.name
                context['statistics']['relationship_types'][rel_type] = \
                    context['statistics']['relationship_types'].get(rel_type, 0) + 1
            
            context['statistics']['total_relationships'] = len(relationships)
            
            # Add attributes if requested
            if include_attributes:
                context['attributes'] = concept.attributes.copy()
            
            return context
    
    def compute_similarity(
        self,
        concept_id_1: str,
        concept_id_2: str,
        method: str = "jaccard"
    ) -> float:
        """
        Compute similarity between two concepts.
        
        Args:
            concept_id_1: First concept ID
            concept_id_2: Second concept ID
            method: Similarity method ('jaccard', 'cosine', 'path_based')
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        with self._lock:
            concept1 = self.concepts.get(concept_id_1)
            concept2 = self.concepts.get(concept_id_2)
            
            if not concept1 or not concept2:
                return 0.0
            
            if method == "jaccard":
                # Jaccard similarity based on common attributes
                attrs1 = set(str(v) for v in concept1.attributes.values())
                attrs2 = set(str(v) for v in concept2.attributes.values())
                
                if not attrs1 and not attrs2:
                    return 1.0
                
                intersection = len(attrs1.intersection(attrs2))
                union = len(attrs1.union(attrs2))
                
                return intersection / union if union > 0 else 0.0
            
            elif method == "cosine":
                # Cosine similarity based on attribute vectors
                all_attrs = set(concept1.attributes.keys()).union(concept2.attributes.keys())
                
                if not all_attrs:
                    return 1.0
                
                vec1 = np.array([
                    concept1.attributes.get(attr, 0) for attr in all_attrs
                ])
                vec2 = np.array([
                    concept2.attributes.get(attr, 0) for attr in all_attrs
                ])
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                return np.dot(vec1, vec2) / (norm1 * norm2)
            
            elif method == "path_based":
                # Similarity based on shortest path distance
                try:
                    path_length = nx.shortest_path_length(
                        self.concept_graph,
                        concept_id_1,
                        concept_id_2
                    )
                    
                    # Convert path length to similarity (closer = more similar)
                    return 1.0 / (1.0 + path_length)
                except nx.NetworkXNoPath:
                    return 0.0
            
            return 0.0
    
    def discover_associations(
        self,
        concept_id: str,
        min_strength: float = 0.1,
        limit: int = 20
    ) -> List[Tuple[ConceptNode, float, str]]:
        """
        Discover concepts associated with the given concept.
        
        Args:
            concept_id: Source concept ID
            min_strength: Minimum relationship strength
            limit: Maximum number of associations
            
        Returns:
            List of (concept, strength, relationship_type) tuples
        """
        with self._lock:
            associations = []
            
            # Get all relationships
            relationships = self.get_relationships(
                concept_id,
                outgoing=True,
                incoming=True
            )
            
            for relationship, related_concept in relationships:
                if relationship.strength >= min_strength:
                    associations.append((
                        related_concept,
                        relationship.strength,
                        relationship.relationship_type.name
                    ))
            
            # Sort by strength and return top results
            associations.sort(key=lambda x: x[1], reverse=True)
            return associations[:limit]
    
    async def save_to_disk(self):
        """Save semantic memory to disk."""
        with self._lock:
            data = {
                'concepts': {cid: concept.to_dict() for cid, concept in self.concepts.items()},
                'relationships': [rel.to_dict() for rel in self.relationships],
                'metadata': {
                    'created_at': time.time(),
                    'version': '1.0.0',
                    'total_concepts': len(self.concepts),
                    'total_relationships': len(self.relationships)
                }
            }
            
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load_from_disk(self):
        """Load semantic memory from disk."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            with self._lock:
                # Load concepts
                for cid, concept_data in data.get('concepts', {}).items():
                    concept = ConceptNode.from_dict(concept_data)
                    self.concepts[cid] = concept
                    
                    # Add to graph
                    self.concept_graph.add_node(
                        cid,
                        name=concept.name,
                        type=concept.concept_type,
                        description=concept.description,
                        attributes=concept.attributes,
                        confidence=concept.confidence
                    )
                
                # Load relationships
                for rel_data in data.get('relationships', []):
                    relationship = Relationship.from_dict(rel_data)
                    self.relationships.append(relationship)
                    
                    # Add to graph
                    self.concept_graph.add_edge(
                        relationship.source_id,
                        relationship.target_id,
                        relationship_type=relationship.relationship_type,
                        strength=relationship.strength,
                        confidence=relationship.confidence,
                        metadata=relationship.metadata,
                        key=f"{relationship.source_id}_{relationship.target_id}_{relationship.relationship_type.name}"
                    )
        
        except Exception as e:
            print(f"Failed to load semantic memory: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self._lock:
            # Graph statistics
            graph_stats = {
                'nodes': self.concept_graph.number_of_nodes(),
                'edges': self.concept_graph.number_of_edges(),
                'is_connected': nx.is_weakly_connected(self.concept_graph),
                'density': nx.density(self.concept_graph)
            }
            
            # Relationship type distribution
            rel_type_counts = {}
            for rel in self.relationships:
                rel_type = rel.relationship_type.name
                rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
            
            # Concept type distribution
            concept_type_counts = {}
            for concept in self.concepts.values():
                ctype = concept.concept_type
                concept_type_counts[ctype] = concept_type_counts.get(ctype, 0) + 1
            
            return {
                'total_concepts': len(self.concepts),
                'total_relationships': len(self.relationships),
                'graph_statistics': graph_stats,
                'relationship_types': rel_type_counts,
                'concept_types': concept_type_counts,
                'performance': self._stats.copy(),
                'storage_path': str(self.storage_path)
            }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_background_save()
        if self.auto_save:
            asyncio.create_task(self.save_to_disk())


# Demo function
async def semantic_memory_demo():
    """Demonstrate the semantic memory system."""
    print("=== Semantic Memory System Demo ===")
    
    async with SemanticMemorySystem(storage_path=":memory:") as semantic_memory:
        
        print("\n1. Building animal kingdom knowledge base:")
        
        # Add animal concepts
        animal_ids = {}
        animals = [
            ("Animal", "Kingdom of living organisms", {"kingdom": "Animalia"}),
            ("Mammal", "Warm-blooded vertebrate with hair", {"class": "Mammalia"}),
            ("Bird", "Feathered vertebrate capable of flight", {"class": "Aves"}),
            ("Fish", "Aquatic vertebrate with gills", {"class": "Pisces"}),
            ("Dog", "Domesticated canine animal", {"domesticated": True, "color": "various"}),
            ("Cat", "Domesticated feline animal", {"domesticated": True, "color": "various"}),
            ("Eagle", "Large bird of prey", {"diet": "carnivore", "habitat": "mountains"}),
            ("Shark", "Large predatory fish", {"diet": "carnivore", "habitat": "ocean"}),
        ]
        
        for name, description, attributes in animals:
            concept_id = semantic_memory.add_concept(
                name=name,
                concept_type="entity",
                description=description,
                attributes=attributes
            )
            animal_ids[name] = concept_id
            print(f"   Added: {name} ({concept_id})")
        
        print("\n2. Creating taxonomic relationships:")
        
        # Add hierarchical relationships
        relationships = [
            (animal_ids["Mammal"], animal_ids["Animal"], RelationshipType.IS_A),
            (animal_ids["Bird"], animal_ids["Animal"], RelationshipType.IS_A),
            (animal_ids["Fish"], animal_ids["Animal"], RelationshipType.IS_A),
            (animal_ids["Dog"], animal_ids["Mammal"], RelationshipType.IS_A),
            (animal_ids["Cat"], animal_ids["Mammal"], RelationshipType.IS_A),
            (animal_ids["Eagle"], animal_ids["Bird"], RelationshipType.IS_A),
            (animal_ids["Shark"], animal_ids["Fish"], RelationshipType.IS_A),
        ]
        
        for source, target, rel_type in relationships:
            semantic_memory.add_relationship(source, target, rel_type)
            print(f"   {semantic_memory.concepts[source].name} {rel_type.name.lower()} {semantic_memory.concepts[target].name}")
        
        print("\n3. Adding functional relationships:")
        
        # Add functional relationships
        functional_rels = [
            (animal_ids["Dog"], animal_ids["Cat"], RelationshipType.RELATED_TO, 0.8),
            (animal_ids["Eagle"], animal_ids["Bird"], RelationshipType.PART_OF, 1.0),
            (animal_ids["Shark"], animal_ids["Fish"], RelationshipType.PART_OF, 1.0),
        ]
        
        for source, target, rel_type, strength in functional_rels:
            semantic_memory.add_relationship(source, target, rel_type, strength=strength)
            print(f"   {semantic_memory.concepts[source].name} {rel_type.name.lower()} {semantic_memory.concepts[target].name} (strength: {strength})")
        
        print("\n4. Querying concept relationships:")
        
        # Get relationships for Dog
        dog_relationships = semantic_memory.get_relationships(animal_ids["Dog"])
        print(f"\n   Dog relationships ({len(dog_relationships)} found):")
        for relationship, related_concept in dog_relationships[:5]:
            print(f"     {relationship.relationship_type.name}: {related_concept.name}")
        
        print("\n5. Finding paths between concepts:")
        
        # Find path from Dog to Animal
        dog_to_animal = semantic_memory.find_paths(
            animal_ids["Dog"], 
            animal_ids["Animal"], 
            max_depth=3
        )
        
        print(f"   Paths from Dog to Animal ({len(dog_to_animal)} found):")
        for i, path in enumerate(dog_to_animal[:3], 1):
            path_names = [semantic_memory.concepts[animal_ids["Dog"]].name]
            for rel, concept in path:
                path_names.append(f" --{rel.relationship_type.name}--> {concept.name}")
            print(f"     Path {i}: {''.join(path_names)}")
        
        print("\n6. Computing concept similarity:")
        
        # Compute similarity between Dog and Cat
        dog_cat_sim = semantic_memory.compute_similarity(
            animal_ids["Dog"], 
            animal_ids["Cat"], 
            method="jaccard"
        )
        print(f"   Dog-Cat similarity (Jaccard): {dog_cat_sim:.3f}")
        
        # Compute path-based similarity
        dog_shark_sim = semantic_memory.compute_similarity(
            animal_ids["Dog"], 
            animal_ids["Shark"], 
            method="path_based"
        )
        print(f"   Dog-Shark similarity (path-based): {dog_shark_sim:.3f}")
        
        print("\n7. Discovering concept associations:")
        
        # Get associations for Dog
        dog_associations = semantic_memory.discover_associations(
            animal_ids["Dog"], 
            min_strength=0.1
        )
        
        print(f"   Dog associations ({len(dog_associations)} found):")
        for concept, strength, rel_type in dog_associations[:5]:
            print(f"     {concept.name} ({rel_type}, strength: {strength:.2f})")
        
        print("\n8. Getting concept context:")
        
        # Get complete context for Dog
        dog_context = semantic_memory.get_concept_context(animal_ids["Dog"])
        
        print(f"   Dog context:")
        print(f"     Name: {dog_context['concept']['name']}")
        print(f"     Type: {dog_context['concept']['concept_type']}")
        print(f"     Description: {dog_context['concept']['description']}")
        print(f"     Attributes: {dog_context['attributes']}")
        print(f"     Related concepts: {dog_context['statistics']['total_relationships']}")
        
        print("\n9. Finding concepts by criteria:")
        
        # Find all mammals
        mammals = semantic_memory.find_concepts(
            concept_type="entity",
            attributes={"kingdom": "Animalia"}
        )
        
        print(f"   Concepts with kingdom='Animalia': {len(mammals)} found")
        for concept in mammals:
            print(f"     - {concept.name}")
        
        print("\n10. System statistics:")
        
        stats = semantic_memory.get_statistics()
        print(f"   Total concepts: {stats['total_concepts']}")
        print(f"   Total relationships: {stats['total_relationships']}")
        print(f"   Graph nodes: {stats['graph_statistics']['nodes']}")
        print(f"   Graph edges: {stats['graph_statistics']['edges']}")
        print(f"   Graph density: {stats['graph_statistics']['density']:.3f}")
        
        print(f"\n   Relationship types:")
        for rel_type, count in stats['relationship_types'].items():
            print(f"     {rel_type}: {count}")
        
        print(f"\n   Performance:")
        perf = stats['performance']
        print(f"     Concepts added: {perf['concepts_added']}")
        print(f"     Relationships added: {perf['relationships_added']}")
        print(f"     Queries executed: {perf['queries_executed']}")
        print(f"     Paths found: {perf['paths_found']}")
        print(f"     Avg query time: {perf['avg_query_time']:.4f}s")
        
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(semantic_memory_demo())
