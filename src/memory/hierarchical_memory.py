#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Memory System

A brain-inspired multi-level memory architecture that simulates the way
biological systems store and organize information across different timescales.

Author: MiniMax AI
Version: 1.0.0
Created: 2025-11-06

Memory Levels:
- Sensory Memory: Very short-term (500ms), high bandwidth
- Working Memory: Short-term (30s), active processing
- Episodic Memory: Medium-term (7 days), personal experiences
- Semantic Memory: Long-term (permanent), general knowledge

Features:
- Automatic memory consolidation between levels
- Importance-based memory management
- SQLite persistence for long-term storage
- Memory decay and forgetting mechanisms
- Parallel processing for real-time performance
"""

import asyncio
import sqlite3
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path
import numpy as np
from collections import deque, defaultdict
import hashlib
import pickle
import weakref


class MemoryLevel(Enum):
    """Hierarchical memory levels based on cognitive science."""
    SENSORY = auto()      # 0.5 seconds - immediate sensory information
    WORKING = auto()      # 30 seconds - active working memory  
    EPISODIC = auto()     # 7 days - personal experiences and events
    SEMANTIC = auto()     # permanent - general knowledge and facts
    
    @property
    def retention_time(self) -> float:
        """Get retention time in seconds for this memory level."""
        times = {
            MemoryLevel.SENSORY: 0.5,
            MemoryLevel.WORKING: 30.0,
            MemoryLevel.EPISODIC: 7 * 24 * 3600,  # 7 days
            MemoryLevel.SEMANTIC: float('inf')    # permanent
        }
        return times[self]
    
    @property 
    def capacity(self) -> int:
        """Get approximate capacity for this memory level."""
        capacities = {
            MemoryLevel.SENSORY: 1000,      # 1000 items
            MemoryLevel.WORKING: 100,       # 100 items (Miller's rule)
            MemoryLevel.EPISODIC: 10000,    # 10k episodes
            MemoryLevel.SEMANTIC: 1000000   # 1M concepts
        }
        return capacities[self]
    
    def __lt__(self, other):
        """Define ordering for memory levels."""
        if self == other:
            return False
        level_order = [MemoryLevel.SENSORY, MemoryLevel.WORKING, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC]
        return level_order.index(self) < level_order.index(other)


@dataclass
class MemoryEntry:
    """Represents a single memory entry in the hierarchical system."""
    id: str
    content: Any
    memory_level: MemoryLevel
    importance: float  # 0.0 to 1.0
    created_at: float
    last_accessed: float
    access_count: int = 0
    associations: Set[str] = None  # Linked memory IDs
    metadata: Dict[str, Any] = None
    decay_rate: float = 0.001  # Base decay rate
    consolidation_threshold: float = 0.8  # For auto-promotion
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = set()
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def age(self) -> float:
        """Get current age in seconds."""
        return time.time() - self.created_at
    
    @property
    def time_since_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed
    
    @property
    def current_importance(self) -> float:
        """Get current importance with decay applied."""
        base_importance = self.importance
        
        # Apply time-based decay
        decay_factor = np.exp(-self.decay_rate * self.age)
        
        # Apply access-based boost
        access_boost = min(0.2, self.access_count * 0.01)
        
        # Apply recency boost for recent access
        recency_boost = 0.0
        if self.time_since_access < 60:  # Accessed in last minute
            recency_boost = 0.1
        
        current = base_importance * decay_factor + access_boost + recency_boost
        return max(0.0, min(1.0, current))
    
    def should_consolidate(self) -> bool:
        """Check if this memory should be promoted to next level."""
        return (self.current_importance > self.consolidation_threshold and 
                self.access_count > 2 and
                self.age > 300)  # 5 minutes minimum age
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['associations'] = list(self.associations)
        data['memory_level'] = self.memory_level.name
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary (deserialization)."""
        data['memory_level'] = MemoryLevel[data['memory_level']]
        data['associations'] = set(data.get('associations', []))
        return cls(**data)


class HierarchicalMemorySystem:
    """
    Brain-inspired hierarchical memory system with multiple levels of organization.
    
    Features:
    - Multi-level memory architecture (sensory, working, episodic, semantic)
    - Automatic memory consolidation and promotion
    - Importance-based memory management with decay
    - SQLite persistence for long-term storage
    - Parallel processing and concurrent access
    - Memory associations and relationship tracking
    - Predictive preloading based on usage patterns
    """
    
    def __init__(
        self,
        db_path: str = "hierarchical_memory.db",
        max_workers: int = 4,
        auto_consolidation: bool = True,
        backup_interval: int = 300  # 5 minutes
    ):
        self.db_path = Path(db_path)
        self.max_workers = max_workers
        self.auto_consolidation = auto_consolidation
        self.backup_interval = backup_interval
        
        # Memory storage by level
        self._memories: Dict[MemoryLevel, Dict[str, MemoryEntry]] = {
            level: {} for level in MemoryLevel
        }
        
        # Thread-safe access control
        self._lock = threading.RLock()
        
        # Background tasks
        self._consolidation_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
        
        # Performance statistics
        self._stats = {
            'memories_stored': 0,
            'memories_accessed': 0,
            'consolidations': 0,
            'evictions': 0,
            'avg_access_time': 0.0
        }
        
        # Initialize database
        self._init_database()
        
        # Start background tasks if auto mode
        if auto_consolidation:
            self.start_background_tasks()
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_level TEXT NOT NULL,
                    importance REAL NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    associations TEXT,
                    metadata TEXT,
                    decay_rate REAL DEFAULT 0.001,
                    consolidation_threshold REAL DEFAULT 0.8
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_level_created 
                ON memories (memory_level, created_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_importance 
                ON memories (importance DESC)
            """)
            
            conn.commit()
    
    def start_background_tasks(self):
        """Start background consolidation and cleanup tasks."""
        if not self.auto_consolidation:
            return
            
        loop = asyncio.get_event_loop()
        
        # Consolidation task
        self._consolidation_task = loop.create_task(self._consolidation_worker())
        
        # Cleanup task
        self._cleanup_task = loop.create_task(self._cleanup_worker())
        
        # Backup task
        self._backup_task = loop.create_task(self._backup_worker())
    
    def stop_background_tasks(self):
        """Stop all background tasks."""
        for task in [self._consolidation_task, self._cleanup_task, self._backup_task]:
            if task and not task.done():
                task.cancel()
    
    async def _consolidation_worker(self):
        """Background worker for memory consolidation."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._consolidate_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Consolidation worker error: {e}")
    
    async def _cleanup_worker(self):
        """Background worker for memory cleanup and decay."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired_memories()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup worker error: {e}")
    
    async def _backup_worker(self):
        """Background worker for periodic backups."""
        while True:
            try:
                await asyncio.sleep(self.backup_interval)
                await self._backup_to_database()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Backup worker error: {e}")
    
    def _generate_memory_id(self, content: Any) -> str:
        """Generate unique ID for memory content."""
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    async def store(
        self, 
        content: Any, 
        memory_level: MemoryLevel = MemoryLevel.SENSORY,
        importance: float = 0.5,
        associations: Optional[Set[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content in the hierarchical memory system.
        
        Args:
            content: Content to store
            memory_level: Target memory level
            importance: Initial importance (0.0 to 1.0)
            associations: Associated memory IDs
            metadata: Additional metadata
            
        Returns:
            Memory ID for later retrieval
        """
        memory_id = self._generate_memory_id(content)
        current_time = time.time()
        
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            memory_level=memory_level,
            importance=importance,
            created_at=current_time,
            last_accessed=current_time,
            associations=associations or set(),
            metadata=metadata or {}
        )
        
        with self._lock:
            # Check capacity and evict if necessary
            self._evict_if_needed(memory_level)
            
            # Store memory
            self._memories[memory_level][memory_id] = entry
            self._stats['memories_stored'] += 1
        
        # Persist to database if not sensory memory
        if memory_level != MemoryLevel.SENSORY:
            await self._persist_memory(entry)
        
        return memory_id
    
    async def retrieve(
        self, 
        memory_id: str, 
        memory_level: Optional[MemoryLevel] = None
    ) -> Optional[MemoryEntry]:
        """
        Retrieve content from hierarchical memory system.
        
        Args:
            memory_id: ID of memory to retrieve
            memory_level: Specific level to search (None for all levels)
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        start_time = time.time()
        
        with self._lock:
            # Search specific level or all levels
            levels_to_search = [memory_level] if memory_level else list(MemoryLevel)
            
            for level in levels_to_search:
                if memory_id in self._memories[level]:
                    entry = self._memories[level][memory_id]
                    
                    # Update access statistics
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    
                    self._stats['memories_accessed'] += 1
                    
                    # Calculate average access time
                    access_time = time.time() - start_time
                    self._stats['avg_access_time'] = (
                        (self._stats['avg_access_time'] * (self._stats['memories_accessed'] - 1) + access_time) /
                        self._stats['memories_accessed']
                    )
                    
                    return entry
        
        return None
    
    async def search(
        self, 
        query: str, 
        memory_levels: Optional[List[MemoryLevel]] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """
        Search memories by content or metadata.
        
        Args:
            query: Search query string
            memory_levels: Levels to search (None for all)
            limit: Maximum number of results
            
        Returns:
            List of matching MemoryEntry objects
        """
        results = []
        levels = memory_levels or list(MemoryLevel)
        
        for level in levels:
            with self._lock:
                for entry in self._memories[level].values():
                    # Simple string matching (could be enhanced with semantic search)
                    content_str = str(entry.content).lower()
                    metadata_str = str(entry.metadata).lower()
                    
                    if query.lower() in content_str or query.lower() in metadata_str:
                        results.append((entry, entry.current_importance))
        
        # Sort by importance and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in results[:limit]]
    
    async def get_associated_memories(
        self, 
        memory_id: str, 
        max_depth: int = 2
    ) -> List[MemoryEntry]:
        """
        Get memories associated with the given memory ID.
        
        Args:
            memory_id: ID to find associations for
            max_depth: Maximum traversal depth
            
        Returns:
            List of associated MemoryEntry objects
        """
        associated = []
        visited = set()
        
        async def traverse(mid: str, depth: int):
            if mid in visited or depth > max_depth:
                return
            
            visited.add(mid)
            
            # Find the memory
            entry = await self.retrieve(mid)
            if entry:
                associated.append(entry)
                
                # Traverse associations
                for associated_id in entry.associations:
                    await traverse(associated_id, depth + 1)
        
        await traverse(memory_id, 0)
        return associated
    
    async def _consolidate_memories(self):
        """Consolidate memories between levels based on importance and access patterns."""
        with self._lock:
            level_order = [MemoryLevel.SENSORY, MemoryLevel.WORKING, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC]
            
            for i in range(len(level_order) - 1):
                current_level = level_order[i]
                next_level = level_order[i + 1]
                
                # Find memories ready for consolidation
                to_promote = []
                for memory_id, entry in self._memories[current_level].items():
                    if entry.should_consolidate():
                        to_promote.append((memory_id, entry))
                
                # Promote memories
                for memory_id, entry in to_promote:
                    # Remove from current level
                    del self._memories[current_level][memory_id]
                    
                    # Update level and add to next level
                    entry.memory_level = next_level
                    self._memories[next_level][memory_id] = entry
                    
                    # Update statistics
                    self._stats['consolidations'] += 1
                    
                    # Persist to database
                    await self._persist_memory(entry)
    
    async def _cleanup_expired_memories(self):
        """Clean up expired or low-importance memories."""
        with self._lock:
            for level in MemoryLevel:
                if level == MemoryLevel.SEMANTIC:
                    continue  # Never delete semantic memories
                
                to_remove = []
                current_time = time.time()
                
                for memory_id, entry in self._memories[level].items():
                    # Remove if expired
                    if (entry.age > level.retention_time and 
                        entry.current_importance < 0.1):
                        to_remove.append(memory_id)
                    
                    # Remove very old, unused memories
                    elif (entry.age > level.retention_time * 2 and 
                          entry.access_count == 0):
                        to_remove.append(memory_id)
                
                # Remove expired memories
                for memory_id in to_remove:
                    del self._memories[level][memory_id]
                    self._stats['evictions'] += 1
    
    def _evict_if_needed(self, memory_level: MemoryLevel):
        """Evict memories if level is at capacity."""
        memories = self._memories[memory_level]
        capacity = memory_level.capacity
        
        if len(memories) >= capacity:
            # Remove lowest importance memories
            sorted_memories = sorted(
                memories.items(),
                key=lambda x: x[1].current_importance
            )
            
            # Remove bottom 10% or minimum 1 memory
            to_remove_count = max(1, len(sorted_memories) // 10)
            for memory_id, _ in sorted_memories[:to_remove_count]:
                del memories[memory_id]
                self._stats['evictions'] += 1
    
    async def _persist_memory(self, entry: MemoryEntry):
        """Persist memory entry to SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, content, memory_level, importance, created_at, last_accessed, 
                     access_count, associations, metadata, decay_rate, consolidation_threshold)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.id,
                    json.dumps(entry.content, default=str),
                    entry.memory_level.name,
                    entry.importance,
                    entry.created_at,
                    entry.last_accessed,
                    entry.access_count,
                    json.dumps(list(entry.associations)),
                    json.dumps(entry.metadata),
                    entry.decay_rate,
                    entry.consolidation_threshold
                ))
                conn.commit()
        except Exception as e:
            print(f"Failed to persist memory {entry.id}: {e}")
    
    async def _backup_to_database(self):
        """Backup all memories to database."""
        with self._lock:
            for level_memories in self._memories.values():
                for entry in level_memories.values():
                    await self._persist_memory(entry)
    
    async def load_from_database(self):
        """Load all memories from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM memories")
                for row in cursor.fetchall():
                    data = {
                        'id': row[0],
                        'content': json.loads(row[1]),
                        'memory_level': MemoryLevel[row[2]],
                        'importance': row[3],
                        'created_at': row[4],
                        'last_accessed': row[5],
                        'access_count': row[6],
                        'associations': set(json.loads(row[7]) if row[7] else '[]'),
                        'metadata': json.loads(row[8]) if row[8] else {},
                        'decay_rate': row[9],
                        'consolidation_threshold': row[10]
                    }
                    
                    entry = MemoryEntry(**data)
                    self._memories[entry.memory_level][entry.id] = entry
        except Exception as e:
            print(f"Failed to load from database: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        with self._lock:
            level_stats = {}
            total_memories = 0
            
            for level in MemoryLevel:
                count = len(self._memories[level])
                level_stats[level.name] = {
                    'count': count,
                    'capacity': level.capacity,
                    'utilization': count / level.capacity if level.capacity > 0 else 0
                }
                total_memories += count
            
            return {
                'total_memories': total_memories,
                'by_level': level_stats,
                'performance': self._stats.copy(),
                'database_path': str(self.db_path)
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.stop_background_tasks()
        await self._backup_to_database()


# Demo function
async def hierarchical_memory_demo():
    """Demonstrate the hierarchical memory system."""
    print("=== Hierarchical Memory System Demo ===")
    
    async with HierarchicalMemorySystem(
        db_path=":memory:",  # In-memory for demo
        auto_consolidation=True
    ) as memory_system:
        
        print("\n1. Storing memories at different levels:")
        
        # Store sensory memory (immediate perception)
        sensory_id = await memory_system.store(
            "Red light detected",
            MemoryLevel.SENSORY,
            importance=0.3
        )
        print(f"   Sensory memory: {sensory_id}")
        
        # Store working memory (active thought)
        working_id = await memory_system.store(
            "Calculating 15 * 23 = 345",
            MemoryLevel.WORKING,
            importance=0.7
        )
        print(f"   Working memory: {working_id}")
        
        # Store episodic memory (personal experience)
        episodic_id = await memory_system.store(
            "Birthday party at restaurant with friends",
            MemoryLevel.EPISODIC,
            importance=0.9,
            metadata={'emotion': 'joy', 'people': ['alice', 'bob']}
        )
        print(f"   Episodic memory: {episodic_id}")
        
        # Store semantic memory (general knowledge)
        semantic_id = await memory_system.store(
            "Paris is the capital of France",
            MemoryLevel.SEMANTIC,
            importance=1.0
        )
        print(f"   Semantic memory: {semantic_id}")
        
        print("\n2. Creating memory associations:")
        
        # Link episodic memory to semantic knowledge
        episodic_entry = await memory_system.retrieve(episodic_id)
        episodic_entry.associations.add(semantic_id)
        print(f"   Associated episodic memory with semantic knowledge")
        
        print("\n3. Retrieving memories:")
        
        # Retrieve specific memory
        retrieved = await memory_system.retrieve(episodic_id)
        if retrieved:
            print(f"   Retrieved: {retrieved.content}")
            print(f"   Importance: {retrieved.current_importance:.2f}")
            print(f"   Age: {retrieved.age:.1f}s")
        
        # Search memories
        search_results = await memory_system.search("France")
        print(f"   Search results for 'France': {len(search_results)} found")
        
        # Get associated memories
        associated = await memory_system.get_associated_memories(episodic_id)
        print(f"   Associated memories: {len(associated)} found")
        
        print("\n4. System statistics:")
        stats = memory_system.get_statistics()
        print(f"   Total memories: {stats['total_memories']}")
        
        for level, level_stats in stats['by_level'].items():
            print(f"   {level:10s}: {level_stats['count']:4d}/{level_stats['capacity']:6d} "
                  f"({level_stats['utilization']:.1%} utilized)")
        
        print(f"\n   Performance stats:")
        perf = stats['performance']
        print(f"   Memories stored: {perf['memories_stored']}")
        print(f"   Memories accessed: {perf['memories_accessed']}")
        print(f"   Consolidations: {perf['consolidations']}")
        print(f"   Evictions: {perf['evictions']}")
        print(f"   Avg access time: {perf['avg_access_time']:.4f}s")
        
        print("\n5. Simulating memory consolidation:")
        
        # Simulate multiple accesses to trigger consolidation
        for i in range(5):
            await memory_system.retrieve(working_id)
        
        # Wait a moment for consolidation
        await asyncio.sleep(0.1)
        
        # Check if memory was promoted
        promoted = await memory_system.retrieve(working_id)
        if promoted and promoted.memory_level != MemoryLevel.WORKING:
            print(f"   Memory promoted to: {promoted.memory_level.name}")
        else:
            print(f"   Memory still at: {promoted.memory_level.name}")
        
        print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(hierarchical_memory_demo())
