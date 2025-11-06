import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
import heapq
import random

logger = logging.getLogger(__name__)

@dataclass
class PrefetchQuery:
    """Represents a query for prefetching with confidence score"""
    query: str
    timestamp: float
    frequency: int
    last_access: float
    related_queries: Set[str] = field(default_factory=set)
    confidence: float = 0.0
    
    def update_frequency(self):
        """Update frequency and recalculate confidence"""
        self.frequency += 1
        now = time.time()
        time_since_last = now - self.last_access
        
        # Calculate confidence based on frequency and recency
        freq_score = min(self.frequency / 10.0, 1.0)  # Normalize frequency
        recency_score = max(0, 1.0 - time_since_last / 3600)  # Decay over 1 hour
        
        self.confidence = (freq_score * 0.6) + (recency_score * 0.4)
        self.last_access = now

@dataclass 
class PrefetchCandidate:
    """Candidate item for prefetching with priority score"""
    key: str
    data: Any
    priority: float
    estimated_load_time: float
    memory_size: int
    source: str
    
    def __lt__(self, other):
        return self.priority > other.priority  # Max-heap behavior

class IntelligentPrefetcher:
    """
    Intelligent prefetching system that predicts and preloads data based on:
    
    1. **Access Pattern Analysis**: Historical query patterns and frequency
    2. **Sequential Access Detection**: Detect sequential patterns in data access
    3. **Semantic Similarity**: Use embeddings to find related content
    4. **Time-based Prefetching**: Preload based on time patterns
    5. **Collaborative Filtering**: Learn from user behavior patterns
    
    Features:
    - Multi-strategy prefetching (frequency, sequential, semantic, temporal)
    - Adaptive confidence scoring
    - Memory-aware prefetching (evict low-priority prefetches)
    - Background prefetching with limited concurrency
    - Prefetch hit/miss tracking and optimization
    """
    
    def __init__(self, 
                 max_prefetch_items: int = 1000,
                 max_concurrent_prefetches: int = 5,
                 confidence_threshold: float = 0.3,
                 prefetch_batch_size: int = 10,
                 semantic_threshold: float = 0.8,
                 enable_semantic_prefetch: bool = True,
                 enable_sequential_prefetch: bool = True,
                 enable_temporal_prefetch: bool = True):
        
        # Configuration
        self.max_prefetch_items = max_prefetch_items
        self.max_concurrent_prefetches = max_concurrent_prefetches
        self.confidence_threshold = confidence_threshold
        self.prefetch_batch_size = prefetch_batch_size
        self.semantic_threshold = semantic_threshold
        
        # Feature flags
        self.enable_semantic_prefetch = enable_semantic_prefetch
        self.enable_sequential_prefetch = enable_sequential_prefetch  
        self.enable_temporal_prefetch = enable_temporal_prefetch
        
        # Query tracking and analysis
        self.query_history = {}  # query -> PrefetchQuery
        self.recent_queries = deque(maxlen=1000)  # Recent query sequence
        self.access_patterns = defaultdict(list)  # user_pattern -> query_sequence
        self.sequential_patterns = defaultdict(deque)  # query -> next queries
        
        # Prefetch queues and caches
        self.prefetch_queue = []  # Priority queue of PrefetchCandidate
        self.prefetched_items = {}  # key -> prefetched data
        self.prefetch_stats = {
            'total_prefetches': 0,
            'prefetches_used': 0,
            'prefetches_evicted': 0,
            'prefetches_by_strategy': defaultdict(int)
        }
        
        # Semantic similarity tracking (placeholder for embeddings)
        self.query_embeddings = {}  # query -> embedding vector
        self.similarity_cache = {}  # query_pair -> similarity_score
        
        # Background prefetching
        self.prefetch_thread = None
        self.prefetch_running = True
        self.prefetch_semaphore = threading.Semaphore(max_concurrent_prefetches)
        
        # Locks for thread safety
        self._locks = {
            'queries': RLock(),
            'prefetch': RLock(),
            'patterns': RLock(),
            'stats': RLock()
        }
        
        # Start background prefetching
        self._start_background_prefetching()
        
        logger.info(f"Intelligent prefetcher initialized: max_items={max_prefetch_items}, strategies=[semantic={enable_semantic_prefetch}, sequential={enable_sequential_prefetch}, temporal={enable_temporal_prefetch}]")
    
    def record_query(self, query: str, user_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Record a query for pattern analysis and prefetching"""
        with self._locks['queries']:
            current_time = time.time()
            
            # Update or create query record
            if query not in self.query_history:
                self.query_history[query] = PrefetchQuery(
                    query=query,
                    timestamp=current_time,
                    frequency=1,
                    last_access=current_time
                )
            else:
                self.query_history[query].update_frequency()
            
            # Add to recent queries
            self.recent_queries.append(query)
            
            # Update sequential patterns
            if self.enable_sequential_prefetch and len(self.recent_queries) >= 2:
                self._update_sequential_patterns()
            
            # Record access pattern for user
            if user_id:
                with self._locks['patterns']:
                    if len(self.recent_queries) >= 5:
                        pattern = tuple(list(self.recent_queries)[-5:])  # Last 5 queries
                        self.access_patterns[user_id].append(pattern)
            
            # Analyze and trigger prefetching
            self._analyze_and_prefetch(query, metadata)
    
    def get_prefetched(self, query: str) -> Optional[Any]:
        """Get prefetched data for a query"""
        with self._locks['prefetch']:
            # Direct key match
            if query in self.prefetched_items:
                self._stats_counter('prefetches_used')
                return self.prefetched_items[query]
            
            # Semantic similarity match
            if self.enable_semantic_prefetch:
                return self._get_semantic_match(query)
            
            return None
    
    def _get_semantic_match(self, query: str) -> Optional[Any]:
        """Find prefetched items using semantic similarity"""
        if not self.enable_semantic_prefetch:
            return None
            
        query_key = self._normalize_query(query)
        best_match = None
        best_score = 0.0
        
        for prefetch_key in self.prefetched_items.keys():
            similarity = self._calculate_query_similarity(query_key, prefetch_key)
            if similarity > best_score and similarity >= self.semantic_threshold:
                best_score = similarity
                best_match = self.prefetched_items[prefetch_key]
        
        if best_match:
            self._stats_counter('prefetches_used')
            return best_match
        
        return None
    
    def _analyze_and_prefetch(self, current_query: str, metadata: Optional[Dict] = None):
        """Analyze patterns and trigger prefetching"""
        candidates = []
        
        # Strategy 1: Frequency-based prefetching
        if self.enable_temporal_prefetch:
            freq_candidates = self._frequency_based_prefetch(current_query)
            candidates.extend(freq_candidates)
        
        # Strategy 2: Sequential pattern prefetching
        if self.enable_sequential_prefetch:
            seq_candidates = self._sequential_prefetch(current_query)
            candidates.extend(seq_candidates)
        
        # Strategy 3: Semantic similarity prefetching
        if self.enable_semantic_prefetch:
            semantic_candidates = self._semantic_prefetch(current_query)
            candidates.extend(semantic_candidates)
        
        # Strategy 4: Temporal pattern prefetching
        if self.enable_temporal_prefetch:
            temporal_candidates = self._temporal_prefetch(current_query)
            candidates.extend(temporal_candidates)
        
        # Sort by priority and trigger prefetching
        if candidates:
            self._trigger_prefetching(candidates)
    
    def _frequency_based_prefetch(self, query: str) -> List[PrefetchCandidate]:
        """Generate prefetch candidates based on query frequency"""
        candidates = []
        
        with self._locks['queries']:
            # Find frequently co-occurring queries
            for hist_query, pq in self.query_history.items():
                if hist_query != query and pq.confidence >= self.confidence_threshold:
                    # Calculate priority based on frequency and confidence
                    priority = pq.confidence * pq.frequency
                    
                    candidate = PrefetchCandidate(
                        key=hist_query,
                        data=None,  # Will be loaded by background thread
                        priority=priority,
                        estimated_load_time=0.1,  # Assume 100ms for data loading
                        memory_size=1024,  # Assume 1KB per query result
                        source="frequency"
                    )
                    candidates.append(candidate)
        
        return candidates[:self.prefetch_batch_size]
    
    def _sequential_prefetch(self, query: str) -> List[PrefetchCandidate]:
        """Generate prefetch candidates based on sequential patterns"""
        candidates = []
        
        if not self.enable_sequential_prefetch:
            return candidates
            
        with self._locks['patterns']:
            # Check if current query has known sequential followers
            if query in self.sequential_patterns:
                next_queries = self.sequential_patterns[query]
                
                for next_query in list(next_queries)[-5:]:  # Last 5 predicted next queries
                    if next_query in self.query_history:
                        pq = self.query_history[next_query]
                        # Higher priority for sequential patterns
                        priority = pq.confidence * pq.frequency * 1.5
                        
                        candidate = PrefetchCandidate(
                            key=next_query,
                            data=None,
                            priority=priority,
                            estimated_load_time=0.1,
                            memory_size=1024,
                            source="sequential"
                        )
                        candidates.append(candidate)
        
        return candidates
    
    def _semantic_prefetch(self, query: str) -> List[PrefetchCandidate]:
        """Generate prefetch candidates based on semantic similarity"""
        candidates = []
        
        if not self.enable_semantic_prefetch:
            return candidates
            
        query_key = self._normalize_query(query)
        
        with self._locks['queries']:
            for hist_query, pq in self.query_history.items():
                if hist_query != query:
                    similarity = self._calculate_query_similarity(query_key, hist_query)
                    if similarity >= self.semantic_threshold:
                        priority = similarity * pq.frequency * pq.confidence
                        
                        candidate = PrefetchCandidate(
                            key=hist_query,
                            data=None,
                            priority=priority,
                            estimated_load_time=0.2,  # Slightly slower due to embedding calc
                            memory_size=1024,
                            source="semantic"
                        )
                        candidates.append(candidate)
        
        return candidates[:self.prefetch_batch_size // 2]  # Limit semantic candidates
    
    def _temporal_prefetch(self, query: str) -> List[PrefetchCandidate]:
        """Generate prefetch candidates based on temporal patterns"""
        candidates = []
        
        if not self.enable_temporal_prefetch:
            return candidates
            
        current_hour = time.localtime().tm_hour
        current_day = time.localtime().tm_wday
        
        with self._locks['queries']:
            # Find queries that are typically accessed at this time
            for hist_query, pq in self.query_history.items():
                if hist_query != query:
                    # Simple temporal heuristic - can be enhanced
                    temporal_score = 0.5  # Base score
                    
                    # Adjust based on time patterns (placeholder logic)
                    if hasattr(pq, 'access_hours'):
                        # Would track typical access hours
                        if current_hour in pq.access_hours:
                            temporal_score += 0.3
                    
                    if temporal_score >= self.confidence_threshold:
                        priority = temporal_score * pq.frequency * pq.confidence
                        
                        candidate = PrefetchCandidate(
                            key=hist_query,
                            data=None,
                            priority=priority,
                            estimated_load_time=0.15,
                            memory_size=1024,
                            source="temporal"
                        )
                        candidates.append(candidate)
        
        return candidates[:self.prefetch_batch_size // 3]  # Limit temporal candidates
    
    def _update_sequential_patterns(self):
        """Update sequential access patterns from recent queries"""
        if len(self.recent_queries) < 2:
            return
            
        # Analyze recent query sequence for patterns
        queries_list = list(self.recent_queries)
        
        # Track query -> next query relationships
        for i in range(len(queries_list) - 1):
            current = queries_list[i]
            next_query = queries_list[i + 1]
            
            if current not in self.sequential_patterns:
                self.sequential_patterns[current] = deque(maxlen=20)
            
            self.sequential_patterns[current].append(next_query)
    
    def _trigger_prefetching(self, candidates: List[PrefetchCandidate]):
        """Trigger background prefetching for high-priority candidates"""
        # Sort by priority and limit
        candidates.sort(key=lambda x: x.priority, reverse=True)
        
        for candidate in candidates[:self.prefetch_batch_size]:
            with self._locks['prefetch']:
                # Check if already prefetched or in queue
                if candidate.key in self.prefetched_items:
                    continue
                
                # Add to priority queue
                heapq.heappush(self.prefetch_queue, candidate)
                
                # Start background prefetch if thread not running
                if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
                    self._start_background_prefetching()
    
    def _start_background_prefetching(self):
        """Start background thread for prefetching"""
        def prefetch_worker():
            while self.prefetch_running:
                try:
                    # Get next prefetch candidate
                    with self._locks['prefetch']:
                        if not self.prefetch_queue:
                            time.sleep(1)
                            continue
                        
                        candidate = heapq.heappop(self.prefetch_queue)
                    
                    # Load data with semaphore control
                    if self.prefetch_semaphore.acquire(timeout=0.1):
                        try:
                            self._load_prefetch_data(candidate)
                        finally:
                            self.prefetch_semaphore.release()
                    
                    # Check if we should stop
                    if not self.prefetch_running:
                        break
                        
                except Exception as e:
                    logger.error(f"Prefetch worker error: {e}")
                    time.sleep(1)
        
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
    
    def _load_prefetch_data(self, candidate: PrefetchCandidate):
        """Load data for a prefetch candidate (placeholder implementation)"""
        try:
            # This would integrate with actual data sources
            # For now, simulate data loading
            
            # Simulate data loading delay
            time.sleep(min(candidate.estimated_load_time, 0.5))
            
            # Generate mock data (in real implementation, this would load from actual sources)
            mock_data = {
                'query': candidate.key,
                'results': [f"Mock result {i} for {candidate.key}" for i in range(3)],
                'metadata': {
                    'prefetch_source': candidate.source,
                    'priority': candidate.priority,
                    'timestamp': time.time()
                }
            }
            
            # Store prefetched data
            with self._locks['prefetch']:
                # Check if we need to evict old prefetches
                if len(self.prefetched_items) >= self.max_prefetch_items:
                    self._evict_prefetch()
                
                self.prefetched_items[candidate.key] = mock_data
                self._stats_counter('total_prefetches')
                self._stats_counter('prefetches_by_strategy', candidate.source)
            
        except Exception as e:
            logger.error(f"Failed to prefetch data for {candidate.key}: {e}")
    
    def _evict_prefetch(self):
        """Evict lowest priority prefetched items"""
        if not self.prefetched_items:
            return
            
        # Remove oldest prefetched item
        oldest_key = min(self.prefetched_items.keys(), 
                        key=lambda k: self.prefetched_items[k]['metadata']['timestamp'])
        del self.prefetched_items[oldest_key]
        self._stats_counter('prefetches_evicted')
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate semantic similarity between queries (simplified implementation)"""
        # This is a placeholder - in real implementation would use embeddings
        # For now, use simple string similarity
        
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Jaccard similarity
        similarity = len(intersection) / len(union) if union else 0.0
        
        # Boost for exact phrase matches
        if query1.lower() in query2.lower() or query2.lower() in query1.lower():
            similarity += 0.3
        
        return min(similarity, 1.0)
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent processing"""
        return ' '.join(query.lower().strip().split())
    
    def _stats_counter(self, stat_type: str, value: str = None):
        """Thread-safe statistics counter"""
        with self._locks['stats']:
            if stat_type in self.prefetch_stats:
                if isinstance(self.prefetch_stats[stat_type], defaultdict):
                    self.prefetch_stats[stat_type][value] += 1
                else:
                    self.prefetch_stats[stat_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive prefetching statistics"""
        with self._locks['stats']:
            # Calculate prefetch hit rate
            total_attempts = self.prefetch_stats['total_prefetches']
            used_prefetches = self.prefetch_stats['prefetches_used']
            hit_rate = (used_prefetches / total_attempts * 100) if total_attempts > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_attempts': total_attempts,
                'prefetches_used': used_prefetches,
                'prefetches_evicted': self.prefetch_stats['prefetches_evicted'],
                'prefetches_by_strategy': dict(self.prefetch_stats['prefetches_by_strategy']),
                'active_prefetched_items': len(self.prefetched_items),
                'max_prefetch_items': self.max_prefetch_items,
                'patterns': {
                    'total_queries_tracked': len(self.query_history),
                    'sequential_patterns': len(self.sequential_patterns),
                    'recent_queries': len(self.recent_queries)
                }
            }
    
    def clear_prefetch_cache(self):
        """Clear all prefetched items"""
        with self._locks['prefetch']:
            self.prefetched_items.clear()
            self.prefetch_queue.clear()
            logger.info("Cleared prefetch cache")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.prefetch_running = False
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=5)
        logger.info("Prefetcher shutdown complete")