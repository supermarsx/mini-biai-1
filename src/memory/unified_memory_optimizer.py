"""
Unified Memory Optimization System
Integrates all memory optimization components into a cohesive high-performance system.

This system combines:
- Advanced cache hierarchy
- Intelligent prefetching strategies  
- Memory pooling
- Optimized FAISS indexing
- Smart embedding caching
- Memory compression
- Efficient attention mechanisms

Target: <20ms retrieval on 1M+ entries with minimal memory footprint
"""

import time
import threading
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import RLock
import asyncio
import gc

# Import all optimization components
from .advanced_cache_hierarchy import AdvancedCacheHierarchy
from .prefetching_strategies import IntelligentPrefetcher
from .memory_pooling import get_memory_pool_manager, MemoryPoolManager
from .optimized_faiss import OptimizedFAISSIndex, IndexConfig, create_optimal_index
from .embedding_cache import SmartEmbeddingCache, get_embedding_cache
from .compression_engine import UnifiedCompressionEngine, get_compression_engine
from .efficient_attention import MemoryEfficientAttention, compare_attention_mechanisms

logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """Configuration for the unified memory optimization system"""
    # Cache hierarchy settings
    l1_cache_size: int = 1000
    l2_cache_size: int = 10000
    l3_cache_size: int = 100000
    enable_disk_cache: bool = True
    
    # Prefetching settings
    enable_prefetching: bool = True
    max_prefetch_items: int = 1000
    prefetch_confidence_threshold: float = 0.3
    
    # Memory pooling settings
    enable_memory_pooling: bool = True
    pool_manager_config: Dict[str, Any] = field(default_factory=dict)
    
    # FAISS settings
    faiss_config: Dict[str, Any] = field(default_factory=lambda: {
        'index_type': 'IVF_PQ',
        'nlist': 1000,
        'm': 16,
        'nbits': 8,
        'use_gpu': False
    })
    
    # Embedding cache settings
    embedding_cache_size: int = 100000
    enable_semantic_caching: bool = True
    enable_predictive_caching: bool = True
    
    # Compression settings
    enable_compression: bool = True
    compression_level: int = 6
    enable_vector_quantization: bool = True
    
    # Attention settings
    enable_efficient_attention: bool = True
    attention_memory_limit_mb: float = 100.0
    
    # Performance targets
    target_latency_ms: float = 20.0
    target_memory_mb: float = 1000.0
    target_throughput_qps: float = 100.0

@dataclass
class PerformanceMetrics:
    """Performance metrics for the memory optimization system"""
    timestamp: float
    operation: str
    latency_ms: float
    memory_usage_mb: float
    throughput_qps: float
    cache_hit_rate: float
    compression_ratio: float
    attention_efficiency: float
    
    @property
    def meets_targets(self) -> bool:
        """Check if metrics meet performance targets"""
        return (self.latency_ms <= 20.0 and 
                self.memory_usage_mb <= 1000.0 and
                self.cache_hit_rate >= 0.8)

class UnifiedMemoryOptimizer:
    """
    Unified Memory Optimization System
    
    This system integrates multiple memory optimization techniques to achieve:
    - Sub-20ms retrieval latency on 1M+ entries
    - Minimal memory footprint through compression and pooling
    - High throughput with intelligent prefetching
    - Adaptive performance optimization
    
    Architecture:
    1. Multi-level cache hierarchy for fast access
    2. Intelligent prefetching based on access patterns
    3. Memory pooling for efficient allocation
    4. Optimized FAISS indexing for vector search
    5. Smart embedding caching with semantic similarity
    6. Advanced compression for memory efficiency
    7. Memory-efficient attention mechanisms
    
    Features:
    - Auto-tuning based on workload patterns
    - Real-time performance monitoring
    - Graceful degradation under memory pressure
    - Thread-safe operations
    - Persistent configuration
    """
    
    def __init__(self, config: SystemConfig = None):
        """Initialize the unified memory optimization system"""
        
        # Configuration
        self.config = config or SystemConfig()
        
        # Core components
        self.cache_hierarchy = None
        self.prefetcher = None
        self.pool_manager = None
        self.faiss_index = None
        self.embedding_cache = None
        self.compression_engine = None
        self.attention_engine = None
        
        # System state
        self.is_initialized = False
        self.startup_time = time.time()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.system_stats = {
            'total_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'prefetches_triggered': 0,
            'compressions_performed': 0,
            'memory_optimizations': 0,
            'avg_latency_ms': 0.0,
            'peak_memory_mb': 0.0
        }
        
        # Monitoring and optimization
        self.monitoring_enabled = True
        self.auto_tuning_enabled = True
        self.adaptation_thread = None
        self.monitoring_thread = None
        
        # Thread safety
        self._lock = RLock()
        
        # Initialize all components
        self._initialize_components()
        
        # Start background services
        self._start_background_services()
        
        logger.info(f"Unified Memory Optimizer initialized with {len(self.config.__dict__)} configuration parameters")
    
    def _initialize_components(self):
        """Initialize all optimization components"""
        try:
            # 1. Advanced Cache Hierarchy
            logger.info("Initializing cache hierarchy...")
            self.cache_hierarchy = AdvancedCacheHierarchy(
                l1_size=self.config.l1_cache_size,
                l2_size=self.config.l2_cache_size,
                l3_size=self.config.l3_cache_size,
                enable_disk_cache=self.config.enable_disk_cache
            )
            
            # 2. Intelligent Prefetcher
            if self.config.enable_prefetching:
                logger.info("Initializing prefetcher...")
                self.prefetcher = IntelligentPrefetcher(
                    max_prefetch_items=self.config.max_prefetch_items,
                    confidence_threshold=self.config.prefetch_confidence_threshold
                )
            
            # 3. Memory Pool Manager
            if self.config.enable_memory_pooling:
                logger.info("Initializing memory pool manager...")
                self.pool_manager = get_memory_pool_manager()
            
            # 4. Optimized FAISS Index
            logger.info("Initializing FAISS index...")
            faiss_config = IndexConfig(**self.config.faiss_config)
            self.faiss_index = OptimizedFAISSIndex(
                config=faiss_config,
                enable_monitoring=True
            )
            
            # 5. Smart Embedding Cache
            logger.info("Initializing embedding cache...")
            self.embedding_cache = SmartEmbeddingCache(
                max_cache_size=self.config.embedding_cache_size,
                enable_semantic_caching=self.config.enable_semantic_caching,
                enable_predictive_caching=self.config.enable_predictive_caching
            )
            
            # 6. Compression Engine
            if self.config.enable_compression:
                logger.info("Initializing compression engine...")
                self.compression_engine = get_compression_engine()
                self.compression_engine.default_compression_level = self.config.compression_level
                self.compression_engine.enable_vector_quantization = self.config.enable_vector_quantization
            
            # 7. Memory-Efficient Attention
            if self.config.enable_efficient_attention:
                logger.info("Initializing attention engine...")
                self.attention_engine = MemoryEfficientAttention(
                    hidden_dim=512,
                    num_heads=8,
                    max_memory_mb=self.config.attention_memory_limit_mb
                )
            
            self.is_initialized = True
            logger.info("All memory optimization components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory optimization components: {e}")
            raise
    
    def _start_background_services(self):
        """Start background monitoring and optimization services"""
        
        # Performance monitoring thread
        def monitoring_worker():
            while self.monitoring_enabled:
                try:
                    self._monitor_performance()
                    time.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring worker error: {e}")
        
        # Auto-tuning thread
        def adaptation_worker():
            while self.auto_tuning_enabled:
                try:
                    self._adapt_to_workload()
                    time.sleep(60)  # Adapt every minute
                except Exception as e:
                    logger.error(f"Adaptation worker error: {e}")
        
        # Start threads
        self.monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()
        
        self.adaptation_thread = threading.Thread(target=adaptation_worker, daemon=True)
        self.adaptation_thread.start()
        
        logger.info("Background services started: monitoring and auto-tuning")
    
    def search_memory(self, query: str, k: int = 10, use_prefetching: bool = True,
                     return_stats: bool = False) -> Tuple[List[Dict], Optional[Dict]]:
        """
        Perform optimized memory search using all optimization components.
        
        Args:
            query: Search query
            k: Number of results to return
            use_prefetching: Whether to use prefetching
            return_stats: Whether to return performance statistics
            
        Returns:
            Tuple of (search_results, performance_stats)
        """
        start_time = time.time()
        
        try:
            with self._lock:
                self.system_stats['total_operations'] += 1
                
                # 1. Check prefetched results first
                if use_prefetching and self.prefetcher:
                    prefetched_result = self.prefetcher.get_prefetched(query)
                    if prefetched_result is not None:
                        self.system_stats['prefetches_triggered'] += 1
                        logger.debug(f"Used prefetched result for query: {query[:50]}...")
                        
                        if return_stats:
                            stats = self._collect_performance_stats(start_time, "prefetch_hit")
                            return prefetched_result.get('results', []), stats
                        return prefetched_result.get('results', []), None
                
                # 2. Check cache hierarchy for embeddings and results
                cache_result = self._search_cache_hierarchy(query, k)
                if cache_result is not None:
                    self.system_stats['cache_hits'] += 1
                    
                    if return_stats:
                        stats = self._collect_performance_stats(start_time, "cache_hit")
                        return cache_result, stats
                    return cache_result, None
                
                # 3. Check embedding cache
                embedding_result = self._search_embedding_cache(query)
                if embedding_result is not None:
                    # Use cached embedding for search
                    query_embedding = embedding_result
                    search_results = self._perform_faiss_search(query_embedding, k)
                    
                    # Cache the results
                    self._cache_search_results(query, search_results)
                    
                    if return_stats:
                        stats = self._collect_performance_stats(start_time, "embedding_cache_hit")
                        return search_results, stats
                    return search_results, None
                
                # 4. Perform full search pipeline
                search_results = self._perform_full_search(query, k)
                
                # Record cache miss
                self.system_stats['cache_misses'] += 1
                
                # Record performance statistics
                if return_stats:
                    stats = self._collect_performance_stats(start_time, "full_search")
                    return search_results, stats
                return search_results, None
                
        except Exception as e:
            logger.error(f"Memory search failed for query '{query}': {e}")
            if return_stats:
                stats = self._collect_performance_stats(start_time, "error")
                stats['error'] = str(e)
                return [], stats
            return [], None
    
    def add_memory(self, content: str, metadata: Dict[str, Any] = None, 
                   embedding: Optional[np.ndarray] = None) -> str:
        """
        Add new memory with all optimizations applied.
        
        Args:
            content: Memory content
            metadata: Associated metadata
            embedding: Pre-computed embedding (if None, will be computed)
            
        Returns:
            Memory ID
        """
        try:
            with self._lock:
                memory_id = f"mem_{int(time.time() * 1000000)}"
                
                # 1. Generate embedding if not provided
                if embedding is None:
                    embedding = self._generate_embedding(content)
                
                # 2. Add to FAISS index with compression
                vector_metadata = {
                    'id': memory_id,
                    'content': content,
                    'metadata': metadata or {},
                    'timestamp': time.time(),
                    'size_bytes': len(content.encode('utf-8'))
                }
                
                # Compress metadata if compression is enabled
                if self.compression_engine:
                    compressed_data = self.compression_engine.compress_embedding_with_metadata(
                        embedding, vector_metadata
                    )
                    self.system_stats['compressions_performed'] += 1
                
                # Add to FAISS index
                vector_ids = self.faiss_index.add_vectors(
                    embedding.reshape(1, -1), 
                    [vector_metadata]
                )
                
                # 3. Cache embedding for future use
                if self.embedding_cache:
                    self.embedding_cache.cache_embedding(
                        content, embedding, 
                        embedding_model="sentence-transformers",
                        embedding_dim=embedding.shape[0],
                        semantic_tags=self._extract_semantic_tags(content)
                    )
                
                # 4. Cache in hierarchy
                cache_key = f"memory_{memory_id}"
                self.cache_hierarchy.put(
                    cache_key, 
                    {
                        'id': memory_id,
                        'content': content,
                        'metadata': metadata,
                        'embedding_shape': embedding.shape
                    },
                    ttl=86400  # 24 hours
                )
                
                # 5. Record for prefetching patterns
                if self.prefetcher:
                    self.prefetcher.record_query(content)
                
                logger.debug(f"Added memory: {memory_id}")
                return memory_id
                
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise
    
    def _search_cache_hierarchy(self, query: str, k: int) -> Optional[List[Dict]]:
        """Search through cache hierarchy for results"""
        # Try exact match
        cache_key = f"search_{hash(query)}"
        cached_result = self.cache_hierarchy.get(cache_key)
        
        if cached_result is not None:
            return cached_result.get('results', [])[:k]
        
        return None
    
    def _search_embedding_cache(self, query: str) -> Optional[np.ndarray]:
        """Search embedding cache for query embedding"""
        if not self.embedding_cache:
            return None
        
        return self.embedding_cache.get_embedding(
            query, 
            embedding_model="sentence-transformers",
            embedding_dim=384  # Default sentence transformer dimension
        )
    
    def _perform_faiss_search(self, query_embedding: np.ndarray, k: int) -> List[Dict]:
        """Perform FAISS search and format results"""
        try:
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                k=k
            )
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid index
                    memory_metadata = self.faiss_index.get_vector_by_id(idx)
                    if memory_metadata:
                        results.append({
                            'id': memory_metadata.get('id'),
                            'content': memory_metadata.get('content', ''),
                            'metadata': memory_metadata.get('metadata', {}),
                            'score': float(distance),
                            'rank': i + 1
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _perform_full_search(self, query: str, k: int) -> List[Dict]:
        """Perform full search pipeline"""
        # 1. Generate or retrieve embedding
        embedding = self._search_embedding_cache(query)
        if embedding is None:
            embedding = self._generate_embedding(query)
            # Cache for future use
            self.embedding_cache.cache_embedding(
                query, embedding,
                embedding_model="sentence-transformers",
                embedding_dim=embedding.shape[0]
            )
        
        # 2. Perform FAISS search
        results = self._perform_faiss_search(embedding, k)
        
        # 3. Cache results for future hits
        self._cache_search_results(query, results)
        
        return results
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (placeholder implementation)"""
        # In real implementation, would use actual sentence transformer
        # For now, return a random vector
        return np.random.randn(384).astype(np.float32)
    
    def _extract_semantic_tags(self, content: str) -> List[str]:
        """Extract semantic tags from content (placeholder)"""
        # Simple keyword extraction
        words = content.lower().split()
        # Return most common words as tags (simplified)
        return words[:5]  # Top 5 words as tags
    
    def _cache_search_results(self, query: str, results: List[Dict]):
        """Cache search results for future fast access"""
        cache_key = f"search_{hash(query)}"
        cache_data = {
            'query': query,
            'results': results,
            'timestamp': time.time()
        }
        
        self.cache_hierarchy.put(cache_key, cache_data, ttl=1800)  # 30 minutes
    
    def _collect_performance_stats(self, start_time: float, operation: str) -> Dict[str, Any]:
        """Collect comprehensive performance statistics"""
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Collect metrics from all components
        stats = {
            'timestamp': end_time,
            'operation': operation,
            'latency_ms': latency_ms,
            'total_operations': self.system_stats['total_operations'],
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'memory_usage_mb': self._get_current_memory_usage(),
            'faiss_stats': self.faiss_index.get_performance_stats() if self.faiss_index else {},
            'cache_stats': self.cache_hierarchy.get_stats() if self.cache_hierarchy else {},
            'embedding_cache_stats': self.embedding_cache.get_cache_stats() if self.embedding_cache else {},
            'compression_stats': self.compression_engine.get_global_stats() if self.compression_engine else {},
            'meets_targets': latency_ms <= self.config.target_latency_ms
        }
        
        # Store in performance history
        metrics = PerformanceMetrics(
            timestamp=end_time,
            operation=operation,
            latency_ms=latency_ms,
            memory_usage_mb=stats['memory_usage_mb'],
            throughput_qps=1.0 / max(latency_ms / 1000, 0.001),  # Simple throughput calc
            cache_hit_rate=stats['cache_hit_rate'],
            compression_ratio=0.7,  # Estimated
            attention_efficiency=0.8  # Estimated
        )
        
        self.performance_history.append(metrics)
        
        return stats
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        total_requests = self.system_stats['cache_hits'] + self.system_stats['cache_misses']
        return self.system_stats['cache_hits'] / max(total_requests, 1)
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage across all components"""
        total_memory = 0.0
        
        # Add memory from each component
        if self.cache_hierarchy:
            cache_stats = self.cache_hierarchy.get_stats()
            total_memory += cache_stats['total_memory_bytes'] / (1024 * 1024)
        
        if self.faiss_index:
            faiss_stats = self.faiss_index.get_performance_stats()
            total_memory += faiss_stats['memory']['current_usage_mb']
        
        if self.embedding_cache:
            embedding_stats = self.embedding_cache.get_cache_stats()
            total_memory += embedding_stats['memory_usage_mb']
        
        # Update peak memory
        self.system_stats['peak_memory_mb'] = max(
            self.system_stats['peak_memory_mb'], 
            total_memory
        )
        
        return total_memory
    
    def _monitor_performance(self):
        """Monitor system performance and log metrics"""
        if not self.performance_history:
            return
        
        # Calculate recent metrics
        recent_metrics = list(self.performance_history)[-10:]  # Last 10 operations
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        avg_cache_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])
        
        # Log performance summary
        logger.info(f"Performance Summary - Avg Latency: {avg_latency:.2f}ms, "
                   f"Memory: {avg_memory:.1f}MB, Cache Hit Rate: {avg_cache_hit_rate:.1%}")
        
        # Check if auto-tuning is needed
        if self.auto_tuning_enabled:
            if avg_latency > self.config.target_latency_ms * 1.2:
                self._trigger_latency_optimization()
            if avg_memory > self.config.target_memory_mb * 0.9:
                self._trigger_memory_optimization()
    
    def _adapt_to_workload(self):
        """Adapt system configuration based on workload patterns"""
        if not self.performance_history:
            return
        
        # Analyze recent performance patterns
        recent_metrics = list(self.performance_history)[-60:]  # Last hour
        
        # Calculate workload characteristics
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        operations_count = len(recent_metrics)
        
        # Create workload stats for component adaptation
        workload_stats = {
            'average_latency_ms': avg_latency,
            'average_memory_usage_mb': avg_memory,
            'operations_per_minute': operations_count,
            'target_latency_ms': self.config.target_latency_ms,
            'target_memory_mb': self.config.target_memory_mb
        }
        
        # Adapt components
        if self.attention_engine:
            self.attention_engine.adapt_to_workload(workload_stats)
        
        if self.compression_engine:
            self.compression_engine.optimize_for_workload(workload_stats)
        
        self.system_stats['memory_optimizations'] += 1
    
    def _trigger_latency_optimization(self):
        """Trigger optimizations to improve latency"""
        logger.info("Triggering latency optimization...")
        
        # Increase cache sizes
        if self.cache_hierarchy:
            # Increase L1 and L2 cache sizes by 20%
            self.cache_hierarchy.l1_max_size = int(self.cache_hierarchy.l1_max_size * 1.2)
            self.cache_hierarchy.l2_max_size = int(self.cache_hierarchy.l2_max_size * 1.2)
        
        # Optimize FAISS parameters
        if self.faiss_index:
            self.faiss_index.auto_tuning_enabled = True
            self.faiss_index.optimize_index()
        
        # Enable more aggressive prefetching
        if self.prefetcher:
            self.prefetcher.confidence_threshold *= 0.9  # Lower threshold = more prefetching
    
    def _trigger_memory_optimization(self):
        """Trigger optimizations to reduce memory usage"""
        logger.info("Triggering memory optimization...")
        
        # Increase compression levels
        if self.compression_engine:
            self.compression_engine.default_compression_level = min(9, 
                self.compression_engine.default_compression_level + 1)
        
        # Reduce cache sizes
        if self.cache_hierarchy:
            # Reduce L3 cache size
            self.cache_hierarchy.l3_max_size = int(self.cache_hierarchy.l3_max_size * 0.8)
        
        # Force garbage collection
        gc.collect()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        with self._lock:
            # Base system stats
            stats = {
                'system_info': {
                    'initialization_time': self.startup_time,
                    'uptime_seconds': time.time() - self.startup_time,
                    'is_initialized': self.is_initialized,
                    'configuration': self.config.__dict__
                },
                'performance_summary': {
                    'total_operations': self.system_stats['total_operations'],
                    'cache_hit_rate': self._calculate_cache_hit_rate(),
                    'average_latency_ms': self._calculate_average_latency(),
                    'peak_memory_mb': self.system_stats['peak_memory_mb'],
                    'meets_target_latency': self._meets_latency_target()
                },
                'component_stats': {}
            }
            
            # Add component-specific stats
            if self.cache_hierarchy:
                stats['component_stats']['cache_hierarchy'] = self.cache_hierarchy.get_stats()
            
            if self.prefetcher:
                stats['component_stats']['prefetcher'] = self.prefetcher.get_stats()
            
            if self.faiss_index:
                stats['component_stats']['faiss_index'] = self.faiss_index.get_performance_stats()
            
            if self.embedding_cache:
                stats['component_stats']['embedding_cache'] = self.embedding_cache.get_cache_stats()
            
            if self.compression_engine:
                stats['component_stats']['compression_engine'] = self.compression_engine.get_global_stats()
            
            if self.attention_engine:
                stats['component_stats']['attention_engine'] = self.attention_engine.get_stats()
            
            return stats
    
    def _calculate_average_latency(self) -> float:
        """Calculate average latency from performance history"""
        if not self.performance_history:
            return 0.0
        return np.mean([m.latency_ms for m in self.performance_history])
    
    def _meets_latency_target(self) -> bool:
        """Check if system meets latency target"""
        avg_latency = self._calculate_average_latency()
        return avg_latency <= self.config.target_latency_ms
    
    def optimize_for_workload(self, workload_profile: Dict[str, Any]):
        """
        Optimize the system for a specific workload profile.
        
        Args:
            workload_profile: Dictionary describing workload characteristics
        """
        logger.info(f"Optimizing system for workload: {workload_profile}")
        
        # Extract workload characteristics
        avg_sequence_length = workload_profile.get('average_sequence_length', 512)
        data_size_mb = workload_profile.get('data_size_mb', 100)
        query_frequency = workload_profile.get('queries_per_second', 10)
        accuracy_requirement = workload_profile.get('accuracy_requirement', 'balanced')
        
        # Adapt FAISS configuration
        if self.faiss_index:
            if data_size_mb > 1000:  # Large datasets
                # Use more aggressive compression
                self.faiss_index.config.m = min(32, self.faiss_index.config.m * 2)
                self.faiss_index.config.nbits = min(12, self.faiss_index.config.nbits + 2)
            
            if accuracy_requirement == 'high':
                # Increase nlist for better accuracy
                self.faiss_index.config.nlist = min(2000, self.faiss_index.config.nlist * 2)
            elif accuracy_requirement == 'fast':
                # Decrease nlist for speed
                self.faiss_index.config.nlist = max(100, self.faiss_index.config.nlist // 2)
        
        # Adapt cache hierarchy
        if self.cache_hierarchy:
            if query_frequency > 50:  # High frequency queries
                # Increase L1 cache size
                self.cache_hierarchy.l1_max_size = min(5000, self.cache_hierarchy.l1_max_size * 2)
            
            if avg_sequence_length > 1000:  # Long sequences
                # Increase all cache sizes
                self.cache_hierarchy.l1_max_size = int(self.cache_hierarchy.l1_max_size * 1.5)
                self.cache_hierarchy.l2_max_size = int(self.cache_hierarchy.l2_max_size * 1.5)
        
        # Adapt embedding cache
        if self.embedding_cache:
            if data_size_mb > 500:  # Large dataset
                # Increase cache size
                self.embedding_cache.max_cache_size = min(500000, self.embedding_cache.max_cache_size * 2)
        
        # Adapt compression engine
        if self.compression_engine:
            if data_size_mb > 1000:  # Very large data
                # Enable more aggressive compression
                self.compression_engine.default_compression_level = 9
            
            if query_frequency > 100:  # High frequency
                # Optimize for speed over compression
                self.compression_engine.default_compression_level = 3
        
        logger.info("System optimization for workload completed")
    
    def benchmark_performance(self, test_queries: List[str], iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark system performance with test queries.
        
        Args:
            test_queries: List of test queries
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        logger.info(f"Running performance benchmark with {len(test_queries)} queries, {iterations} iterations")
        
        latencies = []
        memory_usages = []
        cache_hit_rates = []
        
        for iteration in range(iterations):
            for query in test_queries:
                start_time = time.time()
                
                # Perform search
                results, _ = self.search_memory(query, k=10, return_stats=True)
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                
                # Record memory usage
                memory_mb = self._get_current_memory_usage()
                memory_usages.append(memory_mb)
                
                # Record cache hit rate
                cache_hit_rates.append(self._calculate_cache_hit_rate())
        
        # Calculate statistics
        results = {
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'avg_memory_mb': np.mean(memory_usages),
            'max_memory_mb': np.max(memory_usages),
            'avg_cache_hit_rate': np.mean(cache_hit_rates),
            'total_operations': len(latencies),
            'meets_20ms_target': np.mean(latencies) <= 20.0,
            'meets_80_hit_rate': np.mean(cache_hit_rates) >= 0.8
        }
        
        logger.info(f"Benchmark completed: avg latency = {results['avg_latency_ms']:.2f}ms, "
                   f"cache hit rate = {results['avg_cache_hit_rate']:.1%}")
        
        return results
    
    def shutdown(self):
        """Graceful shutdown of the memory optimization system"""
        logger.info("Shutting down Unified Memory Optimizer...")
        
        # Stop background services
        self.monitoring_enabled = False
        self.auto_tuning_enabled = False
        
        # Shutdown components
        if self.cache_hierarchy:
            self.cache_hierarchy.shutdown()
        
        if self.prefetcher:
            self.prefetcher.shutdown()
        
        if self.faiss_index:
            self.faiss_index.shutdown()
        
        if self.embedding_cache:
            self.embedding_cache.shutdown()
        
        if self.attention_engine:
            # Attention engine doesn't have explicit shutdown
            pass
        
        # Wait for background threads
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.adaptation_thread and self.adaptation_thread.is_alive():
            self.adaptation_thread.join(timeout=5)
        
        logger.info("Unified Memory Optimizer shutdown complete")

# Global instance for convenient access
_global_optimizer = None

def get_memory_optimizer(config: SystemConfig = None) -> UnifiedMemoryOptimizer:
    """
    Get global unified memory optimizer instance.
    
    Args:
        config: Optional system configuration
        
    Returns:
        UnifiedMemoryOptimizer instance
    """
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = UnifiedMemoryOptimizer(config)
    return _global_optimizer

def create_optimized_system(target_scale: str = "1M_entries") -> UnifiedMemoryOptimizer:
    """
    Create optimized memory system for specific scale.
    
    Args:
        target_scale: Target scale ("1M_entries", "10M_entries", "100M_entries")
        
    Returns:
        Configured UnifiedMemoryOptimizer instance
    """
    
    # Scale-specific configurations
    scale_configs = {
        "1M_entries": SystemConfig(
            l1_cache_size=2000,
            l2_cache_size=20000,
            l3_cache_size=200000,
            max_prefetch_items=2000,
            embedding_cache_size=200000,
            faiss_config={
                'index_type': 'IVF_PQ',
                'nlist': 1000,
                'm': 16,
                'nbits': 8,
                'use_gpu': False
            }
        ),
        "10M_entries": SystemConfig(
            l1_cache_size=5000,
            l2_cache_size=50000,
            l3_cache_size=500000,
            max_prefetch_items=5000,
            embedding_cache_size=500000,
            faiss_config={
                'index_type': 'IVF_PQ',
                'nlist': 2000,
                'm': 32,
                'nbits': 8,
                'use_gpu': True
            }
        ),
        "100M_entries": SystemConfig(
            l1_cache_size=10000,
            l2_cache_size=100000,
            l3_cache_size=1000000,
            max_prefetch_items=10000,
            embedding_cache_size=1000000,
            faiss_config={
                'index_type': 'IVF_PQ',
                'nlist': 5000,
                'm': 64,
                'nbits': 8,
                'use_gpu': True
            },
            target_memory_mb=5000.0
        )
    }
    
    config = scale_configs.get(target_scale, scale_configs["1M_entries"])
    
    logger.info(f"Creating optimized memory system for {target_scale}")
    
    return UnifiedMemoryOptimizer(config)
