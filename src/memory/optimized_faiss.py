"""
Optimized FAISS Indexing System
Enhanced FAISS configurations for sub-20ms retrieval on 1M+ entries with memory optimization.
"""

import time
import threading
import logging
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from threading import RLock
import pickle

# FAISS imports
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

logger = logging.getLogger(__name__)

@dataclass
class IndexConfig:
    """Optimized configuration for FAISS index"""
    dimension: int
    index_type: str = "IVF_PQ"  # IVF_PQ, IVF_FLAT, HNSW, IVFPQ_HNSW
    nlist: int = 1000  # Number of IVF clusters
    m: int = 16        # PQ subvectors
    nbits: int = 8     # PQ bits per subvector
    hnsw_m: int = 32   # HNSW connections per node
    hnsw_ef_construction: int = 200  # HNSW construction parameter
    hnsw_ef_search: int = 64         # HNSW search parameter
    use_gpu: bool = False
    gpu_id: int = 0
    quantizer_bits: int = 8
    
    def get_memory_estimate(self, num_vectors: int) -> Dict[str, float]:
        """Estimate memory usage for given number of vectors"""
        if self.index_type == "IVF_PQ":
            # IVF-PQ memory: vectors + index + quantizer
            vector_memory = num_vectors * self.dimension * 4  # float32
            pq_memory = num_vectors * self.m * 1  # PQ codes (1 byte each)
            index_memory = self.nlist * self.dimension * 4  # centroids
            quantizer_memory = self.nlist * self.dimension * 4  # quantizer
            
            total_mb = (vector_memory + pq_memory + index_memory + quantizer_memory) / (1024 * 1024)
            compressed_ratio = vector_memory / (pq_memory + index_memory + quantizer_memory)
            
        elif self.index_type == "IVF_FLAT":
            vector_memory = num_vectors * self.dimension * 4
            index_memory = self.nlist * self.dimension * 4
            
            total_mb = (vector_memory + index_memory) / (1024 * 1024)
            compressed_ratio = 1.0
            
        elif self.index_type == "HNSW":
            # HNSW memory: vectors + connections graph
            vector_memory = num_vectors * self.dimension * 4
            graph_memory = num_vectors * self.hnsw_m * 4  # approximated
            
            total_mb = (vector_memory + graph_memory) / (1024 * 1024)
            compressed_ratio = vector_memory / vector_memory  # No compression
            
        else:
            total_mb = num_vectors * self.dimension * 4 / (1024 * 1024)
            compressed_ratio = 1.0
        
        return {
            'total_memory_mb': total_mb,
            'compressed_ratio': compressed_ratio,
            'memory_per_vector_kb': (total_mb * 1024) / num_vectors if num_vectors > 0 else 0
        }

class OptimizedFAISSIndex:
    """
    Highly optimized FAISS index with multiple index types and smart configurations.
    
    Optimizations:
    - Multi-tier indexing (coarse + fine)
    - Adaptive parameter tuning based on data size
    - GPU acceleration support
    - Memory-efficient quantization
    - Batch processing optimization
    - Performance monitoring and auto-tuning
    
    Target: <20ms retrieval on 1M+ entries with minimal memory usage
    """
    
    def __init__(self, config: IndexConfig, enable_monitoring: bool = True):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install with: pip install faiss-cpu")
        
        self.config = config
        self.enable_monitoring = enable_monitoring
        
        # Index storage
        self.index = None
        self.quantizer = None
        self.id_map = {}  # id -> metadata
        self.reverse_id_map = {}  # metadata -> id
        self.next_id = 0
        
        # Performance monitoring
        self.performance_stats = {
            'total_searches': 0,
            'total_additions': 0,
            'search_times': [],
            'batch_search_times': [],
            'add_times': [],
            'memory_usage_mb': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self._lock = RLock()
        
        # Caching system
        self.search_cache = {}
        self.cache_max_size = 1000
        self.query_cache_ttl = 300  # 5 minutes
        
        # Auto-tuning
        self.auto_tuning_enabled = True
        self.performance_threshold_ms = 20.0
        self.adaptive_parameters = {
            'nprobe': 10,  # IVF search parameter
            'ef_search': self.config.hnsw_ef_search  # HNSW search parameter
        }
        
        # Initialize index
        self._initialize_index()
        
        logger.info(f"Optimized FAISS index initialized: {config.index_type}, dimension={config.dimension}")
    
    def _initialize_index(self):
        """Initialize FAISS index based on configuration"""
        try:
            if self.config.index_type == "IVF_PQ":
                self._setup_ivf_pq_index()
            elif self.config.index_type == "IVF_FLAT":
                self._setup_ivf_flat_index()
            elif self.config.index_type == "HNSW":
                self._setup_hnsw_index()
            elif self.config.index_type == "IVFPQ_HNSW":
                self._setup_ivfpq_hnsw_index()
            else:
                raise ValueError(f"Unsupported index type: {self.config.index_type}")
            
            # Setup GPU if enabled
            if self.config.use_gpu:
                self._setup_gpu()
            
            # Start monitoring thread if enabled
            if self.enable_monitoring:
                self._start_monitoring()
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    
    def _setup_ivf_pq_index(self):
        """Setup IVF-PQ index for memory efficiency"""
        # Create quantizer
        self.quantizer = faiss.IndexFlatL2(self.config.dimension)
        
        # Determine optimal nlist based on data size estimation
        # Rule of thumb: nlist ≈ sqrt(num_vectors) but capped
        estimated_vectors = 100000  # Initial estimate
        optimal_nlist = min(max(int(np.sqrt(estimated_vectors)), 100), 1000)
        self.config.nlist = optimal_nlist
        
        # Create IVF-PQ index
        self.index = faiss.IndexIVFPQ(
            self.quantizer,
            self.config.dimension,
            self.config.nlist,
            self.config.m,
            self.config.nbits
        )
        
        # Optimize parameters for speed
        self.index.cp = faiss.IndexIVFPQ.SearchParameters(
            nprobe=self.adaptive_parameters['nprobe']
        )
        
        logger.info(f"Setup IVF-PQ index: nlist={self.config.nlist}, m={self.config.m}, nbits={self.config.nbits}")
    
    def _setup_ivf_flat_index(self):
        """Setup IVF-FLAT index for maximum accuracy"""
        # Create quantizer
        self.quantizer = faiss.IndexFlatL2(self.config.dimension)
        
        # Create IVF-FLAT index
        self.index = faiss.IndexIVFFlat(
            self.quantizer,
            self.config.dimension,
            self.config.nlist
        )
        
        # Optimize search parameters
        self.adaptive_parameters['nprobe'] = min(self.config.nlist // 10, 50)
        
        logger.info(f"Setup IVF-FLAT index: nlist={self.config.nlist}")
    
    def _setup_hnsw_index(self):
        """Setup HNSW index for fast approximate search"""
        self.index = faiss.IndexHNSWFlat(
            self.config.dimension,
            self.config.hnsw_m
        )
        
        # Set HNSW parameters
        self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
        self.index.hnsw.efSearch = self.adaptive_parameters['ef_search']
        
        logger.info(f"Setup HNSW index: m={self.config.hnsw_m}, ef_search={self.adaptive_parameters['ef_search']}")
    
    def _setup_ivfpq_hnsw_index(self):
        """Setup hybrid IVF-PQ+HNSW index"""
        # Create HNSW quantizer
        hnsw_quantizer = faiss.IndexHNSWFlat(
            self.config.dimension,
            self.config.hnsw_m
        )
        hnsw_quantizer.hnsw.efConstruction = self.config.hnsw_ef_construction
        
        # Create IVF-PQ index with HNSW quantizer
        self.index = faiss.IndexIVFPQ(
            hnsw_quantizer,
            self.config.dimension,
            self.config.nlist,
            self.config.m,
            self.config.nbits
        )
        
        logger.info(f"Setup IVFPQ-HNSW index: nlist={self.config.nlist}, m={self.config.m}")
    
    def _setup_gpu(self):
        """Setup GPU acceleration"""
        try:
            # GPU resource management
            self.res = faiss.StandardGpuResources()
            
            # Convert CPU index to GPU
            if hasattr(self.index, 'index'):
                # For IVF indexes
                gpu_index = faiss.index_cpu_to_gpu(
                    self.res,
                    self.config.gpu_id,
                    self.index.index
                )
                self.index.index = gpu_index
            else:
                # For direct indexes
                self.index = faiss.index_cpu_to_gpu(
                    self.res,
                    self.config.gpu_id,
                    self.index
                )
            
            logger.info(f"GPU acceleration enabled: device {self.config.gpu_id}")
        except Exception as e:
            logger.warning(f"GPU setup failed, falling back to CPU: {e}")
            self.config.use_gpu = False
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        def monitor():
            while self.enable_monitoring:
                try:
                    # Update performance statistics
                    self._update_performance_metrics()
                    
                    # Auto-tune parameters if performance is poor
                    if self.auto_tuning_enabled:
                        self._auto_tune_parameters()
                    
                    time.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring thread error: {e}")
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        with self._lock:
            # Memory usage
            if hasattr(self.index, 'ntotal'):
                num_vectors = self.index.ntotal
                memory_estimate = self.config.get_memory_estimate(num_vectors)
                self.performance_stats['memory_usage_mb'].append(memory_estimate['total_memory_mb'])
            
            # Cleanup old statistics
            for key in ['search_times', 'batch_search_times', 'add_times', 'memory_usage_mb']:
                if len(self.performance_stats[key]) > 1000:
                    self.performance_stats[key] = self.performance_stats[key][-500:]
    
    def _auto_tune_parameters(self):
        """Auto-tune search parameters based on performance"""
        if len(self.performance_stats['search_times']) < 10:
            return
        
        recent_searches = self.performance_stats['search_times'][-10:]
        avg_search_time = np.mean(recent_searches) * 1000  # Convert to ms
        
        if avg_search_time > self.performance_threshold_ms:
            # Performance is poor, increase search scope
            if 'nprobe' in self.adaptive_parameters:
                old_nprobe = self.adaptive_parameters['nprobe']
                self.adaptive_parameters['nprobe'] = min(old_nprobe + 5, self.config.nlist // 2)
                logger.info(f"Auto-tuning: increased nprobe from {old_nprobe} to {self.adaptive_parameters['nprobe']}")
            
            elif 'ef_search' in self.adaptive_parameters:
                old_ef = self.adaptive_parameters['ef_search']
                self.adaptive_parameters['ef_search'] = min(old_ef + 16, 512)
                logger.info(f"Auto-tuning: increased ef_search from {old_ef} to {self.adaptive_parameters['ef_search']}")
        
        elif avg_search_time < self.performance_threshold_ms * 0.5:
            # Performance is good, we can optimize for speed
            if 'nprobe' in self.adaptive_parameters and self.adaptive_parameters['nprobe'] > 5:
                old_nprobe = self.adaptive_parameters['nprobe']
                self.adaptive_parameters['nprobe'] = max(old_nprobe - 2, 1)
                logger.info(f"Auto-tuning: decreased nprobe from {old_nprobe} to {self.adaptive_parameters['nprobe']}")
    
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> List[int]:
        """
        Add vectors to the index with optimized batch processing.
        
        Args:
            vectors: Array of vectors (num_vectors, dimension)
            metadata: Optional metadata for each vector
            
        Returns:
            List of assigned IDs
        """
        if len(vectors.shape) != 2:
            raise ValueError("Vectors must be 2D array (num_vectors, dimension)")
        
        if vectors.shape[1] != self.config.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.config.dimension}, got {vectors.shape[1]}")
        
        start_time = time.time()
        
        with self._lock:
            # Prepare metadata
            if metadata is None:
                metadata = [{}] * len(vectors)
            elif len(metadata) != len(vectors):
                raise ValueError("Metadata length must match number of vectors")
            
            # Train index if needed (for IVF indexes)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("Training FAISS index...")
                # Use a subset for training (FAISS requirement)
                train_size = min(len(vectors), max(39 * self.config.nlist, 100))
                train_vectors = vectors[:train_size]
                self.index.train(train_vectors)
                logger.info("FAISS index training complete")
            
            # Add vectors in batches for better performance
            batch_size = min(10000, len(vectors))  # Optimal batch size
            assigned_ids = []
            
            for i in range(0, len(vectors), batch_size):
                batch_end = min(i + batch_size, len(vectors))
                batch_vectors = vectors[i:batch_end]
                
                # Add to FAISS index
                start_id = self.next_id
                self.index.add(batch_vectors)
                
                # Store metadata
                for j, meta in enumerate(metadata[i:batch_end]):
                    vector_id = start_id + j
                    self.id_map[vector_id] = meta
                    self.reverse_id_map[meta] = vector_id
                    assigned_ids.append(vector_id)
                
                self.next_id += len(batch_vectors)
            
            # Update performance stats
            add_time = time.time() - start_time
            self.performance_stats['add_times'].append(add_time)
            self.performance_stats['total_additions'] += len(vectors)
            
            logger.info(f"Added {len(vectors)} vectors in {add_time:.3f}s (batch size: {batch_size})")
            
            return assigned_ids
    
    def search(self, query_vectors: np.ndarray, k: int = 10, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors with optimized performance.
        
        Args:
            query_vectors: Query vectors (num_queries, dimension)
            k: Number of results to return
            use_cache: Whether to use query caching
            
        Returns:
            Tuple of (distances, indices) arrays
        """
        if len(query_vectors.shape) != 2:
            raise ValueError("Query vectors must be 2D array")
        
        if query_vectors.shape[1] != self.config.dimension:
            raise ValueError(f"Query vector dimension mismatch: expected {self.config.dimension}")
        
        start_time = time.time()
        
        # Check cache first
        cache_key = None
        if use_cache:
            cache_key = self._get_cache_key(query_vectors, k)
            if cache_key in self.search_cache:
                cached_result, cache_time = self.search_cache[cache_key]
                if time.time() - cache_time < self.query_cache_ttl:
                    self.performance_stats['cache_hits'] += 1
                    return cached_result
        
        with self._lock:
            # Perform search with optimized parameters
            search_start = time.time()
            
            # Set search parameters
            if hasattr(self.index, 'cp') and 'nprobe' in self.adaptive_parameters:
                self.index.cp.nprobe = self.adaptive_parameters['nprobe']
            elif hasattr(self.index, 'hnsw') and 'ef_search' in self.adaptive_parameters:
                self.index.hnsw.efSearch = self.adaptive_parameters['ef_search']
            
            # Execute search
            distances, indices = self.index.search(query_vectors, k)
            
            search_time = time.time() - search_start
            
            # Update statistics
            total_time = time.time() - start_time
            self.performance_stats['search_times'].append(total_time)
            self.performance_stats['total_searches'] += 1
            
            # Cache results if caching enabled
            if use_cache and cache_key:
                self._cache_search_result(cache_key, (distances, indices))
                self.performance_stats['cache_misses'] += 1
            
            # Log performance
            if self.performance_stats['total_searches'] % 100 == 0:
                avg_time = np.mean(self.performance_stats['search_times'][-100:]) * 1000
                logger.info(f"Average search time: {avg_time:.2f}ms ({len(query_vectors)} queries, k={k})")
        
        return distances, indices
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 10, 
                    batch_size: int = 1000, use_cache: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform batch search for large query sets with optimized throughput.
        
        Args:
            query_vectors: Query vectors (num_queries, dimension)
            k: Number of results per query
            batch_size: Process queries in batches for memory efficiency
            use_cache: Whether to use query caching
            
        Returns:
            Tuple of (distances, indices) arrays for all queries
        """
        if len(query_vectors) <= batch_size:
            return self.search(query_vectors, k, use_cache)
        
        all_distances = []
        all_indices = []
        
        for i in range(0, len(query_vectors), batch_size):
            batch_end = min(i + batch_size, len(query_vectors))
            batch_queries = query_vectors[i:batch_end]
            
            batch_distances, batch_indices = self.search(batch_queries, k, use_cache)
            
            all_distances.extend(batch_distances)
            all_indices.extend(batch_indices)
        
        return np.array(all_distances), np.array(all_indices)
    
    def _get_cache_key(self, query_vectors: np.ndarray, k: int) -> str:
        """Generate cache key for query vectors"""
        # Simple hash based on query vectors and k
        query_hash = hash(query_vectors.tobytes())
        return f"{query_hash}_{k}"
    
    def _cache_search_result(self, cache_key: str, result: Tuple[np.ndarray, np.ndarray]):
        """Cache search result"""
        with self._lock:
            # Cleanup old cache entries if necessary
            if len(self.search_cache) >= self.cache_max_size:
                # Remove oldest entries
                sorted_cache = sorted(self.search_cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_cache[:len(self.search_cache) - self.cache_max_size + 10]:
                    del self.search_cache[key]
            
            # Add new cache entry
            self.search_cache[cache_key] = (result, time.time())
    
    def get_vector_by_id(self, vector_id: int) -> Optional[Dict]:
        """Get metadata for vector by ID"""
        with self._lock:
            return self.id_map.get(vector_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self._lock:
            # Calculate averages
            avg_search_time = np.mean(self.performance_stats['search_times']) * 1000 if self.performance_stats['search_times'] else 0
            p95_search_time = np.percentile(self.performance_stats['search_times'], 95) * 1000 if self.performance_stats['search_times'] else 0
            p99_search_time = np.percentile(self.performance_stats['search_times'], 99) * 1000 if self.performance_stats['search_times'] else 0
            
            avg_add_time = np.mean(self.performance_stats['add_times']) if self.performance_stats['add_times'] else 0
            
            cache_hit_rate = (self.performance_stats['cache_hits'] / 
                            (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) * 100
                            if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0)
            
            # Memory usage
            current_memory = self.performance_stats['memory_usage_mb'][-1] if self.performance_stats['memory_usage_mb'] else 0
            
            return {
                'index_type': self.config.index_type,
                'total_vectors': self.next_id,
                'total_searches': self.performance_stats['total_searches'],
                'total_additions': self.performance_stats['total_additions'],
                'performance': {
                    'avg_search_time_ms': avg_search_time,
                    'p95_search_time_ms': p95_search_time,
                    'p99_search_time_ms': p99_search_time,
                    'avg_add_time_s': avg_add_time,
                    'meets_20ms_target': avg_search_time < 20.0
                },
                'cache': {
                    'hit_rate_percent': cache_hit_rate,
                    'cache_hits': self.performance_stats['cache_hits'],
                    'cache_misses': self.performance_stats['cache_misses'],
                    'cache_size': len(self.search_cache)
                },
                'memory': {
                    'current_usage_mb': current_memory,
                    'config_estimate': self.config.get_memory_estimate(self.next_id)
                },
                'adaptive_parameters': self.adaptive_parameters.copy(),
                'optimization': {
                    'auto_tuning_enabled': self.auto_tuning_enabled,
                    'gpu_enabled': self.config.use_gpu,
                    'monitoring_enabled': self.enable_monitoring
                }
            }
    
    def save_index(self, path: str):
        """Save FAISS index and metadata to disk"""
        with self._lock:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")
            
            # Save metadata and configuration
            save_data = {
                'config': self.config,
                'id_map': self.id_map,
                'next_id': self.next_id,
                'performance_stats': self.performance_stats,
                'adaptive_parameters': self.adaptive_parameters,
                'search_cache': self.search_cache
            }
            
            with open(f"{path}.meta", 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load FAISS index and metadata from disk"""
        with self._lock:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.faiss")
            
            # Load metadata
            with open(f"{path}.meta", 'rb') as f:
                save_data = pickle.load(f)
            
            self.config = save_data['config']
            self.id_map = save_data['id_map']
            self.next_id = save_data['next_id']
            self.performance_stats = save_data['performance_stats']
            self.adaptive_parameters = save_data['adaptive_parameters']
            self.search_cache = save_data.get('search_cache', {})
            
            logger.info(f"Index loaded from {path}")
    
    def optimize_index(self):
        """Optimize index for current workload"""
        logger.info("Optimizing FAISS index...")
        
        with self._lock:
            # For IVF indexes, consider rebalancing
            if hasattr(self.index, 'index') and hasattr(self.index.index, 'nlist'):
                current_nlist = self.index.index.nlist
                estimated_optimal = min(max(int(np.sqrt(self.next_id)), 100), 2000)
                
                if current_nlist != estimated_optimal:
                    logger.info(f"Rebalancing IVF clusters: {current_nlist} -> {estimated_optimal}")
                    # Note: Full rebalancing would require rebuilding index
                    # For now, just update the estimate
            
            # Update adaptive parameters based on current performance
            if self.performance_stats['search_times']:
                recent_avg = np.mean(self.performance_stats['search_times'][-100:]) * 1000
                
                if recent_avg < 10.0:  # Very fast, can be more aggressive
                    if 'nprobe' in self.adaptive_parameters:
                        self.adaptive_parameters['nprobe'] = max(1, self.adaptive_parameters['nprobe'] - 2)
                elif recent_avg > 30.0:  # Too slow, be more exhaustive
                    if 'nprobe' in self.adaptive_parameters:
                        self.adaptive_parameters['nprobe'] = min(self.config.nlist // 2, self.adaptive_parameters['nprobe'] + 3)
            
            logger.info(f"Index optimization complete. Current parameters: {self.adaptive_parameters}")
    
    def shutdown(self):
        """Graceful shutdown"""
        self.enable_monitoring = False
        
        # Cleanup resources
        if hasattr(self, 'res'):  # GPU resources
            del self.res
        
        logger.info("Optimized FAISS index shutdown complete")

def create_optimal_index(num_vectors: int, dimension: int, memory_budget_mb: float = 1000, 
                        use_gpu: bool = False, target_latency_ms: float = 20.0) -> OptimizedFAISSIndex:
    """
    Create optimally configured FAISS index based on requirements.
    
    Args:
        num_vectors: Expected number of vectors
        dimension: Vector dimension
        memory_budget_mb: Available memory budget in MB
        use_gpu: Whether to use GPU acceleration
        target_latency_ms: Target search latency in milliseconds
        
    Returns:
        Optimized FAISS index instance
    """
    # Determine optimal index type based on constraints
    if num_vectors < 10000:
        # Small dataset - use HNSW for best accuracy
        config = IndexConfig(
            dimension=dimension,
            index_type="HNSW",
            hnsw_m=32,
            hnsw_ef_construction=200,
            hnsw_ef_search=max(32, int(target_latency_ms / 2)),
            use_gpu=use_gpu
        )
    elif memory_budget_mb < 100:
        # Tight memory budget - use aggressive compression
        config = IndexConfig(
            dimension=dimension,
            index_type="IVF_PQ",
            nlist=min(1000, max(100, int(np.sqrt(num_vectors)))),
            m=16,  # More compression
            nbits=6,  # Lower precision for more compression
            use_gpu=use_gpu
        )
    else:
        # Balanced configuration
        config = IndexConfig(
            dimension=dimension,
            index_type="IVF_PQ",
            nlist=min(1000, max(100, int(np.sqrt(num_vectors)))),
            m=16,
            nbits=8,
            use_gpu=use_gpu
        )
    
    logger.info(f"Creating optimal index: {num_vectors} vectors, {memory_budget_mb}MB budget, {target_latency_ms}ms target")
    
    return OptimizedFAISSIndex(config)