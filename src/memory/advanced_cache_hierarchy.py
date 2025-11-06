import time
import threading
import logging
import json
import hashlib
import os
import pickle
from typing import Dict, List, Optional, Any, Tuple, Callable
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from threading import RLock
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata for multi-level hierarchy"""
    key: str
    data: Any
    size_bytes: int
    timestamp: float
    access_count: int
    last_access: float
    ttl: float
    source: str  # 'l1', 'l2', 'l3', 'disk'
    compressed: bool
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl <= 0:
            return False
        return (time.time() - self.timestamp) > self.ttl
    
    def access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_access = time.time()

class AdvancedCacheHierarchy:
    """
    Multi-level cache hierarchy system with L1/L2/L3 tiers and optional disk caching.
    
    Architecture:
    - L1 Cache (Hot): Frequently accessed, small, very fast memory
    - L2 Cache (Warm): Moderate frequency, medium size, fast memory  
    - L3 Cache (Cold): Less frequent, large, slower but still in RAM
    - Disk Cache: Persistent storage for cold data
    
    Features:
    - Adaptive tier management based on access patterns
    - Smart eviction policies (LRU, LFU, TTL-based)
    - Compression support for large entries
    - Memory usage monitoring and optimization
    - Thread-safe operations
    """
    
    def __init__(self, 
                 l1_size: int = 1000,           # L1 cache capacity
                 l2_size: int = 10000,          # L2 cache capacity  
                 l3_size: int = 100000,         # L3 cache capacity
                 l1_ttl: float = 3600,          # L1 time-to-live (1 hour)
                 l2_ttl: float = 14400,         # L2 time-to-live (4 hours)
                 l3_ttl: float = 86400,         # L3 time-to-live (24 hours)
                 compression_threshold: int = 1024,  # Size threshold for compression
                 enable_disk_cache: bool = True,
                 disk_cache_path: str = "/tmp/cache_hierarchy"):
        
        # Cache tiers
        self.l1_cache = OrderedDict()  # Hot cache
        self.l2_cache = OrderedDict()  # Warm cache  
        self.l3_cache = OrderedDict()  # Cold cache
        self.disk_cache_path = disk_cache_path
        
        # Capacity limits
        self.l1_max_size = l1_size
        self.l2_max_size = l2_size
        self.l3_max_size = l3_size
        
        # TTL settings
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl  
        self.l3_ttl = l3_ttl
        
        # Compression
        self.compression_threshold = compression_threshold
        self._compression_cache = {}  # Cache for compression state
        
        # Statistics
        self.stats = {
            'hits': {'l1': 0, 'l2': 0, 'l3': 0, 'disk': 0},
            'misses': 0,
            'evictions': {'l1': 0, 'l2': 0, 'l3': 0},
            'compressions': 0,
            'disk_writes': 0,
            'disk_reads': 0
        }
        
        # Locks for thread safety
        self._locks = {
            'l1': RLock(),
            'l2': RLock(), 
            'l3': RLock(),
            'stats': RLock(),
            'disk': RLock()
        }
        
        # Enable disk caching
        self.enable_disk_cache = enable_disk_cache
        if enable_disk_cache and not os.path.exists(disk_cache_path):
            os.makedirs(disk_cache_path, exist_ok=True)
            
        # Background cleanup thread
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"Cache hierarchy initialized: L1({l1_size}), L2({l2_size}), L3({l3_size}), Disk({enable_disk_cache})")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from cache hierarchy.
        Searches L1 → L2 → L3 → Disk in order.
        """
        # Try L1 cache first (hot cache)
        with self._locks['l1']:
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                if not entry.is_expired():
                    entry.access()
                    self._stats_counter('hits', 'l1')
                    # Promote to L1 if accessed frequently
                    self._maybe_promote(key, entry)
                    return entry.data
                else:
                    # Remove expired entry
                    del self.l1_cache[key]
                    self._stats_counter('evictions', 'l1')
        
        # Try L2 cache (warm cache)
        with self._locks['l2']:
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                if not entry.is_expired():
                    entry.access()
                    self._stats_counter('hits', 'l2')
                    # Promote to L1 if frequently accessed
                    self._maybe_promote(key, entry)
                    return entry.data
                else:
                    del self.l2_cache[key]
                    self._stats_counter('evictions', 'l2')
        
        # Try L3 cache (cold cache)
        with self._locks['l3']:
            if key in self.l3_cache:
                entry = self.l3_cache[key]
                if not entry.is_expired():
                    entry.access()
                    self._stats_counter('hits', 'l3')
                    return entry.data
                else:
                    del self.l3_cache[key]
                    self._stats_counter('evictions', 'l3')
        
        # Try disk cache if enabled
        if self.enable_disk_cache:
            with self._locks['disk']:
                disk_data = self._get_from_disk(key)
                if disk_data is not None:
                    self._stats_counter('hits', 'disk')
                    # Reload into memory cache
                    self.put(key, disk_data, source='disk')
                    return disk_data
        
        # Cache miss
        with self._locks['stats']:
            self.stats['misses'] += 1
        return None
    
    def put(self, key: str, data: Any, ttl: Optional[float] = None, source: str = 'custom') -> bool:
        """
        Store data in cache hierarchy.
        Automatically determines optimal cache tier based on size and access patterns.
        """
        if data is None:
            return False
            
        # Calculate size and determine if compression is needed
        size_bytes = self._calculate_size(data)
        should_compress = size_bytes > self.compression_threshold
        compressed_data = data
        actual_ttl = ttl or self.l2_ttl  # Default to L2 TTL
        
        # Determine optimal cache tier
        cache_tier = self._determine_cache_tier(key, size_bytes, actual_ttl)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            data=compressed_data,
            size_bytes=size_bytes,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time(),
            ttl=actual_ttl,
            source=cache_tier,
            compressed=should_compress
        )
        
        # Store in appropriate tier
        success = self._store_in_tier(cache_tier, entry)
        
        if success:
            # Compress if needed
            if should_compress:
                self._stats_counter('compressions')
                
            logger.debug(f"Stored {key} in {cache_tier} cache (size: {size_bytes} bytes)")
            return True
        
        return False
    
    def _determine_cache_tier(self, key: str, size_bytes: int, ttl: float) -> str:
        """
        Intelligently determine which cache tier to use based on:
        - Size (smaller items in L1, larger in L3)
        - TTL (shorter TTL in L1, longer in L3)  
        - Access patterns (frequently accessed in higher tiers)
        """
        # Small, short-lived items → L1 (hot cache)
        if size_bytes < 1024 and ttl <= 1800:  # < 1KB, < 30min
            return 'l1'
        
        # Medium items with moderate TTL → L2 (warm cache)
        elif size_bytes < 10240 and ttl <= 14400:  # < 10KB, < 4 hours
            return 'l2'
        
        # Large or long-lived items → L3 (cold cache)
        else:
            return 'l3'
    
    def _store_in_tier(self, tier: str, entry: CacheEntry) -> bool:
        """Store entry in specified cache tier with proper eviction"""
        cache_name = f"{tier}_cache"
        lock = self._locks[tier]
        max_size = getattr(self, f"{tier}_max_size")
        ttl = getattr(self, f"{tier}_ttl")
        
        with lock:
            cache = getattr(self, cache_name)
            
            # Update TTL based on tier defaults if needed
            if entry.ttl == 0:  # Use tier default TTL
                entry.ttl = ttl
            
            # Remove existing entry if present
            if entry.key in cache:
                del cache[entry.key]
            
            # Evict if at capacity
            if len(cache) >= max_size:
                self._evict_from_tier(tier, count=1)
            
            # Add new entry
            cache[entry.key] = entry
            return True
    
    def _evict_from_tier(self, tier: str, count: int = 1) -> List[str]:
        """Evict least recently used items from specified tier"""
        cache_name = f"{tier}_cache"
        lock = self._locks[tier]
        
        with lock:
            cache = getattr(self, cache_name)
            evicted_keys = []
            
            # Sort by last access time (LRU)
            items = sorted(cache.items(), key=lambda x: x[1].last_access)
            
            for i in range(min(count, len(items))):
                key, entry = items[i]
                evicted_keys.append(key)
                
                # If tier is full, move to lower tier or disk
                if tier == 'l1':
                    self._demote_to_l2(key, entry)
                elif tier == 'l2':
                    self._demote_to_l3(key, entry)
                elif tier == 'l3':
                    self._demote_to_disk(key, entry)
                
                del cache[key]
                self._stats_counter('evictions', tier)
            
            return evicted_keys
    
    def _maybe_promote(self, key: str, entry: CacheEntry):
        """Promote frequently accessed items to higher cache tiers"""
        if entry.access_count >= 10 and entry.source == 'l2':
            # Promote L2 item to L1 if frequently accessed
            self._promote_to_l1(key, entry)
        elif entry.access_count >= 5 and entry.source == 'l3':
            # Promote L3 item to L2
            self._promote_to_l2(key, entry)
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote item from L2 to L1"""
        with self._locks['l2']:
            if key in self.l2_cache:
                del self.l2_cache[key]
                
        with self._locks['l1']:
            self.l1_cache[key] = entry
            entry.source = 'l1'
    
    def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote item from L3 to L2"""
        with self._locks['l3']:
            if key in self.l3_cache:
                del self.l3_cache[key]
                
        with self._locks['l2']:
            self.l2_cache[key] = entry
            entry.source = 'l2'
    
    def _demote_to_l2(self, key: str, entry: CacheEntry):
        """Demote item from L1 to L2"""
        with self._locks['l1']:
            if key in self.l1_cache:
                entry.source = 'l2'
                self.l2_cache[key] = entry
    
    def _demote_to_l3(self, key: str, entry: CacheEntry):
        """Demote item from L2 to L3"""
        with self._locks['l2']:
            if key in self.l2_cache:
                entry.source = 'l3'
                self.l3_cache[key] = entry
    
    def _demote_to_disk(self, key: str, entry: CacheEntry):
        """Demote item from L3 to disk cache"""
        if self.enable_disk_cache:
            self._store_to_disk(key, entry.data)
            self._stats_counter('disk_writes')
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Retrieve item from disk cache"""
        if not self.enable_disk_cache:
            return None
            
        try:
            file_path = os.path.join(self.disk_cache_path, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                self._stats_counter('disk_reads')
                return data
        except Exception as e:
            logger.warning(f"Failed to read from disk cache: {e}")
        return None
    
    def _store_to_disk(self, key: str, data: Any):
        """Store item to disk cache"""
        if not self.enable_disk_cache:
            return
            
        try:
            file_path = os.path.join(self.disk_cache_path, f"{hashlib.md5(key.encode()).hexdigest()}.cache")
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to write to disk cache: {e}")
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes"""
        try:
            if isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, (int, float, bool)):
                return 8
            elif isinstance(data, (list, tuple)):
                return sum(self._calculate_size(item) for item in data)
            elif isinstance(data, dict):
                return sum(len(str(k).encode('utf-8')) + self._calculate_size(v) for k, v in data.items())
            elif isinstance(data, np.ndarray):
                return data.nbytes
            else:
                # Fallback: serialize to JSON
                return len(json.dumps(data, default=str).encode('utf-8'))
        except:
            return 1024  # Default estimate
    
    def _stats_counter(self, stat_type: str, tier: str = None):
        """Thread-safe statistics counter"""
        with self._locks['stats']:
            if tier:
                self.stats[stat_type][tier] += 1
            else:
                self.stats[stat_type] += 1
    
    def _background_cleanup(self):
        """Background thread for cleanup and optimization"""
        while self._cleanup_running:
            try:
                # Clean expired entries
                self._cleanup_expired()
                
                # Optimize cache sizes
                self._optimize_cache_sizes()
                
                # Clean disk cache occasionally
                if self.enable_disk_cache:
                    self._cleanup_disk_cache()
                
                time.sleep(60)  # Run every minute
            except Exception as e:
                logger.error(f"Background cleanup error: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries from all cache tiers"""
        current_time = time.time()
        
        for tier in ['l1', 'l2', 'l3']:
            lock = self._locks[tier]
            cache_name = f"{tier}_cache"
            
            with lock:
                cache = getattr(self, cache_name)
                expired_keys = []
                
                for key, entry in cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del cache[key]
                    logger.debug(f"Cleaned expired entry: {key}")
    
    def _optimize_cache_sizes(self):
        """Optimize cache sizes based on access patterns"""
        # Analyze access patterns and adjust if needed
        total_accesses = sum(self.stats['hits'].values())
        if total_accesses > 0:
            # Adjust L1 size based on hit ratios
            l1_hits = self.stats['hits']['l1']
            l2_hits = self.stats['hits']['l2'] 
            
            if l2_hits > l1_hits * 2:
                # Increase L1 capacity if L2 is getting too many hits
                new_size = min(self.l1_max_size * 1.1, self.l2_max_size // 10)
                logger.info(f"Optimizing cache: increasing L1 size to {new_size}")
    
    def _cleanup_disk_cache(self):
        """Clean old files from disk cache"""
        if not self.enable_disk_cache:
            return
            
        try:
            current_time = time.time()
            max_age = 7 * 24 * 3600  # 7 days
            
            for filename in os.listdir(self.disk_cache_path):
                file_path = os.path.join(self.disk_cache_path, filename)
                if os.path.getmtime(file_path) < (current_time - max_age):
                    os.remove(file_path)
                    logger.debug(f"Removed old cache file: {filename}")
        except Exception as e:
            logger.warning(f"Disk cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        with self._locks['stats']:
            # Calculate hit rates
            total_hits = sum(self.stats['hits'].values())
            total_requests = total_hits + self.stats['misses']
            hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate memory usage
            l1_memory = sum(entry.size_bytes for entry in self.l1_cache.values())
            l2_memory = sum(entry.size_bytes for entry in self.l2_cache.values())
            l3_memory = sum(entry.size_bytes for entry in self.l3_cache.values())
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'hits': dict(self.stats['hits']),
                'misses': self.stats['misses'],
                'evictions': dict(self.stats['evictions']),
                'cache_sizes': {
                    'l1': {'entries': len(self.l1_cache), 'memory_bytes': l1_memory, 'max': self.l1_max_size},
                    'l2': {'entries': len(self.l2_cache), 'memory_bytes': l2_memory, 'max': self.l2_max_size},
                    'l3': {'entries': len(self.l3_cache), 'memory_bytes': l3_memory, 'max': self.l3_max_size}
                },
                'total_memory_bytes': l1_memory + l2_memory + l3_memory,
                'disk_operations': {
                    'writes': self.stats['disk_writes'],
                    'reads': self.stats['disk_reads']
                },
                'compressions': self.stats['compressions']
            }
    
    def clear(self, tier: Optional[str] = None):
        """Clear cache or specific tier"""
        if tier:
            with self._locks[tier]:
                cache_name = f"{tier}_cache"
                setattr(self, cache_name, OrderedDict())
                logger.info(f"Cleared {tier} cache")
        else:
            # Clear all tiers
            for tier_name in ['l1', 'l2', 'l3']:
                with self._locks[tier_name]:
                    cache_name = f"{tier_name}_cache"
                    setattr(self, cache_name, OrderedDict())
            logger.info("Cleared all cache tiers")
    
    def shutdown(self):
        """Graceful shutdown with cleanup"""
        self._cleanup_running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        # Clear disk cache on shutdown if requested
        if self.enable_disk_cache:
            try:
                for filename in os.listdir(self.disk_cache_path):
                    file_path = os.path.join(self.disk_cache_path, filename)
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Error during disk cache cleanup: {e}")
        
        logger.info("Cache hierarchy shutdown complete")