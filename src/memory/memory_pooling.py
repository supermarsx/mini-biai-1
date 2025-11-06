import time
import threading
import logging
import array
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import RLock
import weakref
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PoolConfig:
    """Configuration for memory pools"""
    pool_size: int
    block_size: int
    max_blocks_per_pool: int
    growth_factor: float = 1.5
    shrink_threshold: float = 0.3  # Pool can shrink when utilization < 30%

class MemoryPool:
    """
    Thread-safe memory pool for efficient allocation/reuse of memory blocks.
    
    Features:
    - Fixed and variable size block pools
    - Automatic pool growth and shrinkage
    - Memory fragmentation prevention
    - Performance monitoring and optimization
    - Integration with NumPy arrays and standard Python objects
    """
    
    def __init__(self, pool_config: PoolConfig, pool_name: str = "default"):
        self.config = pool_config
        self.name = pool_name
        
        # Pool storage
        self.available_blocks = deque()  # Free blocks
        self.allocated_blocks = {}       # block_id -> block_info
        self.block_counter = 0
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'total_frees': 0,
            'current_allocated': 0,
            'peak_allocated': 0,
            'pool_growths': 0,
            'pool_shrinks': 0,
            'allocation_errors': 0,
            'fragmentation_score': 0.0
        }
        
        # Thread safety
        self._lock = RLock()
        
        # Background cleanup
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._cleanup_thread.start()
        
        # Initialize pool with minimum blocks
        self._initialize_pool()
        
        logger.info(f"Memory pool '{pool_name}' initialized: {pool_config.pool_size} blocks of {pool_config.block_size} bytes")
    
    def allocate(self, size: Optional[int] = None, object_type: str = "numpy") -> Union[np.ndarray, memoryview, Any]:
        """
        Allocate a memory block from the pool.
        
        Args:
            size: Requested size (uses config.block_size if None)
            object_type: Type of object to allocate ("numpy", "bytes", "array")
            
        Returns:
            Allocated memory block
        """
        requested_size = size or self.config.block_size
        
        # Check if we need a different sized pool
        if requested_size > self.config.block_size:
            return self._allocate_large_block(requested_size, object_type)
        
        with self._lock:
            # Try to get from available blocks
            if self.available_blocks:
                block = self.available_blocks.popleft()
                self.allocated_blocks[block['id']] = block
                self._update_allocation_stats()
                return block['data']
            
            # Need to grow the pool
            if len(self.allocated_blocks) + len(self.available_blocks) < self.config.max_blocks_per_pool:
                new_block = self._create_new_block(object_type)
                if new_block:
                    self.allocated_blocks[new_block['id']] = new_block
                    self._update_allocation_stats()
                    self.stats['pool_growths'] += 1
                    return new_block['data']
            
            # Pool exhausted
            self.stats['allocation_errors'] += 1
            logger.warning(f"Memory pool '{self.name}' exhausted")
            
            # Try to force garbage collection
            gc.collect()
            
            # Final attempt
            if self.available_blocks:
                block = self.available_blocks.popleft()
                self.allocated_blocks[block['id']] = block
                self._update_allocation_stats()
                return block['data']
            
            raise MemoryError(f"Memory pool '{self.name}' allocation failed - no memory available")
    
    def deallocate(self, block: Union[np.ndarray, memoryview, Any]):
        """
        Return a memory block to the pool for reuse.
        
        Args:
            block: Memory block to deallocate
        """
        if block is None:
            return
            
        with self._lock:
            # Find block info
            block_info = None
            block_id = None
            
            for bid, info in self.allocated_blocks.items():
                if info['data'] is block:
                    block_info = info
                    block_id = bid
                    break
            
            if block_info:
                # Reset block data
                if isinstance(block, np.ndarray):
                    block.fill(0)
                elif isinstance(block, bytearray):
                    block[:] = b'\x00' * len(block)
                
                # Return to available pool
                del self.allocated_blocks[block_id]
                self.available_blocks.append(block_info)
                self.stats['total_frees'] += 1
                self.stats['current_allocated'] -= 1
                
                # Check if we should shrink the pool
                self._maybe_shrink_pool()
            else:
                logger.warning(f"Trying to deallocate unknown block from pool '{self.name}'")
    
    def _allocate_large_block(self, size: int, object_type: str) -> Union[np.ndarray, memoryview, Any]:
        """Allocate a block larger than the standard block size"""
        try:
            if object_type == "numpy":
                block = np.zeros(size, dtype=np.uint8)
            elif object_type == "bytes":
                block = bytearray(size)
            elif object_type == "array":
                block = array.array('B', [0] * size)
            else:
                block = bytearray(size)
                
            self.stats['total_allocations'] += 1
            return block
        except MemoryError:
            self.stats['allocation_errors'] += 1
            raise
    
    def _create_new_block(self, object_type: str) -> Optional[Dict]:
        """Create a new memory block"""
        try:
            if object_type == "numpy":
                data = np.zeros(self.config.block_size, dtype=np.uint8)
            elif object_type == "bytes":
                data = bytearray(self.config.block_size)
            elif object_type == "array":
                data = array.array('B', [0] * self.config.block_size)
            else:
                data = bytearray(self.config.block_size)
                
            self.block_counter += 1
            return {
                'id': f"{self.name}_{self.block_counter}",
                'data': data,
                'allocated_at': time.time(),
                'size': self.config.block_size
            }
        except Exception as e:
            logger.error(f"Failed to create new block in pool '{self.name}': {e}")
            return None
    
    def _initialize_pool(self):
        """Initialize pool with initial blocks"""
        for _ in range(self.config.pool_size):
            try:
                block = self._create_new_block("numpy")
                if block:
                    self.available_blocks.append(block)
            except Exception as e:
                logger.warning(f"Failed to initialize block in pool '{self.name}': {e}")
                break
    
    def _update_allocation_stats(self):
        """Update allocation statistics"""
        self.stats['total_allocations'] += 1
        self.stats['current_allocated'] += 1
        self.stats['peak_allocated'] = max(self.stats['peak_allocated'], self.stats['current_allocated'])
    
    def _maybe_shrink_pool(self):
        """Consider shrinking the pool if utilization is low"""
        total_blocks = len(self.allocated_blocks) + len(self.available_blocks)
        if total_blocks == 0:
            return
            
        utilization = len(self.allocated_blocks) / total_blocks
        
        if utilization < self.config.shrink_threshold and len(self.available_blocks) > self.config.pool_size:
            # Remove some free blocks
            blocks_to_remove = min(
                len(self.available_blocks) - self.config.pool_size,
                int(len(self.available_blocks) * 0.3)  # Remove 30% of excess
            )
            
            for _ in range(blocks_to_remove):
                if self.available_blocks:
                    self.available_blocks.popleft()
                    self.stats['pool_shrinks'] += 1
    
    def _background_cleanup(self):
        """Background thread for pool optimization and cleanup"""
        while self._cleanup_running:
            try:
                # Update fragmentation score
                self._update_fragmentation_score()
                
                # Periodic pool optimization
                if len(self.available_blocks) > self.config.pool_size * 2:
                    self._maybe_shrink_pool()
                
                time.sleep(30)  # Run every 30 seconds
            except Exception as e:
                logger.error(f"Background cleanup error in pool '{self.name}': {e}")
    
    def _update_fragmentation_score(self):
        """Calculate pool fragmentation score"""
        if not self.available_blocks:
            self.stats['fragmentation_score'] = 0.0
            return
        
        # Simple fragmentation metric based on block age distribution
        now = time.time()
        old_blocks = 0
        total_blocks = len(self.available_blocks)
        
        for block in self.available_blocks:
            if now - block['allocated_at'] > 3600:  # Older than 1 hour
                old_blocks += 1
        
        if total_blocks > 0:
            self.stats['fragmentation_score'] = old_blocks / total_blocks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self._lock:
            total_blocks = len(self.allocated_blocks) + len(self.available_blocks)
            utilization = (len(self.allocated_blocks) / total_blocks * 100) if total_blocks > 0 else 0
            
            return {
                'pool_name': self.name,
                'total_blocks': total_blocks,
                'allocated_blocks': len(self.allocated_blocks),
                'available_blocks': len(self.available_blocks),
                'utilization_percent': utilization,
                'total_allocations': self.stats['total_allocations'],
                'total_frees': self.stats['total_frees'],
                'current_allocated': self.stats['current_allocated'],
                'peak_allocated': self.stats['peak_allocated'],
                'pool_growths': self.stats['pool_growths'],
                'pool_shrinks': self.stats['pool_shrinks'],
                'allocation_errors': self.stats['allocation_errors'],
                'fragmentation_score': self.stats['fragmentation_score'],
                'config': {
                    'pool_size': self.config.pool_size,
                    'block_size': self.config.block_size,
                    'max_blocks_per_pool': self.config.max_blocks_per_pool
                }
            }
    
    def optimize(self):
        """Manually optimize the pool"""
        with self._lock:
            # Force shrink if needed
            self._maybe_shrink_pool()
            
            # Update fragmentation score
            self._update_fragmentation_score()
            
            logger.info(f"Optimized memory pool '{self.name}'")
    
    def clear(self):
        """Clear the pool - free all blocks"""
        with self._lock:
            self.available_blocks.clear()
            self.allocated_blocks.clear()
            self.stats['current_allocated'] = 0
            
            # Reinitialize
            self._initialize_pool()
            
            logger.info(f"Cleared memory pool '{self.name}'")
    
    def shutdown(self):
        """Graceful pool shutdown"""
        self._cleanup_running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        with self._lock:
            self.available_blocks.clear()
            self.allocated_blocks.clear()
        
        logger.info(f"Memory pool '{self.name}' shutdown complete")

class MemoryPoolManager:
    """
    Manager for multiple memory pools with different configurations.
    
    Features:
    - Multiple pool types (small, medium, large objects)
    - Automatic pool selection based on allocation size
    - Cross-pool optimization and balancing
    - Performance monitoring and reporting
    """
    
    def __init__(self):
        self.pools = {}  # pool_name -> MemoryPool
        self.pool_configs = {
            'small': PoolConfig(pool_size=100, block_size=1024, max_blocks_per_pool=1000),
            'medium': PoolConfig(pool_size=50, block_size=8192, max_blocks_per_pool=500),
            'large': PoolConfig(pool_size=20, block_size=65536, max_blocks_per_pool=200),
            'numpy_small': PoolConfig(pool_size=50, block_size=4096, max_blocks_per_pool=500),
            'numpy_large': PoolConfig(pool_size=10, block_size=262144, max_blocks_per_pool=100)
        }
        
        # Initialize pools
        for name, config in self.pool_configs.items():
            self.pools[name] = MemoryPool(config, name)
        
        # Statistics
        self._global_stats = {
            'total_allocations': 0,
            'pool_selections': defaultdict(int),
            'allocation_size_distribution': defaultdict(int)
        }
        
        self._lock = RLock()
        
        logger.info(f"Memory pool manager initialized with {len(self.pools)} pools")
    
    def allocate(self, size: int, object_type: str = "numpy", pool_preference: Optional[str] = None) -> Union[np.ndarray, memoryview, Any]:
        """
        Allocate memory using optimal pool selection.
        
        Args:
            size: Requested memory size
            object_type: Type of object ("numpy", "bytes", "array") 
            pool_preference: Preferred pool name (auto-select if None)
            
        Returns:
            Allocated memory block
        """
        with self._lock:
            self._global_stats['total_allocations'] += 1
            self._global_stats['allocation_size_distribution'][min(size, 262144)] += 1
            
            # Auto-select pool if not specified
            if not pool_preference:
                pool_preference = self._select_optimal_pool(size, object_type)
            
            self._global_stats['pool_selections'][pool_preference] += 1
            
            # Allocate from selected pool
            if pool_preference in self.pools:
                return self.pools[pool_preference].allocate(size, object_type)
            else:
                # Fallback to small pool
                return self.pools['small'].allocate(min(size, 1024), object_type)
    
    def deallocate(self, block: Union[np.ndarray, memoryview, Any], pool_name: str = "auto"):
        """
        Deallocate memory back to appropriate pool.
        
        Args:
            block: Memory block to deallocate
            pool_name: Pool name ("auto" to detect automatically)
        """
        if pool_name == "auto":
            # Find which pool owns this block
            for pool in self.pools.values():
                if any(info['data'] is block for info in pool.allocated_blocks.values()):
                    pool.deallocate(block)
                    return
            
            # If not found, log warning
            logger.warning("Could not determine pool for deallocation")
        else:
            if pool_name in self.pools:
                self.pools[pool_name].deallocate(block)
    
    def _select_optimal_pool(self, size: int, object_type: str) -> str:
        """Select the optimal pool for given size and type"""
        # Size-based selection
        if object_type == "numpy":
            if size <= 4096:
                return "numpy_small"
            else:
                return "numpy_large"
        else:
            if size <= 1024:
                return "small"
            elif size <= 8192:
                return "medium"
            else:
                return "large"
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global pool manager statistics"""
        with self._lock:
            # Aggregate pool statistics
            pool_stats = {}
            for name, pool in self.pools.items():
                pool_stats[name] = pool.get_stats()
            
            return {
                'total_allocations': self._global_stats['total_allocations'],
                'pool_selections': dict(self._global_stats['pool_selections']),
                'allocation_size_distribution': dict(self._global_stats['allocation_size_distribution']),
                'pools': pool_stats,
                'num_pools': len(self.pools)
            }
    
    def optimize_all_pools(self):
        """Optimize all managed pools"""
        for pool in self.pools.values():
            pool.optimize()
        
        logger.info("Optimized all memory pools")
    
    def clear_all_pools(self):
        """Clear all managed pools"""
        for pool in self.pools.values():
            pool.clear()
        
        logger.info("Cleared all memory pools")
    
    def shutdown(self):
        """Shutdown all pools"""
        for pool in self.pools.values():
            pool.shutdown()
        
        self.pools.clear()
        logger.info("Memory pool manager shutdown complete")

# Global instance for convenient access
_global_pool_manager = None

def get_memory_pool_manager() -> MemoryPoolManager:
    """Get global memory pool manager instance"""
    global _global_pool_manager
    if _global_pool_manager is None:
        _global_pool_manager = MemoryPoolManager()
    return _global_pool_manager

def allocate_memory(size: int, object_type: str = "numpy", pool_preference: Optional[str] = None) -> Union[np.ndarray, memoryview, Any]:
    """Convenience function for memory allocation"""
    return get_memory_pool_manager().allocate(size, object_type, pool_preference)

def deallocate_memory(block: Union[np.ndarray, memoryview, Any], pool_name: str = "auto"):
    """Convenience function for memory deallocation"""
    get_memory_pool_manager().deallocate(block, pool_name)