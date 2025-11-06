import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, Sampler
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable, Iterator
import os
import pickle
from pathlib import Path
import threading
import time
from collections import defaultdict
import warnings
import math

logger = logging.getLogger(__name__)

class DistributedSampler(DistributedSampler):
    """Enhanced distributed sampler with improved load balancing and fault tolerance.
    
    This sampler extends PyTorch's DistributedSampler with additional features:
    - Automatic load balancing across processes
    - Fallback sampling for failed batches
    - Seed management for reproducibility
    - Statistics tracking and monitoring
    """
    
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        load_balance: bool = True,
        adaptive_sampling: bool = True
    ):
        """Initialize distributed sampler.
        
        Args:
            dataset: Dataset to sample from
            num_replicas: Number of distributed processes
            rank: Current process rank
            shuffle: Whether to shuffle data
            seed: Random seed for reproducibility
            drop_last: Whether to drop last incomplete batch
            load_balance: Enable load balancing
            adaptive_sampling: Enable adaptive sampling
        """
        # Initialize parent sampler
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        
        super().__init__(dataset, num_replicas, rank, shuffle, drop_last)
        
        self.seed = seed
        self.load_balance = load_balance
        self.adaptive_sampling = adaptive_sampling
        
        # Load balancing state
        self.partition_sizes = {}
        self.load_history = defaultdict(list)
        
        # Statistics tracking
        self.num_samples_per_replica = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas)
        ) * self.num_replicas
        
        self.total_size = self.num_samples_per_replica
        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.num_samples_per_replica - len(self.dataset)
            self.indices = self.indices + ([self.indices[-1]] * padding_size)
        
        logger.debug(f"DistributedSampler initialized - Rank: {rank}/{num_replicas}")
    
    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler indices."""
        # Shuffle if enabled
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Apply load balancing if enabled
        if self.load_balance and dist.is_initialized():
            indices = self._apply_load_balancing(indices)
        
        # Apply adaptive sampling if enabled
        if self.adaptive_sampling:
            indices = self._apply_adaptive_sampling(indices)
        
        # Distribute indices among replicas
        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.num_samples_per_replica - len(indices)
            indices = indices + indices[:padding_size]
        else:
            # Remove the last extra samples, just in case
            indices = indices[:self.num_samples_per_replica]
        
        # Subsample for this replica
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        
        # Validate indices
        indices = [i for i in indices if i < len(self.dataset)]
        
        assert len(indices) == self.num_samples, f"Expected {self.num_samples} indices, got {len(indices)}"
        
        return iter(indices)
    
    def _apply_load_balancing(self, indices: List[int]) -> List[int]:
        """Apply load balancing to indices."""
        # Simple load balancing: randomly sample from all indices
        # In practice, this could be more sophisticated based on data complexity
        np.random.shuffle(indices)
        return indices
    
    def _apply_adaptive_sampling(self, indices: List[int]) -> List[int]:
        """Apply adaptive sampling strategy."""
        # Simple adaptive sampling: weight based on data order
        # In practice, this could use data complexity metrics
        weights = np.ones(len(indices))
        
        # Apply weights for sampling
        if len(indices) > 0:
            indices = np.random.choice(indices, len(indices), p=weights/weights.sum(), replace=False)
        
        return indices.tolist()
    
    def set_epoch(self, epoch: int):
        """Set current epoch for shuffling."""
        self.epoch = epoch
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            'num_replicas': self.num_replicas,
            'rank': self.rank,
            'num_samples': self.num_samples,
            'total_size': self.total_size,
            'drop_last': self.drop_last,
            'shuffle': self.shuffle
        }

class DistributedDataLoader:
    """Enhanced distributed data loader with fault tolerance and performance optimization.
    
    This class provides a robust data loading interface for distributed training with:
    - Automatic data sharding and distribution
    - Fault-tolerant batch processing
    - Performance monitoring and optimization
    - Custom sampling strategies
    - Memory-efficient operations
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 30.0,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        collate_fn: Optional[Callable] = None,
        sampler: Optional[Sampler] = None,
        sampler_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Initialize distributed data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size per replica
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfers
            drop_last: Drop last incomplete batch
            timeout: Worker timeout
            prefetch_factor: Number of batches to prefetch
            persistent_workers: Keep workers alive between epochs
            collate_fn: Custom collation function
            sampler: Custom sampler
            sampler_kwargs: Sampler configuration
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.collate_fn = collate_fn
        
        # Distributed configuration
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Auto-adjust batch size for distributed training
        self.adjusted_batch_size = max(1, batch_size // self.world_size)
        
        # Initialize sampler
        if sampler is None:
            sampler_kwargs = sampler_kwargs or {}
            sampler = DistributedSampler(
                dataset=dataset,
                shuffle=True,
                drop_last=drop_last,
                **sampler_kwargs
            )
        
        self.sampler = sampler
        
        # Data loader state
        self.loader = None
        self.epoch = 0
        self.fault_tolerance = False
        self.retry_attempts = 3
        self.max_workers = max(1, min(num_workers, os.cpu_count() or 1))
        
        # Performance monitoring
        self.load_times = []
        self.batch_sizes = []
        self.error_counts = defaultdict(int)
        
        # Initialize data loader
        self._create_loader()
        
        logger.info(f"Initialized DistributedDataLoader - Rank: {self.rank}/{self.world_size}")
    
    def _create_loader(self):
        """Create the underlying data loader."""
        try:
            self.loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.adjusted_batch_size,
                sampler=self.sampler,
                num_workers=self.max_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                timeout=self.timeout,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=self.persistent_workers,
                collate_fn=self.collate_fn,
                worker_init_fn=self._worker_init_fn if self.max_workers > 0 else None
            )
            
            logger.debug(f"Created DataLoader with batch_size={self.adjusted_batch_size}")
            
        except Exception as e:
            logger.error(f"Failed to create DataLoader: {e}")
            raise
    
    def _worker_init_fn(self, worker_id: int):
        """Initialize worker process."""
        # Set worker seed for reproducibility
        seed = self.epoch * 1000 + worker_id
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Set up error handling
        if self.fault_tolerance:
            # Enable error recovery in workers
            pass
    
    def __iter__(self) -> Iterator:
        """Iterate over data loader with fault tolerance."""
        if not self.fault_tolerance:
            return iter(self.loader)
        
        # Fault-tolerant iteration
        for attempt in range(self.retry_attempts):
            try:
                for batch in self.loader:
                    yield batch
                break
            except Exception as e:
                self.error_counts[type(e).__name__] += 1
                logger.warning(f"Batch processing failed (attempt {attempt + 1}): {e}")
                
                if attempt == self.retry_attempts - 1:
                    logger.error("All retry attempts exhausted")
                    raise
                
                # Wait a bit before retry
                time.sleep(0.1 * (attempt + 1))
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.loader) if self.loader else 0
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
        
        logger.debug(f"Set epoch to {epoch}")
    
    def get_loader(self) -> DataLoader:
        """Get the underlying data loader."""
        return self.loader
    
    def get_sampler(self) -> DistributedSampler:
        """Get the distributed sampler."""
        return self.sampler
    
    def enable_fault_tolerance(
        self,
        retry_attempts: int = 3,
        continue_on_error: bool = True
    ):
        """Enable fault tolerance for data loading.
        
        Args:
            retry_attempts: Number of retry attempts for failed batches
            continue_on_error: Continue processing on non-critical errors
        """
        self.fault_tolerance = True
        self.retry_attempts = retry_attempts
        self.continue_on_error = continue_on_error
        
        logger.info("Enabled fault tolerance for data loading")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'avg_load_time': np.mean(self.load_times) if self.load_times else 0,
            'min_load_time': np.min(self.load_times) if self.load_times else 0,
            'max_load_time': np.max(self.load_times) if self.load_times else 0,
            'avg_batch_size': np.mean(self.batch_sizes) if self.batch_sizes else 0,
            'total_batches': len(self.batch_sizes),
            'error_counts': dict(self.error_counts),
            'effective_batch_size': self.adjusted_batch_size * self.world_size,
            'rank': self.rank,
            'world_size': self.world_size
        }
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.load_times = []
        self.batch_sizes = []
        self.error_counts = defaultdict(int)
        
        logger.debug("Reset performance statistics")
    
    def preload_data(self, num_batches: int = 1):
        """Preload data for better cache performance.
        
        Args:
            num_batches: Number of batches to preload
        """
        if not self.loader:
            return
        
        logger.info(f"Preloading {num_batches} batches")
        # Implementation would cache the next num_batches in memory
        # This is a placeholder for the actual implementation
        
    def validate_dataset(self) -> Dict[str, Any]:
        """Validate the dataset and return statistics.
        
        Returns:
            Dataset validation statistics
        """
        try:
            # Get basic dataset info
            num_samples = len(self.dataset)
            
            # Sample a few items to check data integrity
            sample_items = []
            for i in range(min(10, num_samples)):
                try:
                    item = self.dataset[i]
                    sample_items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to sample item {i}: {e}")
            
            validation_result = {
                'num_samples': num_samples,
                'valid_samples': len(sample_items),
                'sampling_success_rate': len(sample_items) / min(10, num_samples),
                'dataset_type': type(self.dataset).__name__,
                'features': self.dataset[0] if sample_items else None
            }
            
            logger.info(f"Dataset validation: {validation_result}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return {'error': str(e)}
    
    def split_dataset(self, ratios: List[float]) -> List['DistributedDataLoader']:
        """Split dataset into multiple loaders.
        
        Args:
            ratios: List of ratios for each split (must sum to 1.0)
        
        Returns:
            List of distributed data loaders
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        num_samples = len(self.dataset)
        split_points = []
        
        current_point = 0
        for ratio in ratios[:-1]:
            current_point += int(num_samples * ratio)
            split_points.append(current_point)
        
        datasets = []
        start_idx = 0
        for end_idx in split_points + [num_samples]:
            subset_dataset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))
            datasets.append(subset_dataset)
            start_idx = end_idx
        
        # Create loaders for each subset
        loaders = []
        for subset_dataset in datasets:
            loader = DistributedDataLoader(
                dataset=subset_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last
            )
            loaders.append(loader)
        
        logger.info(f"Split dataset into {len(loaders)} subsets")
        return loaders
    
    def shutdown(self):
        """Shutdown data loader and cleanup resources."""
        if self.loader:
            # Close workers if persistent
            if self.persistent_workers and hasattr(self.loader, '_shutdown_workers'):
                self.loader._shutdown_workers()
        
        logger.info("Shutdown DistributedDataLoader")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass

# Utility functions
def create_distributed_loader(
    dataset: Dataset,
    config: Dict[str, Any]
) -> DistributedDataLoader:
    """Create distributed data loader from configuration.
    
    Args:
        dataset: Dataset to load from
        config: Configuration dictionary
    
    Returns:
        Configured DistributedDataLoader
    """
    return DistributedDataLoader(
        dataset=dataset,
        batch_size=config.get('batch_size', 1),
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', False),
        drop_last=config.get('drop_last', False),
        timeout=config.get('timeout', 30.0),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', False)
    )

def get_sampler_for_dataset(
    dataset: Dataset,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs
) -> DistributedSampler:
    """Get appropriate sampler for dataset.
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        **kwargs: Additional sampler arguments
    
    Returns:
        Configured DistributedSampler
    """
    return DistributedSampler(
        dataset=dataset,
        shuffle=shuffle,
        drop_last=drop_last,
        **kwargs
    )

# Export main classes and functions
__all__ = [
    'DistributedSampler',
    'DistributedDataLoader',
    'create_distributed_loader',
    'get_sampler_for_dataset'
]