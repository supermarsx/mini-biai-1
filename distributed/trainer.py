import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import logging
import time
import os
import json
import copy
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import warnings
from collections import defaultdict, deque
import numpy as np

from .data_loader import DistributedSampler
from .optimizer import ZeroRedundancyOptimizer, FlexibleOptimizer
from .monitor import TrainingMonitor
from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Comprehensive distributed training orchestrator.
    
    This class provides high-level training orchestration for distributed
    deep learning, including data parallelism, model parallelism support,
    mixed precision training, gradient accumulation, and fault tolerance.
    
    Features:
    - Multi-GPU and multi-node training
    - ZeRO optimizer integration
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Custom learning rate schedules
    - Fault tolerance and recovery
    - Comprehensive monitoring and logging
    - Checkpoint management
    - Memory optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Union[optim.Optimizer, ZeroRedundancyOptimizer, FlexibleOptimizer],
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        loss_function: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize distributed trainer.
        
        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance (supports custom distributed optimizers)
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            loss_function: Loss function (default: CrossEntropyLoss for classification)
            device: Training device (auto-detected if None)
            config: Training configuration dictionary
        """
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function or nn.CrossEntropyLoss()
        self.config = config or {}
        
        # Setup device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_metric = float('inf')
        self.start_time = time.time()
        self.training_history = defaultdict(list)
        
        # Training configuration
        self._setup_config()
        
        # Initialize distributed environment
        self._setup_distributed()
        
        # Initialize monitoring
        if self.rank == 0:
            self.monitor = TrainingMonitor(
                log_interval=self.config.get('log_interval', 10),
                save_logs=self.config.get('save_logs', True),
                track_gpu_memory=self.config.get('track_gpu_memory', True),
                enable_alerts=self.config.get('enable_alerts', True)
            )
        else:
            self.monitor = None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.config.get('checkpoint_dir', './checkpoints'),
            keep_n_checkpoints=self.config.get('keep_n_checkpoints', 3),
            save_interval=self.config.get('checkpoint_interval', 1)
        )
        
        # Training statistics
        self.stats = {
            'total_epochs': 0,
            'total_steps': 0,
            'total_time': 0,
            'validation_time': 0,
            'best_metric': float('inf'),
            'best_epoch': 0
        }
        
        logger.info(f"Initialized DistributedTrainer on device {self.device}")
    
    def _setup_config(self):
        """Setup training configuration with defaults."""
        defaults = {
            'num_epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 0.0,
            'clip_grad_norm': 1.0,
            'grad_accumulation_steps': 1,
            'mixed_precision': False,
            'torch_dtype': torch.float16,
            'validation_interval': 1,
            'early_stopping_patience': 10,
            'lr_scheduler': None,
            'lr_scheduler_config': {},
            'log_interval': 10,
            'save_logs': True,
            'checkpoint_dir': './checkpoints',
            'checkpoint_interval': 1,
            'keep_n_checkpoints': 3,
            'track_gpu_memory': True,
            'enable_alerts': True,
            'debug_mode': False
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
        
        logger.info(f"Training configuration: {self.config}")
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        self.is_distributed = dist.is_initialized()
        
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Wrap model with DDP if using data parallelism
            if not isinstance(self.optimizer, (ZeroRedundancyOptimizer, FlexibleOptimizer)):
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=True
                )
            
            logger.info(f"Initialized distributed training: rank={self.rank}, world_size={self.world_size}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.model = self.model.to(self.device)
            
            logger.info("Initialized single process training")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary with epoch training metrics
        """
        self.model.train()
        epoch_start_time = time.time()
        
        # Setup dataloader for this epoch
        if self.is_distributed and hasattr(self.train_dataloader, 'sampler'):
            self.train_dataloader.sampler.set_epoch(epoch)
        
        # Initialize epoch metrics
        epoch_metrics = defaultdict(float)
        num_batches = len(self.train_dataloader)
        
        # Training loop
        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Forward pass
                loss, batch_metrics = self._train_batch(batch, epoch, batch_idx, num_batches)
                
                # Update metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                
                # Log progress
                if self.rank == 0 and (batch_idx + 1) % self.config['log_interval'] == 0:
                    progress = (batch_idx + 1) / num_batches * 100
                    logger.info(f"Epoch {epoch}/{self.config['num_epochs']} - "
                              f"Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%) - "
                              f"Loss: {loss.item():.6f}")
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx} of epoch {epoch}: {e}")
                if self.config['debug_mode']:
                    raise
                continue
        
        # Average metrics over all batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Add timing information
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['steps_per_second'] = num_batches / epoch_time
        
        # Update statistics
        self.stats['total_steps'] += num_batches
        self.stats['total_time'] += epoch_time
        
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        return dict(epoch_metrics)
    
    def _train_batch(
        self, 
        batch: Any, 
        epoch: int, 
        batch_idx: int, 
        num_batches: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Train a single batch.
        
        Args:
            batch: Input batch
            epoch: Current epoch
            batch_idx: Current batch index
            num_batches: Total number of batches in epoch
        
        Returns:
            Tuple of (loss, batch_metrics)
        """
        # Move batch to device
        if isinstance(batch, (tuple, list)):
            batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
        elif isinstance(batch, dict):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        else:
            batch = batch.to(self.device)
        
        # Forward pass
        outputs = self.model(batch)
        loss = self._compute_loss(outputs, batch)
        
        # Mixed precision training
        if self.config['mixed_precision']:
            with torch.cuda.amp.autocast():
                loss = self._scale_loss(loss)
        
        # Gradient accumulation
        loss = loss / self.config['grad_accumulation_steps']
        loss.backward()
        
        # Update weights if accumulation is complete
        if (batch_idx + 1) % self.config['grad_accumulation_steps'] == 0:
            self._update_parameters()
        
        # Unscale gradients for monitoring
        if self.config['mixed_precision']:
            self.optimizer.unscale_grads()
        
        # Gradient clipping
        if self.config['clip_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['clip_grad_norm']
            )
        
        # Track metrics
        batch_metrics = {
            'loss': loss.item() * self.config['grad_accumulation_steps'],
            'learning_rate': self._get_current_lr()
        }
        
        # Update monitoring
        if self.monitor:
            self.monitor.set_step(self.current_step)
            self.monitor.set_epoch(epoch)
            for key, value in batch_metrics.items():
                self.monitor.track_metric(key, value)
        
        self.current_step += 1
        
        return loss, batch_metrics
    
    def _compute_loss(self, outputs: Any, batch: Any) -> torch.Tensor:
        """Compute loss for the current batch.
        
        Args:
            outputs: Model outputs
            batch: Input batch
        
        Returns:
            Computed loss tensor
        """
        if isinstance(outputs, tuple):
            # Assume first element is the loss
            if isinstance(outputs[0], torch.Tensor):
                return outputs[0]
            # Otherwise, need to compute loss from outputs and targets
            targets = batch[1] if len(batch) > 1 else None
            if targets is not None:
                return self.loss_function(outputs, targets)
        
        elif isinstance(outputs, dict):
            # Loss is provided in outputs
            if 'loss' in outputs:
                return outputs['loss']
        
        # Default: assume outputs are logits and batch contains targets
        targets = batch[1] if len(batch) > 1 else batch.get('targets')
        if targets is not None:
            return self.loss_function(outputs, targets)
        
        raise ValueError("Could not compute loss: unclear output/target format")
    
    def _scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if isinstance(self.optimizer, (ZeroRedundancyOptimizer, FlexibleOptimizer)):
            return self.optimizer.scale_loss(loss)
        return loss
    
    def _update_parameters(self):
        """Update model parameters."""
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Learning rate scheduling
        if self.config['lr_scheduler'] is not None:
            self.config['lr_scheduler'].step()
    
    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate model for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dictionary with validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_start_time = time.time()
        
        # Initialize metrics
        val_metrics = defaultdict(float)
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in batch]
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                batch_metrics = self._compute_validation_metrics(outputs, batch)
                
                # Update metrics
                batch_size = self._get_batch_size(batch)
                for key, value in batch_metrics.items():
                    val_metrics[key] += value * batch_size
                
                num_samples += batch_size
        
        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= max(num_samples, 1)
        
        # Add timing
        val_time = time.time() - val_start_time
        val_metrics['validation_time'] = val_time
        val_metrics['val_samples_per_second'] = num_samples / val_time
        
        # Update statistics
        self.stats['validation_time'] += val_time
        
        logger.info(f"Validation completed in {val_time:.2f}s")
        
        return dict(val_metrics)
    
    def _compute_validation_metrics(self, outputs: Any, batch: Any) -> Dict[str, float]:
        """Compute validation metrics.
        
        Args:
            outputs: Model outputs
            batch: Input batch
        
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        # Loss
        try:
            loss = self._compute_loss(outputs, batch)
            metrics['val_loss'] = loss.item()
        except:
            metrics['val_loss'] = 0.0
        
        # Accuracy (if applicable)
        targets = batch[1] if len(batch) > 1 else batch.get('targets')
        if targets is not None and isinstance(outputs, torch.Tensor):
            with torch.no_grad():
                predictions = torch.argmax(outputs, dim=-1)
                correct = (predictions == targets).float().mean()
                metrics['val_accuracy'] = correct.item()
        
        return metrics
    
    def _get_batch_size(self, batch: Any) -> int:
        """Get batch size from batch object."""
        if isinstance(batch, dict) and 'input_ids' in batch:
            return batch['input_ids'].size(0)
        elif isinstance(batch, (tuple, list)) and len(batch) > 0:
            if isinstance(batch[0], torch.Tensor):
                return batch[0].size(0)
        return 32  # Default assumption
    
    def train(
        self, 
        num_epochs: Optional[int] = None,
        early_stopping: bool = True,
        save_best_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """Run complete training loop.
        
        Args:
            num_epochs: Number of epochs to train (uses config if None)
            early_stopping: Enable early stopping
            save_best_checkpoint: Save checkpoint for best model
        
        Returns:
            Training results dictionary
        """
        num_epochs = num_epochs or self.config['num_epochs']
        early_stopping_patience = self.config['early_stopping_patience']
        
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = {}
            if (epoch + 1) % self.config['validation_interval'] == 0:
                val_metrics = self.validate_epoch(epoch)
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update monitoring
            if self.monitor:
                for key, value in epoch_metrics.items():
                    self.monitor.track_metric(key, value)
                self.monitor.log_metrics(force=True)
            
            # Checkpointing
            is_best = False
            if save_best_checkpoint and 'val_loss' in epoch_metrics:
                if epoch_metrics['val_loss'] < self.best_metric:
                    self.best_metric = epoch_metrics['val_loss']
                    self.stats['best_metric'] = self.best_metric
                    self.stats['best_epoch'] = epoch
                    is_best = True
            
            if self.rank == 0 and (epoch + 1) % self.config['checkpoint_interval'] == 0:
                checkpoint_info = {
                    'epoch': epoch,
                    'model_state_dict': self._get_model_state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': epoch_metrics,
                    'config': self.config,
                    'stats': self.stats
                }
                self.checkpoint_manager.save_checkpoint(checkpoint_info, is_best=is_best)
            
            # Early stopping
            if early_stopping and 'val_loss' in epoch_metrics:
                if not hasattr(self, 'no_improvement_count'):
                    self.no_improvement_count = 0
                
                if epoch_metrics['val_loss'] < self.best_metric:
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
                
                if self.no_improvement_count >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Final training statistics
        total_time = time.time() - start_time
        self.stats['total_epochs'] = self.current_epoch + 1
        self.stats['total_time'] = total_time
        
        logger.info(f"Training completed in {total_time:.2f}s")
        
        if self.monitor:
            self.monitor.log_metrics(force=True)
        
        return {
            'final_metrics': epoch_metrics,
            'training_stats': self.stats,
            'best_metric': self.best_metric,
            'best_epoch': self.stats['best_epoch']
        }
    
    def _get_model_state_dict(self) -> Dict[str, Any]:
        """Get model state dictionary, handling distributed wrappers."""
        if isinstance(self.model, DDP):
            return self.model.module.state_dict()
        elif isinstance(self.optimizer, ZeroRedundancyOptimizer):
            # Get the underlying model for ZeRO
            return self.optimizer.model.state_dict()
        else:
            return self.model.state_dict()
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Loaded training state
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
        
        # Restore model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(self.optimizer, ZeroRedundancyOptimizer):
            self.optimizer.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['stats'].get('best_metric', float('inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        return checkpoint
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary.
        
        Returns:
            Training summary dictionary
        """
        return {
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            },
            'training_config': self.config,
            'training_stats': self.stats,
            'distributed_info': {
                'is_distributed': self.is_distributed,
                'rank': self.rank,
                'world_size': self.world_size,
                'device': str(self.device)
            },
            'optimization_info': {
                'optimizer_type': type(self.optimizer).__name__,
                'mixed_precision': self.config.get('mixed_precision', False),
                'gradient_accumulation': self.config.get('grad_accumulation_steps', 1)
            }
        }
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        if self.is_distributed and self.rank == 0:
            logger.info("Cleaning up distributed training resources")


# Export main class
__all__ = ['DistributedTrainer']