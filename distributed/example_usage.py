#!/usr/bin/env python3
"""Comprehensive examples for using the distributed training module.

This script demonstrates various use cases of the distributed training framework,
from basic single-GPU training to advanced multi-node setups with fault tolerance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import logging
import os
import sys
from typing import Dict, Any, Optional
import json
import time
from pathlib import Path

# Import our distributed training modules
from distributed import (
    setup_distributed_training,
    DistributedTraining,
    DistributedTrainer,
    DistributedDataLoader,
    ZeROOptimizer,
    ModelParallel,
    TensorParallel,
    PipelineParallel,
    CheckpointManager,
    TrainingMonitor,
    get_rank,
    get_world_size,
    is_distributed
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExampleDataset(torch.utils.data.Dataset):
    """Example dataset for demonstration purposes."""
    
    def __init__(self, size: int = 10000, input_dim: int = 100, num_classes: int = 10):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Generate random data
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, num_classes, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_dim: int = 100, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

class ComplexModel(nn.Module):
    """More complex model for parallel training demonstration."""
    
    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024, num_classes: int = 100):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

def example_basic_distributed_training():
    """Example 1: Basic distributed training setup."""
    logger.info("=== Example 1: Basic Distributed Training ===")
    
    # Create model and dataset
    model = SimpleModel()
    train_dataset = ExampleDataset(size=5000)
    val_dataset = ExampleDataset(size=1000)
    
    # Configuration
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'zero_stage': 2,
        'fp16': True,
        'save_dir': './checkpoints/basic',
        'save_interval': 500,
        'log_interval': 10
    }
    
    try:
        # Setup distributed training
        dist_training = setup_distributed_training(model, config)
        
        # Create data loaders
        train_loader = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            pin_memory=True
        )
        
        val_loader = DistributedDataLoader(
            dataset=val_dataset,
            batch_size=config['batch_size']
        )
        
        # Train for a few epochs
        logger.info("Starting training...")
        dist_training.train(epochs=3, data_loader=train_loader)
        
        # Evaluate
        logger.info("Evaluating model...")
        metrics = dist_training.evaluate(data_loader=val_loader)
        logger.info(f"Validation metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Basic training example failed: {e}")

def example_advanced_zerostage3():
    """Example 2: Advanced ZeRO Stage 3 optimization with memory offloading."""
    logger.info("=== Example 2: ZeRO Stage 3 with Memory Offloading ===")
    
    # Create larger model for ZeRO Stage 3 demonstration
    model = ComplexModel(input_dim=1024, hidden_dim=2048, num_classes=100)
    train_dataset = ExampleDataset(size=10000, input_dim=1024, num_classes=100)
    
    # Advanced configuration for ZeRO Stage 3
    config = {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'zero_stage': 3,
        'fp16': True,
        'offload_optimizer': True,
        'offload_param': True,
        'activation_checkpointing': True,
        'gradient_accumulation_steps': 4,
        'gradient_clipping': 1.0,
        'save_dir': './checkpoints/zerostage3',
        'save_interval': 1000,
        'log_interval': 20
    }
    
    try:
        dist_training = setup_distributed_training(model, config)
        
        train_loader = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True
        )
        
        # Train with memory optimization
        logger.info("Starting memory-optimized training...")
        dist_training.train(epochs=2, data_loader=train_loader)
        
        logger.info("ZeRO Stage 3 training completed successfully")
        
    except Exception as e:
        logger.error(f"ZeRO Stage 3 example failed: {e}")

def example_model_parallelism():
    """Example 3: Model parallelism for large models."""
    logger.info("=== Example 3: Model Parallelism ===")
    
    # Create very large model
    model = ComplexModel(input_dim=2048, hidden_dim=4096, num_classes=200)
    train_dataset = ExampleDataset(size=5000, input_dim=2048, num_classes=200)
    
    # Configuration for model parallelism
    config = {
        'batch_size': 8,
        'learning_rate': 5e-5,
        'zero_stage': 2,
        'fp16': True,
        'save_dir': './checkpoints/model_parallel',
        'save_interval': 2000
    }
    
    try:
        # Use model parallelism
        if torch.cuda.device_count() >= 2:
            devices = list(range(min(2, torch.cuda.device_count())))
            model_parallel = ModelParallel(
                model=model,
                devices=devices,
                strategy='layer'  # Split model by layers
            )
            model = model_parallel.get_parallel_model()
        
        dist_training = setup_distributed_training(model, config)
        
        train_loader = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            pin_memory=True
        )
        
        logger.info("Starting model parallel training...")
        dist_training.train(epochs=1, data_loader=train_loader)
        
        logger.info("Model parallel training completed")
        
    except Exception as e:
        logger.error(f"Model parallelism example failed: {e}")

def example_fault_tolerance():
    """Example 4: Fault-tolerant training with automatic recovery."""
    logger.info("=== Example 4: Fault-Tolerant Training ===")
    
    model = SimpleModel()
    train_dataset = ExampleDataset(size=3000)
    
    config = {
        'batch_size': 16,
        'learning_rate': 1e-3,
        'zero_stage': 2,
        'fp16': True,
        'save_dir': './checkpoints/fault_tolerant',
        'save_interval': 200,
        'log_interval': 5
    }
    
    try:
        dist_training = setup_distributed_training(model, config)
        
        # Enable fault tolerance
        train_loader = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size']
        )
        train_loader.enable_fault_tolerance(retry_attempts=3)
        
        # Simulate fault scenario or train normally
        logger.info("Starting fault-tolerant training...")
        try:
            dist_training.train(epochs=2, data_loader=train_loader)
        except Exception as e:
            logger.warning(f"Training interrupted, attempting recovery: {e}")
            # Attempt recovery from checkpoint
            if dist_training.checkpoint_manager.has_checkpoint():
                dist_training.load_checkpoint(
                    dist_training.checkpoint_manager._find_latest_checkpoint()
                )
                logger.info("Successfully recovered from checkpoint")
            else:
                logger.info("No checkpoint found, starting fresh")
        
        logger.info("Fault-tolerant training completed")
        
    except Exception as e:
        logger.error(f"Fault tolerance example failed: {e}")

def example_custom_monitoring():
    """Example 5: Custom monitoring and performance tracking."""
    logger.info("=== Example 5: Custom Monitoring ===")
    
    model = SimpleModel()
    train_dataset = ExampleDataset(size=2000)
    
    config = {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'zero_stage': 2,
        'fp16': True,
        'save_dir': './checkpoints/monitored',
        'save_interval': 500,
        'log_interval': 5
    }
    
    try:
        dist_training = setup_distributed_training(model, config)
        
        # Customize monitoring
        monitor = dist_training.monitor
        
        # Add custom metrics
        def track_custom_metrics(loss, step):
            # Log loss with additional context
            if step % 50 == 0:
                logger.info(f"Step {step}: Loss = {loss:.4f}")
        
        # Add custom callback
        def learning_rate_callback(metrics):
            if 'learning_rate' in metrics:
                logger.debug(f"Current learning rate: {metrics['learning_rate']}")
        
        monitor.add_callback('custom_tracking', learning_rate_callback)
        
        # Configure performance tracking
        monitor.config['track_system_stats'] = True
        monitor.config['track_memory_usage'] = True
        
        train_loader = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            pin_memory=True
        )
        
        logger.info("Starting training with custom monitoring...")
        dist_training.train(epochs=2, data_loader=train_loader)
        
        # Get final performance statistics
        stats = monitor.get_performance_stats()
        logger.info(f"Final training statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Custom monitoring example failed: {e}")

def example_checkpoint_management():
    """Example 6: Advanced checkpoint management."""
    logger.info("=== Example 6: Checkpoint Management ===")
    
    model = SimpleModel()
    train_dataset = ExampleDataset(size=1000)
    
    # Custom checkpoint manager configuration
    config = {
        'batch_size': 16,
        'learning_rate': 1e-3,
        'zero_stage': 2,
        'fp16': True,
        'save_dir': './checkpoints/advanced',
        'save_interval': 100,  # Frequent saves for demonstration
        'keep_checkpoints': 5
    }
    
    try:
        dist_training = setup_distributed_training(model, config)
        
        # Create custom checkpoint manager
        checkpoint_manager = dist_training.checkpoint_manager
        
        # Custom save function
        def custom_save(engine, step, optimizer=None, **kwargs):
            # Create custom checkpoint structure
            checkpoint_dir = checkpoint_manager.save_dir / f'custom_step_{step}'
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save with custom format
            checkpoint = {
                'model_state_dict': engine.state_dict(),
                'step': step,
                'custom_data': {
                    'training_time': time.time(),
                    'batch_count': step
                }
            }
            
            torch.save(checkpoint, checkpoint_dir / 'custom_checkpoint.pt')
            return str(checkpoint_dir)
        
        # Replace save function
        original_save = checkpoint_manager.save_function
        checkpoint_manager.save_function = custom_save
        
        train_loader = DistributedDataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size']
        )
        
        logger.info("Starting training with advanced checkpointing...")
        dist_training.train(epochs=1, data_loader=train_loader)
        
        # List and manage checkpoints
        checkpoints = checkpoint_manager.list_checkpoints()
        logger.info(f"Available checkpoints: {len(checkpoints)}")
        for checkpoint in checkpoints:
            logger.info(f"  - {checkpoint['path']} at step {checkpoint['step']}")
        
        # Clean up old checkpoints
        removed = checkpoint_manager.cleanup_checkpoints(keep=3)
        logger.info(f"Removed {len(removed)} old checkpoints")
        
    except Exception as e:
        logger.error(f"Checkpoint management example failed: {e}")

def main():
    """Run all examples."""
    logger.info("Starting distributed training examples...")
    
    # Create output directories
    os.makedirs('./checkpoints', exist_ok=True)
    
    # Run examples
    examples = [
        example_basic_distributed_training,
        example_advanced_zerostage3,
        example_model_parallelism,
        example_fault_tolerance,
        example_custom_monitoring,
        example_checkpoint_management
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            logger.info(f"Running example {i}/{len(examples)}: {example_func.__name__}")
            example_func()
        except Exception as e:
            logger.error(f"Example {example_func.__name__} failed: {e}")
            continue
    
    logger.info("All examples completed!")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Distributed training examples')
    parser.add_argument('--example', type=str, choices=[
        'basic', 'zerostage3', 'parallel', 'fault_tolerance', 'monitoring', 'checkpointing'
    ], help='Run specific example')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
    parser.add_argument('--world_size', type=int, default=1, help='Number of GPUs')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Run example or all examples
    if args.example:
        example_map = {
            'basic': example_basic_distributed_training,
            'zerostage3': example_advanced_zerostage3,
            'parallel': example_model_parallelism,
            'fault_tolerance': example_fault_tolerance,
            'monitoring': example_custom_monitoring,
            'checkpointing': example_checkpoint_management
        }
        if args.example in example_map:
            example_map[args.example]()
        else:
            logger.error(f"Unknown example: {args.example}")
    else:
        main()
