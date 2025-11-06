# Distributed Training Core System

A production-ready distributed training framework for brain-inspired AI models that enables scalable, fault-tolerant training across multiple GPUs and nodes while maintaining biological plausibility.

## 🚀 Features

### Core Capabilities
- **DeepSpeed Integration**: Full ZeRO optimization (Stage 1, 2, 3) with automatic gradient partitioning
- **Model Parallelism**: Support for large models that exceed single GPU memory
- **Mixed Precision Training**: FP16/BF16 with automatic loss scaling
- **Fault Tolerance**: Automatic recovery from training interruptions and hardware failures
- **Real-time Monitoring**: Comprehensive performance tracking and anomaly detection
- **Memory Optimization**: CPU offloading, gradient checkpointing, and memory-efficient training

### Advanced Training Features
- **Pipeline Parallelism**: Sequential model processing across multiple devices
- **Tensor Parallelism**: Large tensor operations split across workers
- **Load Balancing**: Automatic work distribution for optimal performance
- **Distributed Data Loading**: Fault-tolerant, efficient data preprocessing
- **Incremental Checkpointing**: Atomic saves with automatic cleanup
- **Communication Optimization**: Overlap computation with communication

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## 🔧 Installation

### Prerequisites
- PyTorch >= 1.12.0
- Python >= 3.8
- CUDA >= 11.0 (for GPU training)
- DeepSpeed >= 0.8.0 (optional but recommended)

### Install Dependencies
```bash
pip install torch>=1.12.0
pip install deepspeed  # Optional but recommended
pip install numpy>=1.21.0
pip install psutil  # For system monitoring
pip install matplotlib  # For visualization
pip install tensorboard  # Optional, for logging
pip install wandb  # Optional, for experiment tracking
```

### Verify Installation
```python
from src.distributed import is_distributed, get_world_size
print(f"Distributed training available: {is_distributed()}")
print(f"World size: {get_world_size()}")
```

## 🚀 Quick Start

### Basic Distributed Training
```python
from src.distributed import (
    DistributedTrainer, 
    DistributedTrainerConfig, 
    MixedPrecisionType,
    create_distributed_dataloader,
    DistributedDataLoaderConfig
)

# Configure trainer
trainer_config = DistributedTrainerConfig(
    zero_stage=2,
    mixed_precision=MixedPrecisionType.BF16,
    gradient_accumulation_steps=4,
    enable_recovery=True
)

# Initialize trainer
trainer = DistributedTrainer(
    model=your_model,
    config=trainer_config,
    optimizer=your_optimizer
)

# Load data
data_config = DistributedDataLoaderConfig(
    batch_size=32,
    num_workers=4,
    enable_fault_tolerance=True
)
train_loader = create_distributed_dataloader(train_dataset, data_config)

# Train
results = trainer.train(
    train_loader=train_loader,
    num_epochs=10,
    checkpoint_dir="./checkpoints"
)
```

### Advanced Configuration
```python
# Enable all advanced features
trainer_config = DistributedTrainerConfig(
    zero_stage=3,  # Full ZeRO optimization
    mixed_precision=MixedPrecisionType.BF16,
    gradient_accumulation_steps=8,
    enable_recovery=True,
    checkpoint_frequency=5,
    health_check_interval=100,
    tensorboard_logging=True,
    wandb_logging=True,
    experiment_name="brain_inspired_training"
)

# Apply model parallelism for large models
from src.distributed.parallel import (
    ParallelConfig, 
    ParallelMode, 
    create_parallel_strategy
)

parallel_config = ParallelConfig(
    mode=ParallelMode.HYBRID_PARALLEL,
    devices=[0, 1, 2, 3],
    zero_stage=3
)
parallel_strategy = create_parallel_strategy(parallel_config)
model = parallel_strategy.wrap_model(your_model)
```

## 🏗️ Architecture

### System Overview
```
┌─────────────────────────────────────────────────────────────┐
│              Distributed Training Core                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼─────────────────────┐
│                     ▼                     │
│  ┌──────────────┐   ┌──────────────┐     │
│  │ Distributed  │   │ Model &      │     │
│  │ Trainer      │   │ Tensor        │     │
│  │              │   │ Parallelism   │     │
│  └──────────────┘   └──────────────┘     │
│           │                │              │
└───────────┼────────────────┼──────────────┘
            ▼                ▼
    ┌──────────────┐   ┌──────────────┐
    │  ZeRO        │   │ Distributed  │
    │  Optimizer   │   │ Data Loading │
    │              │   │              │
    └──────────────┘   └──────────────┘
            │                │
            └────────┬───────┘
                     ▼
        ┌─────────────────────────────┐
        │  Checkpointing & Monitoring │
        │                             │
        │ • Fault-tolerant saves     │
        │ • Real-time metrics        │
        │ • Performance analysis     │
        └─────────────────────────────┘
```

### Component Interaction
1. **DistributedTrainer** orchestrates the entire training process
2. **ZeROOptimizer** manages memory-efficient gradient synchronization
3. **Parallel Strategies** handle model distribution across devices
4. **DistributedDataLoader** provides efficient, fault-tolerant data loading
5. **CheckpointManager** handles atomic saves and recovery
6. **Monitoring System** tracks performance and detects anomalies

## 🧩 Components

### 1. DistributedTrainer (`trainer.py`)
Main orchestrator for distributed training with DeepSpeed integration.

**Key Features:**
- Mixed precision training (FP16/BF16)
- Automatic gradient accumulation
- Fault tolerance with recovery
- Integration with existing training pipeline
- Real-time performance metrics

**Configuration:**
```python
trainer_config = DistributedTrainerConfig(
    zero_stage=3,                    # ZeRO optimization stage
    mixed_precision=MixedPrecisionType.BF16,
    gradient_accumulation_steps=4,
    gradient_clip_val=1.0,
    enable_recovery=True,
    checkpoint_frequency=10,
    health_check_interval=100,
    log_frequency=10
)
```

### 2. Parallel Strategies (`parallel.py`)
Implements various parallelism strategies for large models.

**Supported Modes:**
- **Data Parallelism**: Replicates model across devices
- **Model Parallelism**: Splits model across devices
- **Tensor Parallelism**: Splits large tensor operations
- **Pipeline Parallelism**: Sequential model processing
- **Hybrid Parallelism**: Combines multiple strategies

**Usage:**
```python
# Auto-detect optimal strategy
from src.distributed.parallel import parallelize_model
model, strategy = parallelize_model(model, auto_detect=True)

# Manual configuration
config = ParallelConfig(
    mode=ParallelMode.TENSOR_PARALLEL,
    devices=[0, 1, 2, 3],
    tensor_parallel_size=4
)
strategy = create_parallel_strategy(config)
model = strategy.wrap_model(model)
```

### 3. ZeRO Optimizer (`optimizer.py`)
Memory-efficient optimizer with gradient and parameter sharding.

**ZeRO Stages:**
- **Stage 1**: Shard optimizer states
- **Stage 2**: Shard gradients
- **Stage 3**: Shard parameters

**Configuration:**
```python
zero_config = ZeROOptimizerConfig(
    stage=ZeROStage.STAGE_3,
    cpu_offload=True,
    offload_param=True,
    offload_optimizer=True,
    contiguous_grad_memory=True,
    overlap_communication=True
)

optimizer = create_zero_optimizer(
    model=model,
    optimizer_class=optim.AdamW,
    optimizer_kwargs={'lr': 1e-3, 'weight_decay': 0.01},
    zero_config=zero_config
)
```

### 4. Distributed DataLoader (`data_loader.py`)
Efficient, fault-tolerant data loading with load balancing.

**Features:**
- Automatic data sharding across workers
- Load balancing for performance optimization
- Fault tolerance with retry mechanisms
- Memory-efficient data preprocessing
- Caching for frequently accessed data

**Configuration:**
```python
data_config = DistributedDataLoaderConfig(
    batch_size=32,
    num_workers=4,
    shuffle=True,
    enable_fault_tolerance=True,
    enable_load_balancing=True,
    max_retries=3,
    persistent_workers=True,
    cache_preprocessed=True
)

train_loader = create_distributed_dataloader(dataset, data_config)
```

### 5. Checkpoint System (`checkpoint.py`)
Fault-tolerant, atomic checkpointing with automatic recovery.

**Features:**
- Atomic writes for data safety
- Incremental checkpointing for large models
- Automatic cleanup of old checkpoints
- Model sharding for memory efficiency
- Recovery from corrupted checkpoints

**Usage:**
```python
# Configure checkpointing
config = CheckpointConfig(
    checkpoint_dir="./checkpoints",
    enable_atomic_writes=True,
    verify_integrity=True,
    shard_model=True,
    max_shard_size_gb=4.0,
    backup_on_failure=True
)

# Save checkpoint
manager = DistributedCheckpointManager(config)
checkpoint_path = manager.save_checkpoint(
    model, optimizer, metadata=metadata, is_best=True
)

# Load checkpoint
metadata = manager.load_checkpoint(checkpoint_path, model, optimizer)
```

### 6. Monitoring System (`monitor.py`)
Real-time training monitoring with performance analysis and alerts.

**Metrics Tracked:**
- Training loss and accuracy
- Learning rate and gradient norms
- Throughput and step times
- Memory usage (CPU/GPU)
- Communication overhead
- System resource utilization

**Features:**
- Real-time dashboards
- Anomaly detection
- Performance alerts
- TensorBoard integration
- Weights & Biases integration

**Usage:**
```python
# Set up monitoring
monitor = create_training_monitor(
    metrics=['loss', 'throughput', 'memory_usage'],
    log_directory="./logs",
    enable_tensorboard=True
)

# Start monitoring
monitor.start_monitoring()

# Update metrics during training
monitor.update_metrics(current_metrics)

# Generate performance report
report = monitor.generate_performance_report()
print(report)
```

## 📖 Usage Examples

### Example 1: Basic Distributed Training
```python
import torch
import torch.nn as nn
from src.distributed import *

# Initialize distributed training
initialize_distributed_training()

# Create model
model = YourBrainInspiredModel()

# Configure trainer
config = DistributedTrainerConfig(
    zero_stage=2,
    mixed_precision=MixedPrecisionType.BF16
)
trainer = DistributedTrainer(model=model, config=config)

# Load data
data_config = DistributedDataLoaderConfig(batch_size=32)
train_loader = create_distributed_dataloader(train_dataset, data_config)

# Train
results = trainer.train(
    train_loader=train_loader,
    num_epochs=100,
    checkpoint_dir="./checkpoints"
)

cleanup_distributed_training()
```

### Example 2: Large Model Training
```python
# For models that don't fit on single GPU
from src.distributed.parallel import *

# Apply model parallelism
parallel_config = ParallelConfig(
    mode=ParallelMode.MODEL_PARALLEL,
    devices=[0, 1, 2, 3]
)
model = create_parallel_strategy(parallel_config).wrap_model(model)

# Configure ZeRO optimization
zero_config = ZeROOptimizerConfig(
    stage=ZeROStage.STAGE_3,
    cpu_offload=True
)
optimizer = create_zero_optimizer(model, optim.AdamW, {}, zero_config)

# Train with monitoring
monitor = create_training_monitor()
monitor.start_monitoring()

# ... training code ...

monitor.stop_monitoring()
```

### Example 3: Fault-Tolerant Training
```python
# Configure maximum fault tolerance
trainer_config = DistributedTrainerConfig(
    enable_recovery=True,
    checkpoint_frequency=5,
    health_check_interval=50,
    max_restarts=10
)

# Enable automatic recovery
checkpoint_manager = DistributedCheckpointManager(
    CheckpointConfig(
        enable_atomic_writes=True,
        verify_integrity=True,
        backup_on_failure=True
    )
)

# Training with automatic recovery
trainer = DistributedTrainer(model, trainer_config)

try:
    trainer.train(train_loader, num_epochs=100)
except Exception as e:
    logger.error(f"Training failed: {e}")
    # Automatic recovery will kick in
```

### Example 4: Custom Training Loop
```python
# Custom training with full control
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        outputs = trainer.engine(**batch)
        loss = outputs['loss']
        
        # Backward pass (handled by DeepSpeed)
        trainer.engine.backward(loss)
        trainer.engine.step()
        
        # Update monitoring
        metrics = TrainingMetrics(
            epoch=epoch,
            step=trainer.global_step,
            total_steps=total_steps,
            loss=loss.item(),
            # ... other metrics
        )
        monitor.update_metrics(metrics)
    
    # Save checkpoint
    trainer.save_checkpoint(f"./checkpoints/epoch_{epoch}")
    
    # Evaluate
    eval_metrics = trainer.evaluate(val_loader)
    logger.info(f"Epoch {epoch}: {eval_metrics}")
```

## ⚙️ Configuration

### DistributedTrainerConfig
```python
DistributedTrainerConfig(
    # ZeRO configuration
    zero_stage=3,                    # 0: disabled, 1: optimizer states, 2: gradients, 3: parameters
    offload_optimizer=False,         # Offload optimizer to CPU
    cpu_offload=False,               # Offload model parameters to CPU
    
    # Mixed precision
    mixed_precision=MixedPrecisionType.BF16,  # fp16, bf16, fp32, or none
    loss_scaling=True,
    initial_loss_scale=2**20,
    
    # Gradient management
    gradient_accumulation_steps=1,
    gradient_clip_val=1.0,
    clip_grad_norm=True,
    
    # Performance
    all_reduce_fp16=False,
    round_robin_bit=True,
    contiguous_grad_memory=False,
    
    # Communication
    communication_data_type="float32",
    sparse_gradients=False,
    
    # Fault tolerance
    enable_recovery=True,
    checkpoint_frequency=10,
    health_check_interval=100,
    max_restarts=3,
    
    # Monitoring
    log_frequency=10,
    save_frequency=10,
    eval_frequency=100,
    tensorboard_logging=True,
    wandb_logging=False,
    experiment_name="training_experiment",
    output_dir="./output"
)
```

### Performance Tuning Guidelines

**For Small Models (< 1B parameters):**
```python
config = DistributedTrainerConfig(
    zero_stage=1,  # Shard optimizer states only
    mixed_precision=MixedPrecisionType.BF16,
    gradient_accumulation_steps=1
)
```

**For Medium Models (1-10B parameters):**
```python
config = DistributedTrainerConfig(
    zero_stage=2,  # Shard gradients
    mixed_precision=MixedPrecisionType.BF16,
    gradient_accumulation_steps=4
)
```

**For Large Models (> 10B parameters):**
```python
config = DistributedTrainerConfig(
    zero_stage=3,  # Full sharding
    mixed_precision=MixedPrecisionType.BF16,
    gradient_accumulation_steps=8,
    cpu_offload=True
)
```

**For Memory-Constrained Environments:**
```python
config = DistributedTrainerConfig(
    zero_stage=3,
    cpu_offload=True,
    offload_optimizer=True,
    gradient_accumulation_steps=16
)
```

## 📊 Performance Optimization

### Memory Optimization
1. **Enable ZeRO-3** for models that exceed GPU memory
2. **Use CPU offloading** for very large models
3. **Increase gradient accumulation** to reduce batch size
4. **Enable gradient checkpointing** to trade computation for memory

### Communication Optimization
1. **Use FP16 communication** (`all_reduce_fp16=True`)
2. **Enable overlap communication** with computation
3. **Use smaller sub-group sizes** for parameter offloading
4. **Implement gradient bucketing** for efficient communication

### Data Loading Optimization
1. **Increase num_workers** based on CPU cores
2. **Enable persistent workers** for multi-epoch training
3. **Use pin_memory** for GPU training
4. **Enable data prefetching** for better throughput

### Model Optimization
1. **Use mixed precision training** (BF16 recommended)
2. **Apply tensor parallelism** for very large layers
3. **Use pipeline parallelism** for sequential models
4. **Implement proper initialization** for stable training

## 📈 Monitoring and Debugging

### Real-time Monitoring
```python
# Set up comprehensive monitoring
monitor = DistributedMonitoring()

# Add custom metrics
def custom_metrics_callback():
    return {
        'spike_activity': calculate_spike_activity(model),
        'routing_efficiency': calculate_routing_efficiency(model),
        'memory_efficiency': get_memory_efficiency()
    }

monitor.add_custom_metrics(custom_metrics_callback)

# Start monitoring
monitor.start_monitoring()
```

### Performance Analysis
```python
# Get detailed performance statistics
stats = trainer.get_performance_metrics()
print(f"Throughput: {stats['throughput']:.1f} samples/sec")
print(f"Memory usage: {stats['memory_allocated_gb']:.2f} GB")
print(f"Communication overhead: {stats['communication_time']:.2f}s")

# Generate performance report
report = monitor.generate_performance_report()
print(report)
```

### Debugging Common Issues

**High Communication Overhead:**
- Enable `overlap_communication=True`
- Reduce communication frequency
- Use gradient bucketing

**Out of Memory Errors:**
- Enable higher ZeRO stage
- Increase gradient accumulation
- Enable CPU offloading
- Use smaller batch sizes

**Slow Data Loading:**
- Increase `num_workers`
- Enable `persistent_workers`
- Check data pipeline bottlenecks
- Use data compression if appropriate

**Training Instability:**
- Use BF16 instead of FP16
- Adjust learning rate
- Enable gradient clipping
- Check for NaN gradients

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue: Distributed training not initializing**
```python
# Check environment variables
import os
print(f"RANK: {os.environ.get('RANK', 'Not set')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")

# Initialize explicitly
initialize_distributed_training(
    backend="nccl",  # or "gloo" for CPU
    rank=int(os.environ.get("RANK", "0")),
    world_size=int(os.environ.get("WORLD_SIZE", "1"))
)
```

**Issue: Model parameters not updating**
```python
# Check if gradients are computed
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"Parameter {name} has no gradient")

# Verify optimizer configuration
print(f"Optimizer type: {type(trainer.optimizer)}")
print(f"Learning rate: {trainer.optimizer.param_groups[0]['lr']}")
```

**Issue: High memory usage**
```python
# Check memory allocation
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
```

### Performance Profiling
```python
# Enable profiling
with torch.profiler.profile() as prof:
    trainer.train_epoch(train_loader)

# Print profiling results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## 📚 API Reference

### Core Classes

#### DistributedTrainer
Main class for distributed training orchestration.

**Methods:**
- `train(train_loader, num_epochs, ...)` - Main training loop
- `train_epoch(train_loader, epoch)` - Single epoch training
- `evaluate(eval_loader)` - Evaluation on validation data
- `save_checkpoint(path)` - Save training state
- `load_checkpoint(path)` - Load training state
- `get_performance_metrics()` - Get current performance stats

#### ZeROOptimizer
Memory-efficient optimizer with gradient and parameter sharding.

**Methods:**
- `step(closure)` - Perform optimization step
- `zero_grad()` - Zero all gradients
- `state_dict()` - Get optimizer state
- `load_state_dict(state_dict)` - Load optimizer state

#### DistributedDataLoader
Efficient distributed data loading with fault tolerance.

**Methods:**
- `__iter__()` - Iterate over data batches
- `get_comprehensive_stats()` - Get data loading statistics
- `get_throughput()` - Get data loading throughput

#### DistributedMonitoring
Real-time training monitoring and analysis.

**Methods:**
- `start_monitoring()` - Start monitoring system
- `stop_monitoring()` - Stop monitoring
- `update_metrics(metrics)` - Update training metrics
- `get_dashboard_data()` - Get data for dashboard
- `generate_performance_report()` - Generate performance report

### Utility Functions

#### Initialization
```python
initialize_distributed_training(backend="nccl", init_method="env://")
cleanup_distributed_training()
is_distributed()
get_world_size()
get_rank()
```

#### Model Parallelism
```python
parallelize_model(model, config=None, auto_detect=True)
create_parallel_strategy(config)
get_optimal_parallel_config(model, devices)
```

#### Data Loading
```python
create_distributed_dataloader(dataset, config, collate_fn=None)
shard_dataset(dataset, world_size, rank)
optimize_dataloader_config(dataset)
```

#### Checkpointing
```python
save_distributed_checkpoint(model, path, optimizer, scheduler, metadata)
load_distributed_checkpoint(path, model, optimizer, scheduler)
```

#### Monitoring
```python
create_training_monitor(metrics, log_directory, enable_tensorboard, enable_wandb)
setup_distributed_monitoring(model, optimizer, config)
```

## 🤝 Integration with Existing Pipeline

The distributed training system is designed to integrate seamlessly with the existing Step 3 training pipeline:

```python
# Extend existing training with distributed capabilities
from src.training import AdvancedTrainingPipeline
from src.distributed import enable_distributed_training

# Enable distributed training
distributed_pipeline = enable_distributed_training(
    training_pipeline=existing_pipeline,
    distributed_config={
        'zero_stage': 2,
        'mixed_precision': True,
        'gradient_accumulation_steps': 4
    }
)

# Train with distributed optimization
results = distributed_pipeline.train_distributed(
    datasets={'train': train_data, 'val': val_data},
    num_epochs=100
)
```

## 🔄 Backward Compatibility

The distributed training system maintains full backward compatibility with the existing training pipeline:

- All existing training methods work without modification
- Models can be trained in both single-GPU and distributed modes
- Checkpoints are compatible across different parallel configurations
- Monitoring integrates with existing logging systems

## 📝 License

This distributed training system is part of the mini-biai-1 framework and follows the same MIT License.

## 🙏 Acknowledgments

- Built on PyTorch's distributed training capabilities
- Integrates DeepSpeed's ZeRO optimization
- Inspired by biologically-plausible neural architectures
- Designed for production-scale AI model training

---

For more examples and detailed documentation, see the `example_usage.py` file and the comprehensive inline documentation in each module.
