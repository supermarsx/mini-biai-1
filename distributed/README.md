# Distributed Training Module

A comprehensive distributed training framework with DeepSpeed integration for efficient multi-GPU and multi-node training.

## Features

### Core Capabilities
- **DeepSpeed Integration**: Full support for ZeRO optimization stages 1-3
- **Multi-Parallelism**: Model, tensor, and pipeline parallelism support
- **Fault Tolerance**: Automatic checkpointing and recovery from failures
- **Real-time Monitoring**: Performance tracking and anomaly detection
- **Mixed Precision**: FP16 and BF16 training support
- **Dynamic Loss Scaling**: Automatic gradient scaling for numerical stability
- **Graceful Fallback**: Works without DeepSpeed for basic training

### Advanced Features
- **Activation Checkpointing**: Memory-efficient training for large models
- **Gradient Accumulation**: Support for large effective batch sizes
- **Elastic Training**: Resume from checkpoints on failures
- **Custom CUDA Kernels**: Optimized operations for specific hardware
- **Communication Optimization**: Efficient gradient synchronization
- **Robust Data Loading**: Fault-tolerant distributed data loading

## Architecture

```
distributed/
├── __init__.py              # Main module initialization
├── trainer.py               # Core training orchestration
├── optimizer.py             # ZeRO optimization implementation
├── parallel.py              # Model/tensor/pipeline parallelism
├── data_loader.py           # Distributed data loading
├── checkpoint.py            # Fault-tolerant checkpointing
├── monitor.py               # Real-time training monitoring
└── example_usage.py         # Usage examples and demos
```

## Quick Start

### Basic Distributed Training

```python
import torch
from distributed import setup_distributed_training

# Define your model
model = YourModel()

# Configure distributed training
config = {
    'batch_size': 8,
    'learning_rate': 1e-4,
    'zero_stage': 2,
    'fp16': True,
    'gradient_accumulation_steps': 1,
    'save_dir': './checkpoints',
    'save_interval': 1000
}

# Initialize distributed training
dist_training = setup_distributed_training(model, config)

# Start training
dist_training.train(epochs=10)
```

### Advanced Configuration

```python
config = {
    # Basic settings
    'batch_size': 32,
    'learning_rate': 1e-4,
    'gradient_accumulation_steps': 4,
    'gradient_clipping': 1.0,
    
    # ZeRO optimization
    'zero_stage': 3,
    'offload_optimizer': True,
    'offload_param': True,
    
    # Mixed precision
    'fp16': True,
    'loss_scale': None,  # Automatic
    
    # Memory optimization
    'activation_checkpointing': True,
    
    # Checkpointing
    'save_dir': './checkpoints',
    'save_interval': 500,
    'keep_checkpoints': 5,
    
    # Monitoring
    'log_interval': 10,
    'save_logs': True,
    'steps_per_print': 10,
    
    # Performance
    'wall_clock_breakdown': False,
    'profiling': False
}
```

## DeepSpeed Integration

### ZeRO Optimization Stages

The module supports all three ZeRO optimization stages:

**Stage 1**: Partition optimizer states across data parallel processes
```python
config = {'zero_stage': 1}
```

**Stage 2**: Partition optimizer states + gradients
```python
config = {
    'zero_stage': 2,
    'gather_16bit_weights_on_model_save': True
}
```

**Stage 3**: Partition optimizer states + gradients + model parameters
```python
config = {
    'zero_stage': 3,
    'offload_optimizer': True,
    'offload_param': True
}
```

### Memory Offloading

Offload optimizer states and parameters to CPU:
```python
config = {
    'zero_stage': 3,
    'offload_optimizer': {
        'device': 'cpu',
        'pin_memory': True
    },
    'offload_param': {
        'device': 'cpu',
        'pin_memory': True
    }
}
```

## Multi-Parallelism

### Model Parallelism

```python
from distributed.parallel import ModelParallel

# Split model across multiple GPUs
model_parallel = ModelParallel(
    model,
    devices=[0, 1, 2, 3],
    strategy='layer'
)
```

### Tensor Parallelism

```python
from distributed.parallel import TensorParallel

# Parallelize tensor operations
tensor_parallel = TensorParallel(
    model,
    devices=[0, 1],
    strategy='tensor'
)
```

### Pipeline Parallelism

```python
from distributed.parallel import PipelineParallel

# Split model into pipeline stages
pipeline_parallel = PipelineParallel(
    model,
    devices=[0, 1, 2, 3],
    stages=4
)
```

## Data Loading

### Custom Dataset

```python
from distributed.data_loader import DistributedDataLoader

# Create distributed data loader
data_loader = DistributedDataLoader(
    dataset=YourDataset(),
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

# Get distributed sampler
sampler = data_loader.get_sampler()
```

### Batch Size Configuration

Configure batch size per GPU:
```python
config = {
    'train_micro_batch_size_per_gpu': 8,  # Per GPU
    'train_batch_size': 64,              # Total across all GPUs
    'gradient_accumulation_steps': 2
}
```

## Checkpointing

### Automatic Checkpointing

```python
from distributed.checkpoint import CheckpointManager

manager = CheckpointManager(
    save_dir='./checkpoints',
    save_interval=1000,  # Save every 1000 steps
    keep_interval=5,     # Keep last 5 checkpoints
    async_save=True      # Save in background
)

# Save checkpoint
manager.save(engine, step, optimizer, lr_scheduler)

# Load checkpoint
manager.load(checkpoint_path, engine, optimizer, lr_scheduler)
```

### Custom Checkpoint Format

```python
def custom_save_function(engine, step, path):
    # Custom save logic
    checkpoint = {
        'engine': engine.state_dict(),
        'step': step,
        'custom_data': get_custom_data()
    }
    torch.save(checkpoint, path)

manager = CheckpointManager(
    save_function=custom_save_function
)
```

## Monitoring

### Real-time Metrics

```python
from distributed.monitor import TrainingMonitor

monitor = TrainingMonitor(
    log_interval=10,
    save_logs=True,
    log_dir='./logs'
)

# Track custom metrics
monitor.track_metric('loss', loss_value, step)
monitor.track_metric('accuracy', accuracy_value, step)

# Add custom callback
def custom_callback(metrics):
    if metrics['loss'] > threshold:
        logger.warning("High loss detected")

monitor.add_callback('loss_alert', custom_callback)
```

### System Resource Monitoring

```python
# Monitor GPU memory usage
monitor = TrainingMonitor(
    track_gpu_memory=True,
    track_cpu_memory=True,
    track_network=True
)

# Get current stats
stats = monitor.get_system_stats()
print(f"GPU Memory: {stats['gpu_memory']}")
print(f"CPU Memory: {stats['cpu_memory']}")
```

## Error Handling

### Fault Tolerance

```python
# Automatic recovery from failures
trainer = DistributedTrainer(
    engine=engine,
    optimizer=optimizer,
    monitor=monitor,
    config=config
)

# Enable fault tolerance
trainer.enable_fault_tolerance(
    auto_resume=True,
    max_retry_attempts=3,
    checkpoint_interval=500
)
```

### Error Recovery

```python
try:
    # Training code
    trainer.train(epochs, data_loader)
except Exception as e:
    logger.error(f"Training error: {e}")
    
    # Attempt recovery
    if trainer.has_checkpoint():
        trainer.restore_from_checkpoint()
    else:
        # Restart training from scratch
        trainer.reset()
```

## Performance Optimization

### Communication Optimization

```python
config = {
    'communication_data_type': 'fp16',  # Use FP16 for communication
    'allgather_base_size': 1000000,     # Allgather optimization
    'reduce_scatter_base_size': 1000000  # Reduce scatter optimization
}
```

### Custom CUDA Kernels

```python
# Register custom kernel
from distributed.parallel import register_custom_kernel

@register_custom_kernel('my_custom_op')
def my_custom_op(*args):
    # Custom CUDA implementation
    pass

# Use in model
output = custom_operation(input)
```

## Memory Optimization

### Gradient Checkpointing

```python
config = {
    'activation_checkpointing': {
        'enabled': True,
        'partition_activations': True,
        'cpu_checkpointing': True
    }
}
```

### Memory Statistics

```python
from distributed.monitor import MemoryProfiler

profiler = MemoryProfiler()
profiler.profile_training()

# Get memory usage report
report = profiler.get_memory_report()
print(report)
```

## Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 1 | Micro batch size per GPU |
| `learning_rate` | float | 1e-4 | Base learning rate |
| `zero_stage` | int | 2 | ZeRO optimization stage |
| `fp16` | bool | True | Enable FP16 training |
| `gradient_accumulation_steps` | int | 1 | Number of accumulation steps |
| `gradient_clipping` | float | 1.0 | Gradient clipping norm |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offload_optimizer` | bool | False | Offload optimizer to CPU |
| `offload_param` | bool | False | Offload parameters to CPU |
| `activation_checkpointing` | bool | False | Enable activation checkpointing |
| `save_interval` | int | 1000 | Checkpoint save interval |
| `log_interval` | int | 10 | Log interval for metrics |
| `steps_per_print` | int | 10 | Print frequency |

## Examples

See `example_usage.py` for comprehensive examples including:

- Basic distributed training
- ZeRO optimization setup
- Multi-node training
- Fault-tolerant training
- Custom monitoring
- Memory optimization
- Performance profiling

## Requirements

- Python 3.7+
- PyTorch 1.9+
- DeepSpeed 0.6+ (optional, for advanced features)
- CUDA 11.0+ (for GPU training)
- NCCL (for distributed training)

## Installation

```bash
pip install torch>=1.9.0
pip install deepspeed>=0.6.0  # Optional, for ZeRO optimization
pip install transformers  # Optional: for transformer models
```

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce batch size
- Enable activation checkpointing
- Use higher ZeRO stage (2 or 3)
- Enable memory offloading

**Slow Training**
- Check data loading pipeline
- Optimize communication backend
- Use gradient accumulation for larger effective batch size
- Profile training with `wall_clock_breakdown: true`

**Communication Errors**
- Verify NCCL is properly installed
- Check network connectivity for multi-node training
- Ensure proper process group initialization

**DeepSpeed Not Available**
- The module falls back to basic PyTorch distributed training
- Install DeepSpeed for advanced features: `pip install deepspeed`
- Check PyTorch and CUDA compatibility with DeepSpeed

### Performance Tuning

1. **Start with basic configuration**
2. **Profile training performance**
3. **Optimize based on bottlenecks**
4. **Fine-tune hyperparameters**

### Getting Help

- Check the examples in `example_usage.py`
- Review DeepSpeed documentation (when available)
- Open issues for bug reports and feature requests

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## License

This module is part of the MiniMax AI project. See the main repository for license information.