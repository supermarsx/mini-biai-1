# Training Guide

Welcome to the comprehensive training documentation for mini-biai-1. This guide covers all aspects of training, from basic concepts to advanced optimization techniques.

## Table of Contents

- [Training Overview](#training-overview)
- [Quick Start Training](#quick-start-training)
- [Training Configurations](#training-configurations)
- [Local Training](#local-training)
- [Cloud Deployment](#cloud-deployment)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Advanced Training Techniques](#advanced-training-techniques)
- [Performance Optimization](#performance-optimization)

## Training Overview

### Architecture

Mini-biai-1 uses a sophisticated training architecture:

- **Multi-Expert Routing**: Specialized experts for different tasks
- **Spiking Neural Networks**: Neuromorphic learning mechanisms
- **Hierarchical Memory**: Working, episodic, and semantic memory systems
- **Online Learning**: Real-time adaptation and plasticity
- **Distributed Training**: Multi-GPU and multi-node support

### Training Components

```python
# Core training components
from mini_biai_1.training import (
    RoutingTrainer,           # Main training orchestrator
    SyntheticRoutingDataset,  # Dataset generation
    CurriculumLearning,       # Progressive training
    ParameterEfficientTraining, # LoRA, adapters
    HyperparameterTuning     # Automated optimization
)
```

### Training Types

1. **Routing Training**: Learn optimal expert routing
2. **Language Model Training**: SSM-based language processing
3. **Multi-Task Training**: Joint training on multiple tasks
4. **Continual Learning**: Online adaptation to new data
5. **Meta-Learning**: Learning to learn new tasks quickly

## Quick Start Training

### Minimal Training Example

```python
#!/usr/bin/env python3
"""
Quick start training example
"""

from mini_biai_1.training import RoutingTrainer
from mini_biai_1.configs import load_config

def quick_training():
    """Quick training example."""
    
    # Load configuration
    config = load_config("configs/quick_training.yaml")
    
    # Create trainer
    trainer = RoutingTrainer(config)
    
    # Generate synthetic dataset
    dataset = trainer.create_synthetic_dataset(
        num_samples=1000,
        input_dim=100,
        num_experts=4
    )
    
    # Train model
    results = trainer.train(
        dataset,
        max_epochs=5,
        validation_split=0.2
    )
    
    print(f"Training completed!")
    print(f"Final loss: {results.final_loss:.4f}")
    print(f"Accuracy: {results.final_accuracy:.4f}")
    
    # Save model
    trainer.save_checkpoint("quick_model.pt")
    
    return results

if __name__ == "__main__":
    quick_training()
```

### Configuration File

Create `configs/quick_training.yaml`:

```yaml
training:
  # Basic training parameters
  max_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.01
  
  # Validation
  validation_split: 0.2
  validation_frequency: 1
  
  # Optimization
  optimizer: "adamw"
  scheduler: "cosine"
  gradient_clipping: 1.0
  
  # Memory management
  mixed_precision: true
  gradient_checkpointing: false
  
  # Routing parameters
  routing_loss_weight: 0.5
  expert_capacity: 100

model:
  # Model architecture
  hidden_dim: 512
  num_layers: 6
  num_experts: 4
  expert_dim: 256
  
  # Spiking parameters
  spike_threshold: 1.0
  time_steps: 8
  plasticity_enabled: true

logging:
  level: "INFO"
  wandb_enabled: false
  save_frequency: 5
```

## Training Configurations

### Basic Training Configuration

```python
from mini_biai_1.training import TrainingConfig

# Create training configuration
config = TrainingConfig(
    # Core parameters
    max_epochs=20,
    batch_size=64,
    learning_rate=0.001,
    
    # Optimization
    optimizer="adamw",
    weight_decay=0.01,
    gradient_clipping=1.0,
    
    # Learning rate scheduling
    scheduler="cosine",
    warmup_steps=1000,
    
    # Memory management
    mixed_precision=True,
    gradient_checkpointing=True,
    
    # Validation
    validation_split=0.2,
    early_stopping=True,
    patience=5,
    
    # Logging and checkpoints
    save_frequency=5,
    log_frequency=100,
    checkpoint_frequency=10
)
```

### Advanced Configuration Options

```yaml
training:
  # Advanced optimization
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
    eps: 1e-8
    
  scheduler:
    type: "cosine_with_restarts"
    t_max: 10000
    eta_min: 1e-6
    
  # Regularization
  dropout: 0.1
  label_smoothing: 0.1
  weight_decay: 0.01
  
  # Data augmentation
  augmentation:
    enabled: true
    noise_level: 0.01
    mask_probability: 0.15
    
  # Distributed training
  distributed:
    enabled: false
    backend: "nccl"
    world_size: 4
    
  # Memory optimization
  memory:
    offload_to_cpu: true
    gradient_accumulation_steps: 4
    
  # Routing-specific
  routing:
    load_balancing_weight: 0.01
    expert_dropout: 0.1
    capacity_factor: 1.25
```

## Local Training

### Single GPU Training

```python
def train_single_gpu():
    """Train on single GPU."""
    
    # Check GPU availability
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load configuration
    config = load_config("configs/single_gpu.yaml")
    config.device = device
    
    # Initialize trainer
    trainer = RoutingTrainer(config)
    
    # Create dataset
    dataset = trainer.create_dataset("path/to/data")
    
    # Train model
    trainer.train(dataset)
    
    # Evaluate
    eval_results = trainer.evaluate(dataset.test)
    print(f"Test accuracy: {eval_results.accuracy:.4f}")
```

### Multi-GPU Training

```python
def train_multi_gpu():
    """Train on multiple GPUs."""
    
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    
    # Get local rank
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    # Load configuration
    config = load_config("configs/multi_gpu.yaml")
    config.local_rank = local_rank
    
    # Initialize model
    model = create_model(config)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    
    # Initialize trainer with distributed model
    trainer = RoutingTrainer(config)
    trainer.set_model(model)
    
    # Train with distributed sampler
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    # Train
    trainer.train(dataloader)
    
    # Cleanup
    dist.destroy_process_group()
```

### CPU Training

```python
def train_cpu_only():
    """Train on CPU only."""
    
    # Configure for CPU training
    config = load_config("configs/cpu.yaml")
    config.device = "cpu"
    config.mixed_precision = False
    config.batch_size = 16  # Smaller batch size for CPU
    
    # Enable CPU optimizations
    torch.set_num_threads(8)
    
    trainer = RoutingTrainer(config)
    
    # Train with smaller batches
    results = trainer.train(dataset)
    
    return results
```

## Cloud Deployment

### AWS EC2 Training

```bash
#!/bin/bash
# AWS EC2 training script

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type p3.2xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --user-data file://user-data.sh

# user-data.sh content:
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git
pip3 install mini-biai-1
git clone https://github.com/mini-biai-1/mini-biai-1.git
cd mini-biai-1
pip3 install -e ".[dev]"
python3 training/launch_aws_training.py
```

### Google Cloud Training

```python
# gcp_training.py
import os
from google.cloud import storage
from mini_biai_1.training import CloudTrainer

def gcp_training():
    """Train on Google Cloud Platform."""
    
    # Configure GCP
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp-credentials.json"
    client = storage.Client()
    
    # Initialize cloud trainer
    trainer = CloudTrainer(
        platform="gcp",
        instance_type="n1-standard-8",
        accelerator="nvidia-tesla-v100"
    )
    
    # Upload data to GCS
    trainer.upload_data("local/data/", "gs://mini-biai-1-data/")
    
    # Launch training job
    job_config = {
        "max_steps": 10000,
        "master_config": {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_V100"
        }
    }
    
    job = trainer.submit_training_job(
        job_config=job_config,
        data_path="gs://mini-biai-1-data/",
        output_path="gs://mini-biai-1-models/"
    )
    
    # Monitor job
    trainer.wait_for_completion(job)
    
    # Download results
    trainer.download_results("gs://mini-biai-1-models/", "./results/")
```

### Azure Training

```python
# azure_training.py
from azureml.core import Workspace, Experiment, Environment
from mini_biai_1.training import AzureTrainer

def azure_training():
    """Train on Azure Machine Learning."""
    
    # Initialize workspace
    ws = Workspace.from_config()
    
    # Create experiment
    experiment = Experiment(workspace=ws, name="mini-biai-1-training")
    
    # Configure environment
    env = Environment.from_conda_specification(
        name="mini-biai-1-env",
        file_path="environment.yml"
    )
    
    # Configure compute target
    compute_target = ws.compute_targets["gpu-cluster"]
    
    # Initialize trainer
    trainer = AzureTrainer(
        workspace=ws,
        experiment=experiment,
        compute_target=compute_target,
        environment=env
    )
    
    # Submit training job
    config = {
        "max_epochs": 20,
        "batch_size": 32,
        "learning_rate": 0.001
    }
    
    run = trainer.submit_training_job(config)
    run.wait_for_completion()
    
    # Download model
    trainer.download_model(run, "./azure_model/")
```

## Hyperparameter Tuning

### Basic Hyperparameter Tuning

```python
from mini_biai_1.training import HyperparameterTuner, ParameterGrid

def basic_hyperparameter_tuning():
    """Basic hyperparameter tuning example."""
    
    # Define search space
    search_space = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64, 128],
        "num_experts": [2, 4, 8],
        "hidden_dim": [256, 512, 1024]
    }
    
    # Initialize tuner
    tuner = HyperparameterTuner(
        search_space=search_space,
        optimization_metric="accuracy",
        maximize=True,
        max_trials=20
    )
    
    # Run tuning
    best_params, best_score = tuner.train_and_evaluate(
        dataset=dataset,
        max_trials=20,
        max_epochs=5
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    # Train final model with best parameters
    final_trainer = RoutingTrainer(best_params)
    final_results = final_trainer.train(dataset)
    
    return best_params, final_results
```

### Advanced Hyperparameter Optimization

```python
from mini_biai_1.training import BayesianOptimizer

def advanced_hyperparameter_tuning():
    """Advanced tuning with Bayesian optimization."""
    
    # Define continuous search space
    continuous_space = {
        "learning_rate": (0.00001, 0.1, "log_uniform"),
        "dropout_rate": (0.0, 0.5, "uniform"),
        "weight_decay": (1e-8, 1e-2, "log_uniform"),
        "expert_dropout": (0.0, 0.3, "uniform")
    }
    
    # Define categorical search space
    categorical_space = {
        "optimizer": ["adamw", "adam", "sgd"],
        "scheduler": ["cosine", "linear", "exponential"],
        "activation": ["gelu", "relu", "swish"]
    }
    
    # Initialize Bayesian optimizer
    optimizer = BayesianOptimizer(
        continuous_space=continuous_space,
        categorical_space=categorical_space,
        n_initial_points=10,
        acquisition_function="expected_improvement"
    )
    
    # Optimization objective
    def objective(params):
        config = load_config("configs/base.yaml")
        config.update(params)
        
        trainer = RoutingTrainer(config)
        results = trainer.train(dataset, max_epochs=3)
        
        return results.validation_accuracy
    
    # Run optimization
    best_params, best_value = optimizer.optimize(
        objective,
        n_calls=50
    )
    
    return best_params, best_value
```

### Automated Hyperparameter Tuning

```python
def automated_hyperparameter_tuning():
    """Automated tuning with early stopping and pruning."""
    
    from mini_biai_1.training import OptunaTuner
    
    # Initialize Optuna tuner
    tuner = OptunaTuner(
        study_name="mini-biai-1-tuning",
        storage="sqlite:///tuning.db",
        direction="maximize"
    )
    
    # Define objective function
    def objective(trial):
        # Sample hyperparameters
        params = {
            "learning_rate": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "num_experts": trial.suggest_int("num_experts", 2, 8),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [256, 512, 1024]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)
        }
        
        try:
            # Create configuration
            config = load_config("configs/base.yaml")
            config.update(params)
            
            # Create trainer
            trainer = RoutingTrainer(config)
            
            # Train with early stopping
            results = trainer.train(
                dataset,
                max_epochs=10,
                validation_split=0.2,
                early_stopping=True,
                patience=3
            )
            
            # Report intermediate value for pruning
            trial.report(results.validation_accuracy, 5)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return results.validation_accuracy
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0
    
    # Run optimization
    study = tuner.optimize(objective, n_trials=100)
    
    print(f"Best trial: {study.best_trial}")
    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params
```

## Advanced Training Techniques

### Curriculum Learning

```python
from mini_biai_1.training import CurriculumLearning

def curriculum_training():
    """Train with curriculum learning."""
    
    # Define curriculum stages
    curriculum = CurriculumLearning(
        stages=[
            {
                "difficulty": "easy",
                "batch_size": 64,
                "learning_rate": 0.001,
                "epochs": 5
            },
            {
                "difficulty": "medium", 
                "batch_size": 32,
                "learning_rate": 0.0005,
                "epochs": 10
            },
            {
                "difficulty": "hard",
                "batch_size": 16,
                "learning_rate": 0.0001,
                "epochs": 15
            }
        ]
    )
    
    # Initialize trainer
    trainer = RoutingTrainer(config)
    
    # Train with curriculum
    results = trainer.train_with_curriculum(dataset, curriculum)
    
    return results
```

### Multi-Task Training

```python
from mini_biai_1.training import MultiTaskTrainer

def multi_task_training():
    """Train on multiple tasks simultaneously."""
    
    # Define tasks
    tasks = {
        "routing": {
            "dataset": routing_dataset,
            "weight": 1.0,
            "metric": "accuracy"
        },
        "language": {
            "dataset": language_dataset,
            "weight": 0.5,
            "metric": "perplexity"
        },
        "memory": {
            "dataset": memory_dataset,
            "weight": 0.3,
            "metric": "retrieval_accuracy"
        }
    }
    
    # Initialize multi-task trainer
    trainer = MultiTaskTrainer(config, tasks)
    
    # Train on all tasks
    results = trainer.train(
        max_epochs=20,
        task_weights_schedule="dynamic",
        gradient_accumulation=True
    )
    
    return results
```

### Parameter-Efficient Training

```python
from mini_biai_1.training import LoRATrainer, AdapterTrainer

def parameter_efficient_training():
    """Train using LoRA or adapters."""
    
    # LoRA configuration
    lora_config = {
        "r": 8,  # rank
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "target_modules": ["query", "value", "key"]
    }
    
    # Train with LoRA
    trainer = LoRATrainer(config, lora_config)
    lora_results = trainer.train(dataset)
    
    # Adapter configuration
    adapter_config = {
        "hidden_size": 256,
        "bottleneck_size": 64,
        "adapter_layers": "all"
    }
    
    # Train with adapters
    adapter_trainer = AdapterTrainer(config, adapter_config)
    adapter_results = adapter_trainer.train(dataset)
    
    return lora_results, adapter_results
```

## Performance Optimization

### Memory Optimization

```python
from mini_biai_1.training import MemoryOptimizedTrainer

def memory_optimized_training():
    """Training with memory optimizations."""
    
    # Memory optimization configuration
    memory_config = {
        "mixed_precision": True,
        "gradient_checkpointing": True,
        "cpu_offloading": True,
        "activation_checkpointing": True,
        "data_loader_num_workers": 4,
        "pin_memory": True
    }
    
    # Initialize optimized trainer
    trainer = MemoryOptimizedTrainer(config, memory_config)
    
    # Monitor memory usage
    with trainer.monitor_memory() as monitor:
        results = trainer.train(dataset)
        
        print(f"Peak memory usage: {monitor.peak_memory:.2f}GB")
        print(f"Average GPU utilization: {monitor.avg_gpu_util:.1f}%")
    
    return results
```

### Distributed Training Optimization

```python
def optimized_distributed_training():
    """Optimized distributed training setup."""
    
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    
    # Distributed training configuration
    dist_config = {
        "backend": "nccl",
        "init_method": "env://",
        "world_size": torch.cuda.device_count(),
        "rank": int(os.environ["RANK"])
    }
    
    # Initialize distributed training
    dist.init_process_group(**dist_config)
    
    # Create model with ZeRO optimization
    from deepspeed.ops.adam import FusedAdam
    from deepspeed.runtime.zero.partition_parameters import zero_param_*
    
    model = create_model(config)
    model = DistributedDataParallel(model)
    
    # Initialize DeepSpeed trainer
    trainer = DeepSpeedTrainer(
        model=model,
        config=deepspeed_config
    )
    
    # Train
    results = trainer.train(dataset)
    
    # Cleanup
    dist.destroy_process_group()
    
    return results
```

### Performance Monitoring

```python
from mini_biai_1.training import PerformanceMonitor

def monitor_training_performance():
    """Monitor training performance metrics."""
    
    trainer = RoutingTrainer(config)
    
    # Create performance monitor
    monitor = PerformanceMonitor(
        metrics=[
            "throughput",
            "memory_usage",
            "gpu_utilization", 
            "gradient_norm",
            "learning_rate"
        ],
        log_frequency=100,
        save_plots=True
    )
    
    # Train with monitoring
    with trainer.monitor_performance(monitor) as (trainer, monitor):
        results = trainer.train(dataset)
        
        # Access monitoring results
        print(f"Training throughput: {monitor.throughput:.2f} samples/sec")
        print(f"Average memory: {monitor.avg_memory:.2f}GB")
        print(f"Training stability: {monitor.stability_score:.4f}")
    
    return results
```

---

## Training Tips and Best Practices

### General Guidelines

1. **Start Simple**: Begin with basic configurations
2. **Monitor Resources**: Watch GPU memory and utilization
3. **Validate Early**: Implement validation early in training
4. **Save Checkpoints**: Regular checkpoints for recovery
5. **Profile Performance**: Use built-in profiling tools

### Common Issues

- **Out of Memory**: Reduce batch size, enable gradient checkpointing
- **Slow Training**: Use mixed precision, increase batch size
- **Poor Convergence**: Check learning rate, adjust optimizer
- **Overfitting**: Add regularization, increase dataset size

### Performance Checklist

- [ ] Mixed precision training enabled
- [ ] Gradient clipping configured
- [ ] Learning rate scheduling set up
- [ ] Memory monitoring active
- [ ] Checkpointing enabled
- [ ] Validation loop implemented
- [ ] Logging configured
- [ ] Performance metrics tracked

---

*For detailed API documentation, see the [Training API Reference](../api/training/index.md). For troubleshooting, check the [Troubleshooting Guide](../user-guides/troubleshooting.md).*