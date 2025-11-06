# Meta-Learning Framework

A comprehensive framework for meta-learning including Model-Agnostic Meta-Learning (MAML), few-shot learning, continual learning, and neural architecture search.

## Overview

This module provides implementations of various meta-learning algorithms and techniques:

- **MAML (Model-Agnostic Meta-Learning)**: Fast adaptation across tasks
- **Few-Shot Learning**: Learning with limited examples using prototypical and relation networks
- **Continual Learning**: Preventing catastrophic forgetting with EWC and progressive networks
- **Neural Architecture Search**: Automated architecture design for meta-learning
- **Adapters**: Parameter-efficient transfer learning with various adapter types
- **Adaptive Optimization**: Meta-learning enhanced optimization with Bayesian hyperparameter tuning

## Installation

```bash
pip install -e .
```

## Core Components

### 1. MAML (Model-Agnostic Meta-Learning)
Fast adaptation across different tasks with gradient-based meta-learning.

### 2. Few-Shot Learning
- **Prototypical Networks**: Class prototype-based classification
- **Relation Networks**: Learning to compare and classify
- **N-way K-shot Learning**: Support/query shot paradigms

### 3. Continual Learning
- **EWC (Elastic Weight Consolidation)**: Fisher information based forgetting prevention
- **Progressive Networks**: Lateral connections for knowledge retention
- **Experience Replay**: Memory-based continual learning

### 4. Neural Architecture Search
- **Random Search**: Baseline architecture search
- **Bayesian Optimization**: Efficient architecture search
- **DARTS**: Differentiable architecture search
- **Evolutionary Search**: Population-based optimization

### 5. Adapter Architectures
- **LoRA**: Low-rank adaptation
- **Linear Adapters**: Simple linear transformation adapters
- **Prefix Tuning**: Prepended context adapters
- **BitFit**: Bias-only fine-tuning
- **IA3**: Infusion adapters
- **Compacter**: Compressed adapters

### 6. Adaptive Optimization
- **Meta-learning Enhanced Optimization**: Automatic learning rate scheduling
- **Bayesian Hyperparameter Optimization**: Gaussian process based tuning
- **Performance Trend Analysis**: Learning curve analysis

## Usage Examples

### Basic MAML Training

```python
from meta_learning import MAML

# Initialize MAML with backbone model
maml = MAML(model=your_model, lr_inner=0.01, lr_outer=0.001)

# Meta-train on tasks
maml.meta_train(tasks_train)

# Fast adapt to new task
adapted_model = maml.adapt(task_support, num_steps=5)
```

### Few-Shot Learning

```python
from meta_learning import PrototypicalNetwork

# Create prototypical network
proto_net = PrototypicalNetwork(
    encoder=your_encoder,
    distance_metric='euclidean'
)

# Train on episodes
proto_net.train_episodes(training_episodes)

# Classify with support set
predictions = proto_net.classify(
    query_samples=query_data,
    support_samples=support_data,
    support_labels=support_labels
)
```

### Continual Learning

```python
from meta_learning import EWC, ProgressiveNetwork

# EWC for forgetting prevention
owa_learner = EWC(model=your_model, ewc_lambda=1000)
owa_learner.learn_task(task1_data, task1_labels)

# Add second task with EWC regularization
owa_learner.learn_task(
    task2_data, task2_labels,
    previous_tasks=[(task1_data, task1_labels)]
)

# Progressive network for incremental learning
progressive_net = ProgressiveNetwork(
    base_model=your_model,
    hidden_size=hidden_dim
)
progressive_net.add_new_task(task2_data, task2_labels)
```

### Neural Architecture Search

```python
from meta_learning import NeuralArchitectureSearch, DARTS

# Random search
nas_random = NeuralArchitectureSearch(
    search_space=your_search_space,
    strategy='random'
)
best_arch = nas_random.search(num_architectures=100)

# DARTS
darts_search = DARTS(
    model_size=your_model_size,
    num_operations=len(your_operations)
)
learned_arch = darts_search.search(
    train_data=training_data,
    val_data=validation_data
)
```

### Parameter-Efficient Transfer Learning

```python
from meta_learning import LoRAAdapter, PrefixTuningAdapter

# LoRA adaptation
lora_adapter = LoRAAdapter(
    model=your_model,
    r=16,  # rank
    lora_alpha=16,
    lora_dropout=0.1
)
lora_adapter.add_adapter()

# Prefix tuning for language models
prefix_tuning = PrefixTuningAdapter(
    model=your_language_model,
    prefix_length=5,
    hidden_size=model_hidden_size
)
prefix_tuning.add_prefix()
```

### Adaptive Optimization

```python
from meta_learning import AdaptiveOptimizer, BayesianOptimizer

# Meta-learning enhanced optimization
adaptive_opt = AdaptiveOptimizer(
    model=your_model,
    base_optimizer='adam',
    meta_learning_enabled=True
)

# Bayesian hyperparameter optimization
bayes_opt = BayesianOptimizer(
    objective='validation_accuracy',
    search_space=hyperparameter_space
)
best_params = bayes_opt.optimize(
    objective_function=your_objective,
    num_trials=50
)
```

## Advanced Features

### Tool Usage Optimization

The framework includes meta-learning for tool selection and optimization:

```python
from meta_learning import ToolMetaLearner

tool_learner = ToolMetaLearner()

# Learn optimal tool selection
optimal_tool = tool_learner.select_tool(
    task_description=your_task,
    available_tools=available_tools,
    task_complexity=complexity_level
)
```

### Multi-Modal Meta-Learning

Support for text, image, and audio modalities:

```python
from meta_learning import MultiModalAdapter

multimodal_adapter = MultiModalAdapter(
    text_model=text_encoder,
    image_model=vision_encoder,
    audio_model=audio_encoder,
    fusion_method='attention'
)
```

## Configuration

```python
# Configuration example
config = {
    'maml': {
        'lr_inner': 0.01,
        'lr_outer': 0.001,
        'num_inner_steps': 5
    },
    'ewc': {
        'lambda': 1000,
        ' fisher_scale': 1.0
    },
    'lora': {
        'r': 16,
        'alpha': 16,
        'dropout': 0.1
    }
}
```

## Performance Metrics

The framework provides comprehensive evaluation metrics:

- **Adaptation Speed**: How quickly models adapt to new tasks
- **Transfer Accuracy**: Performance on unseen tasks
- **Forgetting Rate**: Knowledge retention across tasks
- **Parameter Efficiency**: Number of trainable parameters
- **Computational Cost**: Training and inference efficiency

## Benchmarks

Benchmarked on standard datasets:
- **Omniglot**: Character recognition
- **Mini-ImageNet**: Few-shot image classification
- **CIFAR-100**: Multi-task continual learning
- **GLUE**: Natural language understanding
- **SuperGLUE**: Advanced language understanding

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{meta_learning_framework,
  title={Meta-Learning Framework: MAML, Few-Shot Learning, and Continual Learning},
  author={MiniMax Agent},
  year={2025},
  url={https://github.com/supermarsx/mini-biai-1}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions and support, please open an issue on GitHub.