# API Reference

Welcome to the comprehensive API reference for mini-biai-1. This section provides detailed documentation for all public APIs, classes, functions, and modules.

## 📋 Overview

The mini-biai-1 API is organized into several core modules:

- **[Coordinator API](coordinator/index.md)** - Main orchestration and routing system
- **[Memory API](memory/index.md)** - Hierarchical memory systems and storage
- **[Language API](language/index.md)** - State Space Models and text processing
- **[Training API](training/index.md)** - Training pipelines and optimization
- **[Inference API](inference/index.md)** - Real-time inference and generation
- **[Expert API](experts/index.md)** - Multi-expert architecture components
- **[Learning API](learning/index.md)** - Online learning and adaptation
- **[Optimization API](optimization/index.md)** - Performance optimization tools

## 🚀 Quick API Examples

### Basic Usage

```python
import mini_biai_1
from mini_biai_1.coordinator import SpikingRouter
from mini_biai_1.memory import HierarchicalMemory
from mini_biai_1.language import LinearTextProcessor

# Initialize components
router = SpikingRouter()
memory = HierarchicalMemory(config.memory)
processor = LinearTextProcessor(config.language)

# Process input
result = processor.process("Hello, mini-biai-1!")
```

### Pipeline Creation

```python
from mini_biai_1.inference import create_pipeline

# Create a complete inference pipeline
pipeline = create_pipeline(config_path="configs/production.yaml")
result = pipeline.generate("The brain processes information through")
```

### Training Setup

```python
from mini_biai_1.training import RoutingTrainer

# Initialize trainer
trainer = RoutingTrainer(config.training)
trainer.train(dataset)
```

## 🔧 Configuration

All components accept configuration dictionaries or YAML files:

```python
from mini_biai_1.configs import load_config

config = load_config("configs/step1_base.yaml")
model = MiniBiAiCoordinator(config)
```

## 📚 API Documentation by Module

### Core Modules

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **coordinator** | System orchestration and routing | `SpikingRouter`, `MiniBiAiCoordinator` |
| **memory** | Hierarchical memory management | `HierarchicalMemory`, `ShortTermMemory` |
| **language** | SSM-based language processing | `LinearTextProcessor`, `SSMBackbone` |
| **training** | Model training and optimization | `RoutingTrainer`, `SyntheticRoutingDataset` |
| **inference** | Real-time inference and generation | `InferencePipeline`, `Generator` |
| **experts** | Multi-expert architecture | `LanguageExpert`, `VisionExpert`, `SymbolicExpert` |

### Utility Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **learning** | Online learning mechanisms | `OnlineLearner`, `STDP`, `CircuitBreaker` |
| **optimization** | Performance optimization | `BenchmarkProfiler`, `MemoryOptimizer` |
| **utils** | Utility functions and tools | `PerformanceProfiler`, `BenchmarkRunner` |

## 🔍 API Reference Details

### Auto-Generated Documentation

The API documentation is generated using:

- **Docstrings**: Detailed docstrings in all public APIs
- **Type Hints**: Full type annotation support
- **Examples**: Inline code examples for all major functions
- **Cross-References**: Links between related APIs

### Class Hierarchy

```python
# High-level class hierarchy
MiniBiAiCoordinator
├── SpikingRouter (routing)
├── HierarchicalMemory (storage)
├── LinearTextProcessor (language)
└── RoutingTrainer (training)

# Expert classes
LanguageExpert
VisionExpert  
SymbolicExpert
```

## 📖 Usage Patterns

### Common Patterns

1. **Initialize → Configure → Process**
   ```python
   component = ComponentClass(config)
   result = component.process(input_data)
   ```

2. **Pipeline Creation**
   ```python
   pipeline = Pipeline([component1, component2, component3])
   result = pipeline.run(input_data)
   ```

3. **Training Loop**
   ```python
   trainer = Trainer(config)
   for epoch in range(config.training.max_epochs):
       trainer.train_epoch(dataset)
       trainer.evaluate(validation_dataset)
   ```

## 🔧 API Best Practices

### Error Handling

```python
try:
    result = processor.process(text)
except ValueError as e:
    logger.error(f"Invalid input: {e}")
except RuntimeError as e:
    logger.error(f"Processing error: {e}")
```

### Resource Management

```python
# Use context managers for cleanup
with HierarchicalMemory(config) as memory:
    memory.store("key", "value")
    # Automatic cleanup on exit
```

### Configuration Validation

```python
from mini_biai_1.configs import validate_config

# Validate configuration before use
try:
    config = validate_config(config)
except ValidationError as e:
    raise ValueError(f"Invalid configuration: {e}")
```

## 🐛 API Testing

### Unit Tests

```python
import pytest
from mini_biai_1.language import LinearTextProcessor

def test_text_processor():
    processor = LinearTextProcessor(config)
    result = processor.process("Test input")
    assert result is not None
```

### Integration Tests

```python
def test_pipeline_integration():
    pipeline = create_pipeline("configs/test.yaml")
    result = pipeline.generate("Test prompt")
    assert len(result) > 0
```

## 📈 Versioning

### API Compatibility

- **Major Version**: Breaking changes (v1.0 → v2.0)
- **Minor Version**: New features, backward compatible (v1.0 → v1.1)
- **Patch Version**: Bug fixes (v1.0.0 → v1.0.1)

### Deprecation Policy

- Deprecations are announced 1 major version in advance
- Old APIs remain functional with deprecation warnings
- Removal occurs in the next major version

## 🚀 Performance Considerations

### Memory Management

- Use `with` statements for automatic cleanup
- Prefer generators for large datasets
- Monitor memory usage with `PerformanceProfiler`

### Computation Optimization

- Enable GPU acceleration when available
- Use batch processing for multiple inputs
- Leverage caching for repeated operations

## 🔗 Related Documentation

- [User Guides](../user-guides/index.md) - How to use mini-biai-1
- [Training Guide](../training/index.md) - Training workflows
- [Architecture Overview](../architecture/overview.md) - System design
- [Examples](../examples/index.md) - Code examples

---

*For detailed module-specific API documentation, use the navigation links in the table of contents.*