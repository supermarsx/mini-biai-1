# Documentation Navigation Guide

Welcome to the comprehensive navigation guide for mini-biai-1 documentation. This guide will help you find exactly what you need.

## Quick Navigation Matrix

### For Different User Types

| User Type | Start Here | Next Steps | Advanced Topics |
|-----------|------------|------------|-----------------|
| **New Users** | [Quick Start](user-guides/quick-start.md) | [Installation](user-guides/installation.md) | [Examples](examples/index.md) |
| **Developers** | [Contributing Guide](developer-guides/contributing.md) | [API Reference](api/index.md) | [Architecture](architecture/overview.md) |
| **Researchers** | [Training Guide](training/index.md) | [Performance Guide](training/performance.md) | [Architecture Details](architecture/) |
| **Production Users** | [Configuration Guide](user-guides/configuration.md) | [Deployment Guide](training/cloud-deployment.md) | [Performance Optimization](training/performance.md) |

### Documentation Structure

```
docs/
├── index.md                                    # Main documentation index
├── README.md                                   # This navigation guide
│
├── api/                                       # API Documentation
│   ├── index.md                               # API overview
│   ├── coordinator/                           # Coordinator API
│   │   └── index.md                           # Routing and coordination
│   ├── memory/                                # Memory System API
│   │   └── index.md                           # Hierarchical memory
│   ├── training/                              # Training API
│   ├── language/                              # Language processing API
│   ├── inference/                             # Inference API
│   └── experts/                               # Expert modules API
│
├── user-guides/                               # User Documentation
│   ├── index.md                               # User guides overview
│   ├── installation.md                        # Installation instructions
│   ├── quick-start.md                         # Getting started tutorial
│   ├── configuration.md                       # Configuration reference
│   ├── cli-reference.md                       # Command-line interface
│   └── troubleshooting.md                     # Common issues and solutions
│
├── developer-guides/                          # Developer Resources
│   ├── index.md                               # Developer docs overview
│   ├── contributing.md                        # Contribution guidelines
│   ├── development-setup.md                   # Development environment
│   ├── architecture/                          # Detailed architecture
│   └── api-reference.md                       # Complete API reference
│
├── training/                                  # Training Documentation
│   ├── index.md                               # Training overview
│   ├── local-training.md                      # Local training setup
│   ├── cloud-deployment.md                    # Cloud training deployment
│   ├── hyperparameter-tuning.md               # Hyperparameter optimization
│   └── performance.md                         # Performance optimization
│
├── examples/                                  # Examples & Tutorials
│   ├── index.md                               # Examples overview
│   ├── basic-usage.md                         # Basic usage examples
│   ├── advanced-features.md                   # Advanced features
│   └── tutorials/                             # Step-by-step tutorials
│
└── architecture/                              # Architecture Documentation
    ├── overview.md                            # System architecture overview
    ├── memory-system.md                       # Memory architecture
    ├── snn-architecture.md                    # Spiking neural networks
    └── diagrams/                              # Architecture diagrams
```

## Getting Started Paths

### Path 1: Getting Started (30 minutes)
1. **[Installation Guide](user-guides/installation.md)** (10 min) - Set up your environment
2. **[Quick Start Tutorial](user-guides/quick-start.md)** (15 min) - Run your first examples
3. **[Basic Examples](examples/basic-usage.md)** (5 min) - Explore simple use cases

### Path 2: Understanding the System (2 hours)
1. **[Architecture Overview](architecture/overview.md)** (30 min) - System design principles
2. **[API Reference](api/index.md)** (45 min) - Understand the APIs
3. **[Memory System](architecture/memory-system.md)** (45 min) - Deep dive into memory

### Path 3: Training & Development (4 hours)
1. **[Training Guide](training/index.md)** (2 hours) - Comprehensive training
2. **[Configuration Guide](user-guides/configuration.md)** (30 min) - Customize system
3. **[Performance Optimization](training/performance.md)** (1.5 hours) - Advanced optimization

### Path 4: Advanced Usage (6+ hours)
1. **[Advanced Examples](examples/advanced-features.md)** (2 hours) - Complex use cases
2. **[Contributing Guide](developer-guides/contributing.md)** (2 hours) - Contribute to project
3. **[Architecture Deep Dive](architecture/training-pipeline.md)** (2 hours) - System internals

## Features

### 🧠 Brain-Inspired Architecture
- Multi-expert modular design with specialized components
- Spiking neural networks for neuromorphic computing
- Adaptive synaptic plasticity mechanisms
- Neural spike-based routing and coordination

### 💾 Memory Systems
- **Short-Term Memory (STM)**: Fast-access, limited capacity temporary storage
- **Long-Term Memory (LTM-FAISS)**: Persistent storage with vector indexing
- Hierarchical memory organization
- Efficient retrieval using FAISS similarity search

### 🎭 Affective Computing
- Real-time emotion detection
- Affect modulation and control
- Emotional state tracking
- Integration with learning systems

### 🧩 Expert Modules
- **Language Expert**: SSM-based language processing
- **Vision Expert**: Visual processing capabilities
- **Symbolic Expert**: Symbolic reasoning and logic
- Pluggable expert architecture

### 🔄 Online Learning
- Real-time adaptation
- Circuit breaker patterns for stability
- STDP (Spike-Timing-Dependent Plasticity) learning
- Replay buffer for experience replay
- Continuous performance optimization

## Installation

### Requirements
- Python 3.8+
- PyTorch >= 1.9.0
- CUDA-compatible GPU (optional, for accelerated training)

### Install mini-biai-1

```bash
# Clone the repository
git clone https://github.com/mini-biai-1/mini-biai-1.git
cd mini-biai-1

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage

```python
import mini_biai_1
from mini_biai_1.coordinator import MiniBiAiCoordinator
from mini_biai_1.configs import load_config

# Load configuration
config = load_config("configs/step1_base.yaml")

# Initialize the model
coordinator = MiniBiAi1Coordinator(config)

# Train the model
coordinator.train()

# Generate text
output = coordinator.generate("The brain processes information through")
print(output)
```

### Configuration

mini-biai-1 uses YAML configuration files to control all aspects of the model. See `configs/step1_base.yaml` for a complete example.

Key configuration sections:
- `memory`: Hierarchical memory architecture settings
- `snn`: Spiking neural network parameters
- `language`: Language model configuration
- `training`: Optimization and training parameters
- `inference`: Generation settings

## Project Structure

```
mini-biai-1/
├── src/                      # Source code
│   ├── affect/              # Emotion detection and modulation
│   ├── coordinator/         # Spiking coordinator and routing
│   ├── data_gatherer/       # Data collection and validation
│   ├── experts/             # Expert modules (language, vision, symbolic)
│   ├── inference/           # Inference and CLI interfaces
│   ├── interfaces/          # Internal interfaces
│   ├── language/            # SSM-based language processing
│   ├── learning/            # Online learning and adaptation
│   ├── memory/              # Memory systems (STM, LTM-FAISS)
│   ├── training/            # Training modules
│   └── utils/               # Utility functions
├── tests/                   # Test suite (unit, integration, performance)
├── docs/                    # Documentation
├── examples/                # Example scripts and demonstrations
├── benchmarks/              # Performance benchmarks and testing
├── deployment/              # Deployment configurations
├── scripts/                 # Utility scripts
├── configs/                 # Configuration files
├── data/                    # Data files and corpora
├── reports/                 # Development reports and summaries
├── logs/                    # Log files
├── Makefile                 # Build automation
├── pyproject.toml           # Project dependencies and metadata
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .gitignore              # Git ignore rules
├── CONTRIBUTING.md         # Contributing guidelines
└── README.md               # This file
```

## Architecture

### Memory Hierarchy

mini-biai-1 implements a three-tier memory system:

1. **Working Memory**: Fast-access, limited capacity (configurable)
2. **Episodic Memory**: Context-rich storage with similarity-based retrieval
3. **Semantic Memory**: Distributed knowledge representation

### Spiking Neural Networks

The SNN components implement:
- Leaky Integrate-and-Fire neuron models
- STDP (Spike-Timing-Dependent Plasticity) learning
- Temporal coding mechanisms
- Bio-realistic connectivity patterns

### Language Processing

Language capabilities include:
- Transformer-style attention with spiking variants
- Token-level and sentence-level processing
- Context-aware generation
- Multi-modal understanding (future feature)

## Usage Examples

### Training a Model

```python
from mini_biai_1.training import Trainer
from mini_biai_1.configs import load_config

config = load_config("configs/step1_base.yaml")
trainer = Trainer(config)

# Train the model
trainer.train()
```

### Memory Operations

```python
from mini_biai_1.memory import HierarchicalMemory

memory = HierarchicalMemory(config.memory)

# Store information
memory.working_memory.store("context", "information")
memory.episodic_memory.store("episode", {"context": "data", "timestamp": 123})

# Retrieve information
retrieved = memory.episodic_memory.retrieve("query", top_k=5)
```

### Generating Text

```python
from mini_biai_1.inference import Generator

generator = Generator(config.inference)
output = generator.generate(
    prompt="The brain processes information through",
    max_length=100,
    temperature=0.7,
    top_p=0.9
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=mini_biai_1 tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
```

### Code Style

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs/
make html
```

## Configuration Guide

### Memory Configuration

```yaml
memory:
  working_memory:
    capacity: 1000
    decay_rate: 0.01
  episodic_memory:
    capacity: 10000
    indexing_method: "faiss"
  semantic_memory:
    vector_dim: 768
```

### Training Configuration

```yaml
training:
  optimizer: "adamw"
  learning_rate: 5e-5
  max_epochs: 10
  batch_size: 32
```

### SNN Configuration

```yaml
snn:
  threshold: 1.0
  time_steps: 8
  plasticity_enabled: true
  learning_rate: 0.001
```

## Performance

mini-biai-1 is designed for both research and production use:

- **Memory Efficient**: Optimized memory usage with gradient checkpointing
- **Scalable**: Supports distributed training across multiple GPUs
- **Fast Inference**: Optimized inference pipeline with caching
- **Hardware Agnostic**: Runs on CPU, GPU, and neuromorphic hardware

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/mini-biai-1/mini-biai-1.git
cd mini-biai-1

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install
```

### Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation for new features
- Use type hints for all public APIs
- Write descriptive commit messages

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citations

If you use mini-biai-1 in your research, please cite:

```bibtex
@software{minibiai1_2024,
  title={mini-biai-1: A Brain-Inspired Computational Model},
  author={mini-biai-1 Team},
  year={2024},
  url={https://github.com/mini-biai-1/mini-biai-1}
}
```

## Acknowledgments

- SpikingJelly team for the excellent spiking neural network framework
- Hugging Face team for the Transformers library
- FAISS team for efficient similarity search and clustering
- The neuromorphic computing research community

## Roadmap

### v0.2.0 (Q2 2024)
- [ ] Multi-modal support (vision + text)
- [ ] Advanced plasticity mechanisms
- [ ] Hardware acceleration for neuromorphic chips
- [ ] Improved memory consolidation algorithms

### v0.3.0 (Q3 2024)
- [ ] Meta-learning capabilities
- [ ] Few-shot learning benchmarks
- [ ] Interactive learning environments
- [ ] Graph neural network integration

### v1.0.0 (Q4 2024)
- [ ] Production deployment tools
- [ ] Comprehensive benchmark suite
- [ ] Real-time inference optimization
- [ ] Extended documentation and tutorials

## Support

- **Documentation**: [https://mini-biai-1.readthedocs.io](https://mini-biai-1.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/mini-biai-1/mini-biai-1/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mini-biai-1/mini-biai-1/discussions)
- **Email**: team@mini-biai-1.org

## FAQ

**Q: What hardware do I need to run mini-biai-1?**
A: mini-biai-1 can run on any system with Python 3.8+, but training is significantly faster with a CUDA-compatible GPU.

**Q: How is mini-biai-1 different from standard Transformers?**
A: mini-biai-1 incorporates biological principles like spiking neurons, synaptic plasticity, and hierarchical memory systems that go beyond traditional transformer architectures.

**Q: Can I use mini-biai-1 for my research?**
A: Yes, mini-biai-1 is designed for research use. Please check the license and cite the project if you use it in academic work.

**Q: How do I contribute?**
A: Check our [Contributing Guide](CONTRIBUTING.md) and look for issues labeled "good first issue" to get started.

---

**mini-biai-1 Team** - Making mini-biai-1-inspired computing accessible to everyone.

---

## CLI Interface and Demo

mini-biai-1 includes a comprehensive command-line interface for corpus indexing and intelligent query processing. The CLI provides easy-to-use commands for building search indexes and querying the system's knowledge base.

### Quick Demo

Run the complete demonstration:

```bash
bash scripts/quick_demo.sh
```

This will create sample corpus files, build a searchable index, and demonstrate the query functionality.

### CLI Commands

- **Build Index**: `python3 src/inference/cli.py build-index --corpus data/corpus`
- **Query System**: `python3 src/inference/cli.py query --query-text "machine learning"`
- **Check Status**: `python3 src/inference/cli.py status`
- **Run Demo**: `python3 src/inference/cli.py demo`

### Features

- Multiple output formats (text, JSON, CSV)
- Configurable chunking and indexing parameters
- Comprehensive error handling and logging
- Reproducible builds with metadata tracking
- Support for various corpus file formats

See the [CLI Documentation](scripts/CLI_README.md) for complete usage instructions.