# Scripts and Utilities

This directory contains utility scripts, automation tools, and development helpers for the Brain-Inspired Modular AI Framework.

## Structure

```
scripts/
├── setup-dev.sh              # Development environment setup
├── run_all_demos.sh          # Run all demonstration scripts
├── advanced_demo.sh          # Advanced demonstration runner
├── affect_demo.sh            # Affect detection demo
├── auto_learning_demo.sh     # Auto-learning system demo
├── performance_demo.sh       # Performance benchmarking demo
├── quick_demo.sh             # Quick start demonstration
├── step2_demo.sh             # Step 2 integration demo
├── build_index.py            # Index building utility
├── CLI_README.md             # CLI documentation
├── CLI_QUICK_REFERENCE.md    # CLI quick reference
├── TESTING_SETUP_COMPLETION_SUMMARY.md
├── TEST_INFRASTRUCTURE_SUMMARY.md
└── step_2_specification.md
```

## Usage

### Development Setup

```bash
# Setup development environment
./scripts/setup-dev.sh

# Run comprehensive test suite
./scripts/run_all_demos.sh
```

### Demonstrations

```bash
# Quick demo
./scripts/quick_demo.sh

# Affect system demo
./scripts/affect_demo.sh

# Auto-learning demo
./scripts/auto_learning_demo.sh

# Performance benchmarking
./scripts/performance_demo.sh

# Advanced features demo
./scripts/advanced_demo.sh
```

### Index Management

```bash
# Build search index
python scripts/build_index.py
```

## CLI Tools

The framework provides several CLI utilities. See `CLI_README.md` for detailed documentation and `CLI_QUICK_REFERENCE.md` for quick command reference.

## Development Scripts

- `setup-dev.sh`: Sets up the development environment with all required dependencies
- `run_all_demos.sh`: Orchestrates running all demonstration scripts
- Each demo script is designed to be self-contained and demonstrate specific framework capabilities

## Adding New Scripts

When adding new utility scripts:

1. Follow shell scripting best practices
2. Include proper error handling
3. Add documentation comments
4. Make scripts executable (`chmod +x`)
5. Update this README if adding significant functionality