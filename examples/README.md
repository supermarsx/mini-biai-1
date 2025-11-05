# Examples

This directory contains working examples, demos, and demonstration scripts for the mini-biai-1 system.

## Overview

Examples are organized by functionality and complexity level. Each example demonstrates specific features or capabilities of the system.

## Directory Structure

```
examples/
├── README.md                      # This file
├── advanced_memory_optimization_demo.py    # Advanced memory optimization
├── optimization_demo.py          # Performance optimization example
├── auto_learning_demo.py         # Auto-learning system demonstration
├── auto_learning_demo_working.py # Working auto-learning demo
├── comprehensive_auto_learning_demo.py     # Complete auto-learning example
├── comprehensive_auto_learning_tests.py    # Comprehensive test suite
├── performance_demo.py           # Performance benchmarking demo
├── analyze_test_results.py       # Test results analysis tool
├── dev-helper.py                 # Developer helper utilities
└── browser/                      # Browser automation examples
    ├── global_browser.py        # Browser interaction utilities
    └── browser_extension/       # Browser extension components
```

## Getting Started

### Quick Start

1. **Basic Demo**: Run the auto-learning demo to see the system in action:
   ```bash
   python examples/auto_learning_demo.py
   ```

2. **Performance Testing**: Use the performance demo to benchmark the system:
   ```bash
   python examples/performance_demo.py
   ```

3. **Comprehensive Testing**: Run the full test suite:
   ```bash
   python examples/comprehensive_auto_learning_tests.py
   ```

### Advanced Examples

1. **Working Auto-Learning**: A refined version with improved stability:
   ```bash
   python examples/auto_learning_demo_working.py
   ```

2. **Complete System Demo**: Comprehensive demonstration of all features:
   ```bash
   python examples/comprehensive_auto_learning_demo.py
   ```

3. **Browser Automation**: Browser-based interaction examples:
   ```bash
   python examples/browser/global_browser.py
   ```

## Example Categories

### Auto-Learning Demonstrations

- **`auto_learning_demo.py`**: Basic auto-learning system showcase
- **`auto_learning_demo_working.py`**: Stable, tested auto-learning implementation
- **`comprehensive_auto_learning_demo.py`**: Complete feature demonstration

### Performance Testing

- **`performance_demo.py`**: System performance benchmarking
- **`analyze_test_results.py`**: Test results analysis and reporting
- **`comprehensive_auto_learning_tests.py`**: Full system validation

### Developer Tools

- **`dev-helper.py`**: Helper utilities for development
- **`browser/`**: Browser automation and testing utilities

## Prerequisites

Before running examples, ensure you have:

1. **Installed Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development examples
   ```

2. **Configuration**: Set up your configuration files in `configs/`

3. **Data**: Ensure necessary data files are available in `data/`

## Running Examples

### Individual Examples

Each example can be run independently. Most examples accept command-line arguments:

```bash
python examples/auto_learning_demo.py --config configs/quickstart.yaml --output results.json
```

### Batch Execution

Use the provided shell scripts in `scripts/` for batch execution:

```bash
scripts/quick_demo.sh          # Run quick demo
scripts/auto_learning_demo.sh  # Run auto-learning demo
scripts/advanced_demo.sh       # Run advanced demo
scripts/performance_demo.sh    # Run performance tests
```

## Configuration

Examples typically use configuration files from the `configs/` directory. Available configurations:

- `quickstart.yaml`: Basic configuration for testing
- `step1_base.yaml`: Step 1 baseline configuration
- `step2_base.yaml`: Step 2 baseline configuration
- `test_*.yaml`: Test-specific configurations

## Output

Examples generate various outputs:

- **Log Files**: Saved in `logs/` directory
- **Results**: JSON/CSV files in `demo_results/`
- **Test Reports**: Generated test artifacts

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration Errors**:
   - Check config file syntax
   - Verify file paths
   - Ensure required fields are present

3. **Data Issues**:
   - Verify data files exist in `data/`
   - Check data format compatibility

4. **Permission Errors**:
   - Ensure write permissions for output directories
   - Check log file access

### Getting Help

- Check the main README.md for general guidance
- Review module-specific documentation in `docs/`
- Examine test files for usage patterns

## Contributing Examples

When adding new examples:

1. **Follow Naming Convention**: Use descriptive, snake_case names
2. **Add Documentation**: Include docstrings and comments
3. **Error Handling**: Implement robust error handling
4. **Configuration**: Support configuration files
5. **Testing**: Include tests in the example or separate test file

Example template:

```python
"""
Example: Brief description

This example demonstrates...
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from module import Class


def main():
    """Main example function."""
    parser = argparse.ArgumentParser(description="Example description")
    parser.add_argument("--config", help="Configuration file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run example
    example = Class(config)
    result = example.run()
    
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

## Further Reading

- [Main README](../README.md) - Project overview
- [Development Guide](../docs/DEVELOPMENT.md) - Architecture details
- [API Documentation](../src/) - Module documentation
- [Performance Guide](../docs/PERFORMANCE_OPTIMIZATION.md) - Optimization tips