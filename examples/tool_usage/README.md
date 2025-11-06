# Tool Usage Examples

This directory contains comprehensive demonstration scripts for the mini-biai-1 tool_usage system.

## Overview

The tool_usage system provides advanced command execution, cross-platform compatibility, security features, and intelligent optimization for various platforms and environments.

## Demo Files

### 1. basic_usage_demo.py
Basic tool usage examples including:
- Simple command execution
- Platform detection
- Basic error handling
- Output formatting

**Usage:**
```bash
python examples/tool_usage/basic_usage_demo.py
python examples/tool_usage/basic_usage_demo.py --command "ls -la"
python examples/tool_usage/basic_usage_demo.py --interactive
```

### 2. advanced_optimization_demo.py
Advanced optimization features including:
- Usage pattern analysis
- Performance optimization
- Caching strategies
- Batch operations

**Usage:**
```bash
python examples/tool_usage/advanced_optimization_demo.py
python examples/tool_usage/advanced_optimization_demo.py --benchmark
python examples/tool_usage/advanced_optimization_demo.py --cache-size 1000
```

### 3. security_features_demo.py
Security and safety demonstrations including:
- Command validation
- Permission checking
- Sandbox execution
- Audit logging

**Usage:**
```bash
python examples/tool_usage/security_features_demo.py
python examples/tool_usage/security_features_demo.py --audit
python examples/tool_usage/security_features_demo.py --sandbox
```

### 4. cross_platform_demo.py
Cross-platform compatibility examples including:
- Platform-specific handling
- Windows/Linux/Mac compatibility
- Shell detection and adaptation
- Path handling

**Usage:**
```bash
python examples/tool_usage/cross_platform_demo.py
python examples/tool_usage/cross_platform_demo.py --platform windows
python examples/tool_usage/cross_platform_demo.py --test-all
```

### 5. integration_demo.py
Integration with mini-biai-1 modules including:
- Affect system integration
- Memory system integration
- Learning system integration
- Coordinator integration

**Usage:**
```bash
python examples/tool_usage/integration_demo.py
python examples/tool_usage/integration_demo.py --module affect
python examples/tool_usage/integration_demo.py --full-integration
```

### 6. learning_demo.py
Machine learning and intelligence features including:
- Usage pattern learning
- Intelligent optimization suggestions
- Performance prediction
- Adaptive learning

**Usage:**
```bash
python examples/tool_usage/learning_demo.py
python examples/tool_usage/learning_demo.py --train
python examples/tool_usage/learning_demo.py --analyze
```

## Prerequisites

Ensure you have:
- Python 3.7+
- All dependencies from requirements.txt
- Access to the src/ directory
- Appropriate permissions for command execution

## Quick Start

Run all demos sequentially:
```bash
for demo in basic_usage advanced_optimization security_features cross_platform integration learning; do
    python examples/tool_usage/${demo}_demo.py
done
```

## Configuration

Each demo accepts command-line arguments. Use `--help` for detailed options:
```bash
python examples/tool_usage/basic_usage_demo.py --help
```

## Output

Demos generate outputs in:
- Console (real-time feedback)
- `demo_results/tool_usage/` directory (detailed logs)
- Temporary files for analysis results

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src/ is in Python path
2. **Permission Errors**: Check execution permissions
3. **Platform Issues**: Verify platform-specific requirements
4. **Missing Dependencies**: Install required packages

### Getting Help

- Check individual demo docstrings
- Review src/tool_usage/ module documentation
- Examine integration examples in src/tool_usage/integration/

## Architecture

The tool_usage system consists of:

- **Command Execution**: Platform-adaptive command execution
- **Security Framework**: Safe command validation and execution
- **Optimization Engine**: Performance optimization and caching
- **Intelligence Layer**: Learning and adaptive optimization
- **Integration Hub**: Connection with mini-biai-1 modules
- **Cross-Platform Support**: Windows/Linux/Mac compatibility