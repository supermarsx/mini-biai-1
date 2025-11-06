# Tool Usage Testing Suite

Comprehensive testing suite for the mini-biai-1 tool_usage module, designed to achieve >90% code coverage and ensure robust functionality across different platforms and use cases.

## Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Coverage Analysis](#coverage-analysis)
- [Test Categories](#test-categories)
- [Manual Testing](#manual-testing)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

The tool_usage testing suite provides comprehensive validation of the tool_usage module components:

- **ShellDetector**: Cross-platform shell identification and capabilities detection
- **CommandExecutor**: Secure command execution with validation and error handling
- **ToolRegistry**: Tool discovery, registration, and metadata management
- **UsageOptimizer**: Command pattern analysis and optimization recommendations
- **PlatformAdapter**: OS-specific adaptations for POSIX and Windows systems

## Test Structure

```
tests/tool_usage/
├── conftest.py                     # Shared fixtures and configuration
├── unit/                           # Unit tests for individual components
│   ├── test_shell_detector.py
│   ├── test_command_executor.py
│   ├── test_tool_registry.py
│   ├── test_usage_optimizer.py
│   └── test_platform_adapter.py
├── integration/                    # Integration tests
│   └── test_tool_usage_integration.py
├── security/                       # Security and safety tests
│   └── test_security_safety.py
├── compatibility/                  # Cross-platform compatibility tests
│   └── test_cross_platform.py
├── performance/                    # Performance and load tests
│   └── test_performance_load.py
├── manual_tests/                   # Manual testing scripts
│   └── run_manual_tests.py
└── README.md                       # This file
```

## Running Tests

### Quick Test Run

Run all tests with a single command:

```bash
# From project root
python -m pytest tests/tool_usage/ -v --cov=src.tool_usage --cov-report=html --cov-report=term
```

### Run Specific Test Categories

```bash
# Unit tests only
python -m pytest tests/tool_usage/unit/ -v

# Integration tests
python -m pytest tests/tool_usage/integration/ -v

# Security tests
python -m pytest tests/tool_usage/security/ -v

# Compatibility tests
python -m pytest tests/tool_usage/compatibility/ -v

# Performance tests
python -m pytest tests/tool_usage/performance/ -v -s
```

### Run Specific Test Files

```bash
# Run ShellDetector tests
python -m pytest tests/tool_usage/unit/test_shell_detector.py -v

# Run CommandExecutor tests
python -m pytest tests/tool_usage/unit/test_command_executor.py -v

# Run all tests for a component
python -m pytest tests/tool_usage/ -k "shell_detector" -v
```

### Run Tests with Specific Markers

```bash
# Run only fast tests
python -m pytest -m "not slow" -v

# Run only cross-platform tests
python -m pytest -m "cross_platform" -v

# Run only security tests
python -m pytest -m "security" -v

# Run only performance tests
python -m pytest -m "performance" -v
```

### Test Output Options

```bash
# Run tests with detailed output
python -m pytest tests/tool_usage/ -v --tb=long

# Run tests and show local variables on failure
python -m pytest tests/tool_usage/ -v --tb=long --showlocals

# Run tests and stop on first failure
python -m pytest tests/tool_usage/ -v -x

# Run tests and continue after first failure
python -m pytest tests/tool_usage/ -v --maxfail=3

# Run tests with timing information
python -m pytest tests/tool_usage/ --durations=10
```

## Coverage Analysis

### Generating Coverage Reports

```bash
# Generate coverage report in terminal
python -m pytest tests/tool_usage/ --cov=src.tool_usage --cov-report=term

# Generate HTML coverage report
python -m pytest tests/tool_usage/ --cov=src.tool_usage --cov-report=html

# Generate XML coverage report (for CI/CD)
python -m pytest tests/tool_usage/ --cov=src.tool_usage --cov-report=xml
```

### Coverage Goals

- **Overall Coverage**: >90%
- **Critical Paths**: 100%
- **Error Handling**: >95%
- **Security Functions**: 100%

### Coverage Breakdown by Component

```bash
# Check coverage for specific component
python -m pytest tests/tool_usage/unit/test_shell_detector.py --cov=src.tool_usage.shell_detector --cov-report=term
python -m pytest tests/tool_usage/unit/test_command_executor.py --cov=src.tool_usage.command_executor --cov-report=term
python -m pytest tests/tool_usage/unit/test_tool_registry.py --cov=src.tool_usage.tool_registry --cov-report=term
python -m pytest tests/tool_usage/unit/test_usage_optimizer.py --cov=src.tool_usage.usage_optimizer --cov-report=term
python -m pytest tests/tool_usage/unit/test_platform_adapter.py --cov=src.tool_usage.platform_adapter --cov-report=term
```

### Coverage Report Interpretation

The HTML coverage report (`htmlcov/index.html`) provides:
- **Line Coverage**: Percentage of lines executed
- **Branch Coverage**: Percentage of branches taken
- **Function Coverage**: Percentage of functions called
- **Missing Lines**: Specific lines not covered

## Test Categories

### Unit Tests (`tests/tool_usage/unit/`)

Tests individual components in isolation with comprehensive mocking:

- **ShellDetector**: Shell detection, capabilities, version parsing
- **CommandExecutor**: Command execution, error handling, timeout management
- **ToolRegistry**: Tool discovery, registration, metadata management
- **UsageOptimizer**: Pattern analysis, optimization suggestions, learning
- **PlatformAdapter**: OS detection, path normalization, command translation

### Integration Tests (`tests/tool_usage/integration/`)

Tests component interactions and end-to-end workflows:

- Shell detection + command execution workflows
- Tool registry + usage optimization integration
- Platform adaptation + command execution
- Multi-component error handling scenarios

### Security Tests (`tests/tool_usage/security/`)

Tests security and safety features:

- Command injection prevention
- Input validation and sanitization
- Sandbox execution verification
- Permission and access control
- Malicious command detection

### Compatibility Tests (`tests/tool_usage/compatibility/`)

Tests cross-platform behavior:

- Windows/Linux/macOS shell detection
- Path handling differences
- Command translation accuracy
- Platform-specific error handling
- Shell capabilities validation

### Performance Tests (`tests/tool_usage/performance/`)

Tests performance characteristics:

- Command execution benchmarks
- Memory usage profiling
- Concurrent execution load testing
- Large-scale tool discovery performance
- Optimization algorithm efficiency

## Manual Testing

### Interactive Manual Testing

The manual testing suite provides interactive scenarios for real-world validation:

```bash
# Run all manual tests interactively
python tests/tool_usage/manual_tests/run_manual_tests.py

# Run specific component manual tests
python tests/tool_usage/manual_tests/run_manual_tests.py --component shell-detector
python tests/tool_usage/manual_tests/run_manual_tests.py --component command-executor
python tests/tool_usage/manual_tests/run_manual_tests.py --component tool-registry
python tests/tool_usage/manual_tests/run_manual_tests.py --component usage-optimizer
python tests/tool_usage/manual_tests/run_manual_tests.py --component platform-adapter
python tests/tool_usage/manual_tests/run_manual_tests.py --component workflow
```

### Manual Test Scenarios

The interactive runner provides:

1. **Real-world shell detection scenarios**
2. **Actual command execution with timeout testing**
3. **Tool discovery on your system**
4. **Command optimization recommendations**
5. **Platform-specific behavior validation**
6. **End-to-end workflow demonstration**

### Non-Interactive Testing

```bash
# Run in non-interactive mode
python tests/tool_usage/manual_tests/run_manual_tests.py --non-interactive --component workflow

# Verbose output
python tests/tool_usage/manual_tests/run_manual_tests.py --verbose
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tool Usage Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        python -m pytest tests/tool_usage/ \
          --cov=src.tool_usage \
          --cov-report=xml \
          --cov-report=term \
          -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: tool_usage
```

### Local CI Simulation

```bash
# Run tests as they would run in CI
python -m pytest tests/tool_usage/ \
  --cov=src.tool_usage \
  --cov-report=xml \
  --cov-report=html \
  --cov-fail-under=90 \
  -v \
  --tb=short
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Error: No module named 'src.tool_usage'
# Solution: Run from project root directory
cd /path/to/project/root
python -m pytest tests/tool_usage/
```

#### Coverage Not Generated
```bash
# Error: Coverage file not found
# Solution: Ensure pytest-cov is installed
pip install pytest-cov
```

#### Slow Tests
```bash
# Tests taking too long
# Solution: Run with -x flag to stop on first failure
python -m pytest tests/tool_usage/ -x
```

#### Platform-Specific Failures
```bash
# Tests failing on specific OS
# Solution: Run compatibility tests separately
python -m pytest tests/tool_usage/compatibility/ -v
```

### Debug Mode

```bash
# Run tests with debugger
python -m pytest tests/tool_usage/unit/test_shell_detector.py --pdb

# Run tests with verbose logging
python -m pytest tests/tool_usage/ -v --log-level=DEBUG
```

### Test Data Issues

If tests fail due to missing test data:

```bash
# Check if test fixtures are properly set up
python -c "from tests.tool_usage.conftest import *; print('Fixtures loaded successfully')"
```

## Best Practices

### Writing New Tests

1. **Follow naming conventions**: `test_component_functionality`
2. **Use descriptive test names**: `test_detects_bash_shell_on_linux`
3. **Include docstrings**: Explain what the test validates
4. **Mock external dependencies**: Use `@patch` and `Mock`
5. **Test edge cases**: Invalid inputs, missing files, network errors
6. **Use fixtures**: Leverage `conftest.py` shared fixtures

### Test Organization

```python
import pytest
from unittest.mock import Mock, patch

class TestComponent:
    """Test cases for Component class."""
    
    def test_basic_functionality(self):
        """Test basic component functionality."""
        # Arrange
        component = Component()
        
        # Act
        result = component.method()
        
        # Assert
        assert result == expected_value
    
    @pytest.mark.parametrize("input_value,expected", [
        (value1, expected1),
        (value2, expected2),
    ])
    def test_parametrized_cases(self, input_value, expected):
        """Test multiple input cases."""
        component = Component()
        result = component.method(input_value)
        assert result == expected
    
    @patch('module.external_dependency')
    def test_with_mocks(self, mock_external):
        """Test with mocked dependencies."""
        mock_external.return_value = "mocked"
        component = Component()
        result = component.method()
        assert result == "expected"
```

### Coverage Improvement

1. **Identify missing coverage**: Use HTML report to find untested lines
2. **Add edge case tests**: Test error conditions and boundary values
3. **Test exception paths**: Ensure try/except blocks are covered
4. **Mock systematically**: Isolate units under test
5. **Test integration points**: Verify component interactions

### Performance Testing

1. **Benchmark critical functions**: Use `@pytest.mark.performance`
2. **Profile memory usage**: Check for memory leaks
3. **Test concurrent scenarios**: Multi-threading and multiprocessing
4. **Monitor resource usage**: CPU and I/O utilization

### Security Testing

1. **Test input validation**: Ensure all inputs are sanitized
2. **Verify sandboxing**: Commands run in isolation
3. **Check permission handling**: Proper access control
4. **Test injection prevention**: SQL, command, and code injection
5. **Validate error messages**: No sensitive information leaked

## Maintenance

### Regular Tasks

- **Update test dependencies**: Keep pytest and related packages current
- **Review test coverage**: Aim for >90% coverage on all components
- **Clean up obsolete tests**: Remove tests for deprecated features
- **Performance monitoring**: Track test execution times
- **Security audit**: Review security test effectiveness

### Test Documentation

- **Keep README updated**: Document new test categories
- **Update comments**: Explain complex test logic
- **Maintain examples**: Provide working test examples
- **Document fixes**: Record troubleshooting solutions

---

## Summary

This testing suite provides comprehensive validation of the tool_usage module with:

- ✅ **650+ lines** of unit tests covering all components
- ✅ **800+ lines** of integration tests for workflows
- ✅ **950+ lines** of security and safety tests
- ✅ **780+ lines** of cross-platform compatibility tests
- ✅ **1000+ lines** of performance and load tests
- ✅ **470 lines** of interactive manual testing scripts
- ✅ **Shared fixtures** and configuration in conftest.py
- ✅ **>90% coverage** target across all components

Run `python tests/tool_usage/manual_tests/run_manual_tests.py --help` for manual testing options.