# Brain-Inspired Modular AI Framework - Test Suite

Comprehensive testing suite for the Brain-Inspired Modular AI Framework.

## Structure

```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── requirements.txt            # Test dependencies
├── README.md                   # This file
├── run_tests.py               # Main test runner
├── run_comprehensive_tests.py # Comprehensive test suite
├── run_step2_tests.py         # Step 2 integration tests
├── run_step2_integration_test.py # Integration tests
├── unit/                      # Unit tests for individual components
│   ├── test_affect.py        # Affect detection system tests
│   ├── test_data_gatherer.py # Data collection tests
│   ├── test_faiss.py         # FAISS vector database tests
│   ├── test_memory.py        # Memory system tests
│   ├── test_spiking.py       # Spiking neural network tests
│   └── ...
├── integration/               # Integration and system tests
│   ├── test_auto_learning.py # Auto-learning system tests
│   ├── test_multi_expert.py  # Multi-expert system tests
│   ├── test_performance_system.py # Performance tests
│   └── ...
├── performance/              # Performance and stress tests
│   ├── test_optimization_suite.py
│   ├── performance_benchmarks.py
│   └── ...
├── fixtures/                 # Test fixtures and mock data
│   ├── mock_data_generators.py
│   ├── mini_biai_1_test_utils.py
│   └── ...
└── test_*.py                # Root level test files
```

## Overview

This testing suite provides comprehensive coverage of all framework components:

- **FAISS Tests** (`test_faiss.py`) - Long-term memory vector database functionality
- **Spiking Tests** (`test_spiking.py`) - LIF neuron behavior and spiking router
- **Memory Tests** (`test_memory.py`) - Short-term and long-term memory systems
- **Pipeline Tests** (`test_pipeline.py`) - End-to-end inference pipeline integration

## Quick Start

### Install Dependencies

```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Or use the Makefile
make install
```

### Run Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit         # Unit tests only
make test-integration  # Integration tests only
make test-performance  # Performance tests only

# Run component-specific tests
make test-faiss        # FAISS vector database tests
make test-spiking      # Spiking neural network tests
make test-memory       # Memory system tests
make test-pipeline     # Pipeline integration tests

# Generate coverage report
make test-coverage     # Terminal coverage report
make coverage-html     # HTML coverage report
```

## Test Categories

### Unit Tests (`-m "unit"`)
- Individual component functionality
- Function-level testing
- Mock-based testing
- Fast execution (< 1 minute)

### Integration Tests (`-m "integration"`)
- Component interaction testing
- End-to-end workflows
- Real integration with external services
- Moderate execution time (1-5 minutes)

### Performance Tests (`-m "performance"`)
- Latency and throughput benchmarks
- Memory usage profiling
- Scalability testing
- Longer execution time (5+ minutes)

## Test Structure

### FAISS Tests (`test_faiss.py`)

Tests the Long-Term Memory (LTM) vector database functionality:

- **Add/Query Invariants**: Verify correct addition and retrieval of vectors
- **Serialization**: Test index persistence and loading
- **Index Consistency**: Ensure data integrity after operations
- **Performance**: Test with various data sizes
- **Edge Cases**: Handle empty inputs, high dimensionality, etc.

**Key Test Cases**:
- `test_add_invariants()` - Validates vector addition behavior
- `test_query_invariants()` - Tests query result consistency
- `test_serialization_roundtrip()` - Index persistence testing
- `test_performance_boundaries()` - Large-scale data handling

### Spiking Tests (`test_spiking.py`)

Tests the spiking neural network components:

- **LIF Neuron Behavior**: Membrane potential evolution, spike generation
- **SpikingRouter**: Routing decisions, spike rate monitoring
- **Temporal Dynamics**: Time-based neuron behavior
- **Parameter Sensitivity**: Response to different LIF parameters

**Key Test Cases**:
- `test_membrane_potential_evolution()` - Neuron dynamics
- `test_spike_generation()` - Spike threshold behavior
- `test_spike_rate_consistency()` - Stable spike patterns
- `test_routing_weights_generation()` - Router decision logic

### Memory Tests (`test_memory.py`)

Tests both Short-Term Memory (STM) and Long-Term Memory (LTM):

- **Ring Buffer Operations**: Token storage, overflow handling
- **Key-Value Scratchpad**: Temporary storage functionality
- **Memory Consolidation**: STM to LTM transfer
- **Conversation Flow**: Multi-turn memory persistence

**Key Test Cases**:
- `test_ring_buffer_behavior()` - Buffer overflow handling
- `test_kv_scratchpad()` - Temporary storage testing
- `test_memory_consolidation()` - STM→LTM transfer
- `test_conversation_continuity()` - Multi-turn context

### Pipeline Tests (`test_pipeline.py`)

End-to-end integration testing:

- **Input Processing**: Tokenization, embedding generation
- **Memory Retrieval**: Combined STM/LTM operations
- **Spiking Coordination**: Router decision making
- **Response Generation**: Language model output
- **Performance**: Latency and throughput requirements

**Key Test Cases**:
- `test_end_to_end_processing()` - Complete pipeline execution
- `test_pipeline_latency_requirements()` - Performance validation
- `test_conversation_continuity()` - Multi-turn conversations
- `test_scalability()` - Load testing

## Shared Fixtures

The testing suite uses shared fixtures defined in `conftest.py`:

- **`sample_config`**: Default configuration for testing
- **`sample_texts`**: Reproducible test text data
- **`mock_components`**: Mock implementations of mini-biai-1 components
- **`performance_monitor`**: Utilities for measuring test performance
- **`test_data_generator`**: Generates reproducible test data

## Configuration

### Environment Variables

```bash
# Disable GPU for tests (recommended)
export CUDA_VISIBLE_DEVICES=""

# Enable test mode
export TESTING="true"

# Custom coverage threshold
export COVERAGE_MIN=80
```

### Pytest Configuration

Configuration is managed through `pytest.ini`:

- Test discovery paths
- Coverage requirements (80% minimum)
- Timeout settings (5 minutes)
- Warning filters
- Parallel execution options

## Performance Benchmarks

### Latency Requirements

- **End-to-end latency**: < 150ms (target: 50ms)
- **FAISS query latency**: < 20ms for 1M entries
- **Spike rate**: 5-15% (target: 10%)
- **Memory retrieval**: < 10ms

### Coverage Goals

- **Overall coverage**: ≥ 80%
- **Critical paths**: ≥ 95%
- **Performance tests**: 100% coverage

## Test Data Management

### Reproducible Test Data

All tests use deterministic data generation:

```python
# Example: Generate reproducible vectors
np.random.seed(42)
vectors = np.random.normal(0, 1, (100, 512))
```

### Test Data Cleanup

Tests automatically clean up temporary files:

- Temporary directories are created with `temp_dir` fixture
- Files are cleaned up after each test
- Mock data is reset between tests

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Pull requests
- Push to main branch
- Daily scheduled runs

### Test Execution

```bash
# CI-compatible test command
pytest tests/ --cov=src --cov-report=xml --cov-fail-under=80 -m "not slow"
```

### Quality Gates

- **Coverage**: Minimum 80%
- **Linting**: No flake8 violations
- **Type Checking**: No mypy errors
- **Test Pass Rate**: 100%

## Debugging Tests

### Verbose Output

```bash
# Detailed test output
pytest tests/ -v -s

# Show local variables on failure
pytest tests/ --tb=long
```

### Run Specific Test

```bash
# Run single test
pytest tests/test_faiss.py::TestLTMFaiss::test_add_invariants -v

# Run tests matching pattern
pytest tests/ -k "test_memory" -v
```

### Debug Mode

```bash
# Run with debugger
pytest tests/ --pdb

# Drop into debugger on first failure
pytest tests/ --pdbcls=IPython.terminal.debugger:Pdb
```

## Advanced Testing

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4
```

### Memory Profiling

```bash
# Profile memory usage
python -m memory_profiler tests/test_memory.py

# Profile CPU usage
python -m cProfile -o profile.stats tests/test_pipeline.py
```

### Benchmarking

```bash
# Run performance benchmarks
make benchmark

# Compare with baseline
make benchmark-compare
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure mini-biai-1 is installed or PYTHONPATH includes src/
2. **Missing Dependencies**: Run `pip install -r tests/requirements.txt`
3. **CUDA Issues**: Set `CUDA_VISIBLE_DEVICES=""` to disable GPU
4. **Timeout Issues**: Use `-m "not slow"` to skip slow tests

### Performance Issues

1. **Slow Tests**: Use `-n auto` for parallel execution
2. **Memory Leaks**: Run with memory_profiler
3. **Coverage Failures**: Check test coverage with `--cov-report=term-missing`

## Contributing

### Writing New Tests

1. Place test files in `tests/` directory
2. Name files with `test_*.py` pattern
3. Use existing fixtures from `conftest.py`
4. Follow test naming conventions
5. Include both positive and negative test cases

### Test Documentation

- Document complex test logic
- Use descriptive test names
- Include docstrings for test classes
- Explain edge case handling

### Quality Standards

- All new code must have tests
- Maintain 80% coverage minimum
- Tests should be deterministic
- Include performance tests for critical paths
- Test both success and failure cases