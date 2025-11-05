# Comprehensive Benchmarking Suite

This directory contains a comprehensive benchmarking suite for measuring tok/s, TTFT, memory usage, energy consumption, spike rates, and retrieval performance with stress testing and regression detection.

## Overview

**Target Metrics:**
- **≥30 tok/s on RTX 4090**
- **<5ms TTFT**
- **5-15% spike rate**
- **<20ms retrieval on 1M entries**

The benchmarking suite provides standardized benchmarks for measuring system performance, scalability, and reliability under various conditions, with comprehensive monitoring and alerting capabilities.

## Quick Start

```bash
# Quick performance check (5 minutes)
python run_benchmarks.py --type quick

# RTX 4090 optimized benchmark
python run_benchmarks.py --type rtx_4090

# Full comprehensive benchmark
python run_benchmarks.py --type full

# Stress testing
python stress_test_runner.py --duration 600 --max-concurrency 100

# Real-time monitoring dashboard
python performance_dashboard.py
```

## Directory Structure

```
benchmarks/
├── README.md                               # This file
├── QUICK_START.md                          # Detailed quick start guide
├── requirements.txt                        # Python dependencies
├── benchmark_config.yaml                   # Default configuration
├── comprehensive_benchmark_suite.py        # Main benchmarking framework
├── run_benchmarks.py                       # CLI runner with presets
├── stress_test_runner.py                   # Stress testing suite
├── performance_dashboard.py                # Real-time monitoring dashboard
├── output/                                 # Benchmark results directory
│   ├── reports/                           # HTML and JSON reports
│   ├── monitoring_data/                   # Monitoring data storage
│   └── stress_test_results/               # Stress testing outcomes
└── baseline/                               # Baseline results for regression
    └── baseline_config.json               # Baseline configuration
```

## Benchmark Categories

### 1. Core Performance Benchmarks

**`comprehensive_benchmark_suite.py`**

Main benchmarking framework with support for:
- Token throughput measurement (≥30 tok/s target)
- Time to First Token (TTFT) analysis (<5ms target)
- Spike rate monitoring (5-15% target range)
- Memory usage profiling
- Energy consumption tracking
- Retrieval performance on 1M entries (<20ms target)

**Key Metrics:**
- Tokens per second with batch size analysis
- Latency percentiles (p50, p95, p99, p999)
- Memory consumption (peak and average)
- CPU/GPU utilization
- Energy consumption in joules
- Spike rate percentage (3σ outlier detection)

### 2. Benchmark Runner

**`run_benchmarks.py`**

CLI interface with preset configurations:
- **Quick**: Fast validation (5 minutes)
- **Tok Throughput**: Focused performance testing
- **Retrieval**: Vector search performance
- **Full**: Comprehensive testing (15-30 minutes)
- **RTX 4090**: Hardware-optimized settings
- **Production**: CI/CD friendly configuration
- **Regression**: Performance comparison against baseline

**Features:**
- Preset configurations for different use cases
- Configurable iterations and parameters
- Baseline comparison for regression detection
- HTML report generation
- Automated assessment and grading

### 3. Stress Testing Suite

**`stress_test_runner.py`**

Comprehensive stress testing including:
- Steady-state load testing
- Ramp-up/ramp-down load patterns
- Spike load simulation
- Breaking point identification
- Resource pressure testing (memory/CPU)
- System stability assessment

**Load Patterns:**
- **Steady**: Constant concurrent load
- **Ramp**: Linear increase to target load
- **Spike**: Regular high-intensity bursts
- **Resource pressure**: Memory and CPU stress testing

**Analysis:**
- Maximum sustainable load identification
- Error rate monitoring
- Latency degradation tracking
- System breaking point detection

### 4. Real-time Monitoring Dashboard

**`performance_dashboard.py`**

Web-based monitoring interface featuring:
- Real-time metrics display
- Target status indicators
- Alert notifications
- Performance trend analysis
- Regression detection
- Data export capabilities

**Features:**
- Web dashboard with auto-refresh
- Status color coding (green/yellow/red)
- Alert cooldown management
- Historical data analysis
- Baseline comparison
- Mobile-responsive design

## Running Benchmarks

### Quick Performance Check

Run basic performance tests targeting RTX 4090:

```bash
# Quick validation (5 minutes)
python benchmarks/run_benchmarks.py --type quick

# RTX 4090 optimized test
python benchmarks/run_benchmarks.py --type rtx_4090
```

### Comprehensive Testing

Run full benchmark suite:

```bash
# Complete benchmark suite (15-30 minutes)
python benchmarks/run_benchmarks.py --type full

# With custom configuration
python benchmarks/run_benchmarks.py \
    --type full \
    --iterations 2000 \
    --baseline-file baseline/baseline.json \
    --output-dir custom_results
```

### Focused Benchmarking

Run specific benchmark types:

```bash
# Token throughput focused
python benchmarks/run_benchmarks.py --type tok_throughput

# Retrieval performance only
python benchmarks/run_benchmarks.py --type retrieval

# Regression testing
python benchmarks/run_benchmarks.py --type regression --baseline-file baseline.json
```

### Stress Testing

Run stress tests:

```bash
# Basic stress test (10 minutes)
python benchmarks/stress_test_runner.py --duration 600 --max-concurrency 50

# Breaking point test
python benchmarks/stress_test_runner.py --breaking-point --max-concurrency 200

# Memory stress test
python benchmarks/stress_test_runner.py --memory-stress --duration 300

# Production load simulation (30 minutes)
python benchmarks/stress_test_runner.py --production-load
```

### Real-time Monitoring

Start monitoring dashboard:

```bash
# Web dashboard
python benchmarks/performance_dashboard.py

# With baseline for regression detection
python benchmarks/performance_dashboard.py --baseline-file baseline.json

# Monitoring-only mode
python benchmarks/performance_dashboard.py --monitor-only

# Custom port
python benchmarks/performance_dashboard.py --port 9000
```

## Benchmark Configuration

### Configuration Files

Use the default configuration or create custom configurations:

```bash
# Use default configuration
python benchmarks/run_benchmarks.py --type full

# Create custom configuration file (benchmark_config.yaml):
cp benchmarks/benchmark_config.yaml my_config.yaml
# Edit my_config.yaml with your settings
```

### Target Performance Metrics

| Metric | RTX 4090 Target | Measurement Method |
|--------|----------------|-------------------|
| **Token Throughput** | ≥30 tok/s | Model inference benchmarking |
| **TTFT (Time to First Token)** | <5ms | First token latency |
| **Spike Rate** | 5-15% | 3σ outlier percentage |
| **Retrieval (1M entries)** | <20ms p95 | Vector similarity search |
| **Memory Usage** | <90% system | Peak under load |
| **Energy Consumption** | Monitored | Power usage tracking |

### Configuration Parameters

```yaml
# Core target metrics
target_metrics:
  tok_s: 30.0                  # Token throughput target
  ttft_ms: 5.0                 # TTFT target
  spike_rate_range: [5.0, 15.0] # Spike rate range
  retrieval_ms: 20.0           # Retrieval latency target

# Test parameters
test_params:
  iterations: 1000             # Main benchmark iterations
  warmup_iterations: 100       # Warmup runs
  timeout_seconds: 300         # Test timeout
  batch_sizes: [1, 4, 8, 16, 32] # Batch sizes to test

# Model configuration
model:
  name: "microsoft/DialoGPT-medium" # Model to benchmark
  test_text_lengths: [10, 50, 100, 500, 1000] # Text lengths

# Retrieval test configuration
retrieval:
  index_size: 1000000          # 1M entries
  queries: 1000                # Number of queries
  top_k: 10                    # Top-K results

# Stress test parameters
stress_test:
  concurrency_levels: [1, 5, 10, 20, 50] # Concurrency levels
  duration_seconds: 300        # Stress test duration

# Monitoring configuration
monitoring:
  interval_ms: 100             # Monitoring interval
  energy_sample_interval_ms: 1000 # Energy monitoring

# Output configuration
output:
  directory: "benchmark_results"
  save_detailed_logs: true
  generate_html_report: true

# Hardware-specific presets
hardware_configs:
  rtx_4090:
    target_tok_s: 30.0
    target_ttft_ms: 5.0
    batch_sizes: [1, 4, 8, 16, 32, 64]
    
  rtx_3080:
    target_tok_s: 25.0
    target_ttft_ms: 8.0
    batch_sizes: [1, 4, 8, 16, 32]
    
  cpu_only:
    target_tok_s: 10.0
    target_ttft_ms: 15.0
    batch_sizes: [1, 2, 4, 8]
```

## Interpreting Results

### Performance Metrics

#### Token Throughput
- **Tokens/Second**: Average tokens processed per second
- **Target**: ≥30 tok/s on RTX 4090
- **Batch Analysis**: Performance across different batch sizes
- **Context**: Model inference rate measurement

#### Time to First Token (TTFT)
- **p50 (Median)**: 50% of requests get first token within this time
- **p95**: 95% of requests get first token within this time  
- **Target**: <5ms for optimal user experience
- **Context**: Critical for conversational AI response time

#### Spike Rate
- **Percentage**: Outliers beyond 3σ threshold
- **Target Range**: 5-15% indicates healthy system
- **Too Low**: May indicate insufficient load testing
- **Too High**: Indicates system instability or resource constraints

#### Retrieval Performance
- **Query Latency**: Time to retrieve top-K results
- **p95 Target**: <20ms for 1M entry index
- **Throughput**: Queries per second capacity
- **Index Scaling**: Performance vs. index size

#### Resource Usage
- **Memory**: Peak and average memory consumption (MB/GB)
- **CPU**: CPU utilization percentage with sustained load
- **GPU**: GPU utilization, memory usage, temperature
- **Energy**: Power consumption and energy efficiency

### Performance Assessment Grades

| Grade | Description | Target Achievement |
|-------|-------------|-------------------|
| **🌟 Excellent** | ≥90% targets met, optimal performance | System performing at peak efficiency |
| **👍 Good** | 80-89% targets met, good performance | System performing well with minor issues |
| **⚠️ Acceptable** | 60-79% targets met, functional | System operational but needs optimization |
| **❌ Poor** | <60% targets met, needs improvement | Significant issues requiring attention |

### Alert Categories

#### Performance Alerts
- **Token throughput**: Below threshold
- **TTFT**: Exceeding target latency
- **Spike rate**: Outside acceptable range
- **Retrieval latency**: Query time too high

#### Resource Alerts  
- **Memory usage**: Approaching system limits
- **CPU utilization**: Sustained high usage
- **GPU utilization**: Thermal throttling risk
- **Energy consumption**: Unusual power draw

#### Stability Alerts
- **Error rate**: Increasing failure rate
- **Regression**: Performance degradation vs. baseline
- **System instability**: Crash or timeout events

## Benchmark Results

### Output Files

Results are saved in structured JSON formats with HTML reports:

```json
{
  "timestamp": 1704067200.0,
  "config": {
    "target_tok_s": 30.0,
    "target_ttft_ms": 5.0,
    "iterations": 1000
  },
  "system_info": {
    "cpu_count": 16,
    "memory_total_gb": 32.0,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_memory_gb": 24.0
  },
  "benchmarks": {
    "token_throughput": {
      "tokens_per_second": 32.5,
      "p50_ms": 4.2,
      "p95_ms": 8.7,
      "p99_ms": 15.3,
      "spike_rate_percent": 8.3,
      "memory_peak_mb": 4096.0,
      "energy_consumption_joules": 1250.0,
      "target_scores": {
        "tok_s": true,
        "ttft": true,
        "spike_rate": true
      },
      "meets_targets": true
    },
    "retrieval_performance": {
      "p95_ms": 18.3,
      "p50_ms": 12.1,
      "throughput_rps": 125.4,
      "meets_targets": true
    }
  },
  "stress_test": {
    "max_sustainable_load": 75,
    "stable_concurrency_levels": [1, 5, 10, 20, 50]
  },
  "overall_assessment": {
    "performance_grade": "EXCELLENT",
    "targets_met": 4,
    "total_targets": 4,
    "recommendations": []
  }
}
```

### HTML Reports

Automated HTML report generation includes:
- **Interactive charts**: Performance trends and distributions
- **Target status indicators**: Visual pass/fail status
- **System information**: Hardware and configuration details
- **Alert summaries**: Recent performance issues
- **Responsive design**: Mobile and desktop viewing

### Result Analysis Tools

```bash
# Run full benchmark and view HTML report
python benchmarks/run_benchmarks.py --type full

# Compare against baseline
python benchmarks/run_benchmarks.py --type regression --baseline-file baseline.json

# Export monitoring data
python benchmarks/performance_dashboard.py --export

# Quick comparison
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
    print(f\"Grade: {data['overall_assessment']['performance_grade']}\")
    print(f\"Targets: {data['overall_assessment']['targets_met']}/{data['overall_assessment']['total_targets']}\")
"
```

## Continuous Benchmarking

### Automated Testing

Set up automated benchmarks for:

1. **CI/CD Integration**: Run benchmarks on each commit
2. **Daily Tests**: Automated daily performance validation
3. **Release Validation**: Performance validation before releases
4. **Production Monitoring**: Continuous performance monitoring

### Scheduled Benchmarks

Configure automated testing:

```bash
# Quick CI/CD validation (5 minutes)
0 */6 * * * cd /path/to/project && python benchmarks/run_benchmarks.py --type production --output ci_$(date +\%Y\%m\%d_\%H\%M).json

# Daily comprehensive test
0 2 * * * cd /path/to/project && python benchmarks/run_benchmarks.py --type full --output daily_$(date +\%Y\%m\%d).json

# Weekly stress test
0 3 * * 0 cd /path/to/project && python benchmarks/stress_test_runner.py --duration 900 --max-concurrency 100 --output weekly_$(date +\%Y\%W).json

# Continuous monitoring dashboard
nohup python benchmarks/performance_dashboard.py --monitor-only > monitoring.log 2>&1 &
```

## Optimization Guidelines

### Performance Bottlenecks

Common bottlenecks and solutions:

#### Token Throughput Issues (<30 tok/s)
- **GPU Memory**: Optimize batch sizes, check VRAM usage
- **Model Size**: Consider model quantization or distillation
- **Compute Efficiency**: Profile kernel utilization
- **Memory Bandwidth**: Optimize data loading and transfer

#### TTFT Issues (>5ms)
- **Model Loading**: Pre-load models, optimize initialization
- **Warm-up**: Ensure proper model warm-up sequence
- **Memory Allocation**: Reduce memory fragmentation
- **Pipeline Optimization**: Stream processing setup

#### Spike Rate Issues (>15%)
- **Resource Contention**: Monitor CPU/GPU throttling
- **Memory Pressure**: Check for memory leaks or garbage collection
- **Thermal Issues**: Monitor temperature and power limits
- **I/O Bottlenecks**: Optimize data access patterns

#### Retrieval Performance (>20ms on 1M entries)
- **Index Optimization**: Vector quantization, IVF indexes
- **Memory Usage**: Increase available RAM for caching
- **Compute Optimization**: SIMD optimizations, GPU acceleration
- **Data Layout**: Optimize data structures for cache efficiency

### Optimization Process

1. **Baseline**: Establish performance baseline with comprehensive benchmarks
2. **Profile**: Use monitoring dashboard to identify hotspots
3. **Hypothesize**: Identify root causes of performance issues
4. **Optimize**: Implement targeted improvements
5. **Validate**: Re-run benchmarks to measure impact
6. **Regression Test**: Ensure changes don't introduce regressions

### Hardware-Specific Optimizations

#### RTX 4090 Optimization
- **Batch Sizes**: Use 32-64 for optimal GPU utilization
- **Memory Management**: Enable GPU memory optimization
- **Power Management**: Monitor and optimize power limits
- **Thermal Management**: Ensure adequate cooling

#### Memory Optimization
- **Object Pooling**: Reuse objects to reduce allocation overhead
- **Memory Mapping**: Use memory-mapped files for large datasets
- **Garbage Collection**: Tune GC parameters for low-latency requirements
- **Data Structures**: Use memory-efficient structures (arrays vs. lists)

#### CPU Optimization
- **Parallel Processing**: Utilize multi-core architecture
- **Vectorization**: Use SIMD instructions for compute operations
- **Cache Optimization**: Optimize data access patterns
- **Algorithm Selection**: Choose optimal algorithms for data size

## Best Practices

### Benchmarking Guidelines

1. **Consistent Environment**: Use identical hardware/OS configurations
2. **Warm-up Runs**: Always run warm-up iterations before measurements
3. **Statistical Significance**: Use sufficient iterations for reliable averages
4. **Control Variables**: Keep configurations consistent across tests
5. **Baseline Comparison**: Maintain baseline measurements for regression detection
6. **Documentation**: Track system changes affecting performance

### Test Design Principles

1. **Realistic Workloads**: Use representative data and usage patterns
2. **Edge Cases**: Test extreme conditions and boundary cases
3. **Scalability Testing**: Test with varying loads and data sizes
4. **Stability Testing**: Run long-duration tests for system stability
5. **Resource Monitoring**: Monitor system resources during all tests
6. **Alert Integration**: Set up alerts for performance regressions

### Production Deployment

1. **Continuous Monitoring**: Deploy monitoring dashboard in production
2. **Automated Testing**: Integrate benchmarks into CI/CD pipeline
3. **Alert Thresholds**: Configure appropriate alert thresholds
4. **Baseline Updates**: Regularly update performance baselines
5. **Performance Budgets**: Define and enforce performance budgets
6. **Rollback Procedures**: Have rollback plans for performance regressions

### Performance Monitoring

1. **Key Metrics Focus**: Monitor tok/s, TTFT, spike rate, and retrieval latency
2. **Real-time Dashboards**: Use web dashboard for continuous monitoring
3. **Historical Analysis**: Track performance trends over time
4. **Alert Management**: Implement intelligent alert cooldown and filtering
5. **Data Export**: Regular export and backup of monitoring data
6. **Capacity Planning**: Use stress testing for capacity planning

## Contributing Benchmarks

When adding new benchmarks:

1. **Follow Naming**: Use descriptive, snake_case names
2. **Include Metrics**: Measure relevant performance metrics
3. **Document Purpose**: Explain what the benchmark tests
4. **Add Configurations**: Provide example configurations
5. **Update Documentation**: Add to this README

Example benchmark template:

```python
"""
Benchmark: Module Performance Test

Measures performance of specific module under various conditions.
"""

import time
import psutil
from typing import Dict, Any


def benchmark_function():
    """Core benchmarking function."""
    # Setup
    start_time = time.time()
    process = psutil.Process()
    
    # Warm-up runs
    for _ in range(100):
        function_under_test()
    
    # Benchmark runs
    iterations = 1000
    timings = []
    
    for _ in range(iterations):
        iteration_start = time.time()
        function_under_test()
        timings.append(time.time() - iteration_start)
    
    # Calculate metrics
    results = {
        "latency": {
            "mean": sum(timings) / len(timings),
            "p50": sorted(timings)[len(timings) // 2],
            "p95": sorted(timings)[int(len(timings) * 0.95)]
        },
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "iterations": iterations
    }
    
    return results
```

## Further Reading

- [Performance Optimization Guide](../docs/PERFORMANCE_OPTIMIZATION.md) - Detailed optimization strategies
- [Memory Systems Documentation](../docs/MEMORY_SYSTEMS_README.md) - Memory architecture details
- [Development Guide](../docs/DEVELOPMENT.md) - System architecture