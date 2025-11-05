# Comprehensive Benchmarking Suite - Quick Start Guide

## Overview

This benchmarking suite measures and monitors critical performance metrics:
- **Token throughput**: ≥30 tok/s on RTX 4090
- **Time to First Token (TTFT)**: <5ms
- **Spike rate**: 5-15%
- **Retrieval performance**: <20ms on 1M entries
- **Memory usage, CPU/GPU utilization, energy consumption**

## Quick Start

### 1. Installation

```bash
# Install core dependencies
pip install -r benchmarks/requirements.txt

# Optional: Install system monitoring tools
# Ubuntu/Debian:
sudo apt-get install linux-tools-generic  # For perf
sudo apt-get install powermetrics         # macOS only, for energy monitoring

# macOS:
sudo powermetrics                        # Requires sudo for detailed metrics
```

### 2. Quick Performance Check (5 minutes)

```bash
# Run quick benchmark targeting RTX 4090 performance
python benchmarks/run_benchmarks.py --type rtx_4090

# Or run a quick test
python benchmarks/run_benchmarks.py --type quick
```

### 3. Full Benchmark Suite (15-30 minutes)

```bash
# Run comprehensive benchmark with all metrics
python benchmarks/run_benchmarks.py --type full

# Run with custom configuration
python benchmarks/run_benchmarks.py --type full --iterations 2000
```

### 4. Stress Testing

```bash
# Basic stress test
python benchmarks/stress_test_runner.py --duration 600 --max-concurrency 50

# Breaking point test
python benchmarks/stress_test_runner.py --breaking-point --max-concurrency 100

# Memory stress test only
python benchmarks/stress_test_runner.py --memory-stress --duration 300
```

### 5. Real-time Monitoring Dashboard

```bash
# Start monitoring dashboard (web interface)
python benchmarks/performance_dashboard.py

# Start with regression detection
python benchmarks/performance_dashboard.py --baseline results/baseline.json

# Monitoring-only mode (no web interface)
python benchmarks/performance_dashboard.py --monitor-only
```

## Target Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Token Throughput** | ≥30 tok/s on RTX 4090 | Model inference benchmark |
| **TTFT** | <5ms | First token latency |
| **Spike Rate** | 5-15% | Outlier percentage (3σ threshold) |
| **Retrieval (1M entries)** | <20ms p95 | Vector similarity search |
| **Memory Usage** | <90% system | Peak memory under load |
| **CPU Usage** | <90% sustained | Average during operation |

## Benchmark Types

### 1. Token Throughput Benchmark

Measures inference performance and latency characteristics.

```bash
# Focused tok/s measurement
python benchmarks/run_benchmarks.py --type tok_throughput

# With energy monitoring
python benchmarks/run_benchmarks.py --type tok_throughput --iterations 2000
```

**Output includes:**
- Tokens per second (tok/s)
- Time to First Token (TTFT)
- Latency percentiles (p50, p95, p99)
- Spike rate analysis
- Memory and energy consumption

### 2. Retrieval Performance Benchmark

Tests search performance on large vector databases.

```bash
# Full-scale retrieval test (1M entries)
python benchmarks/run_benchmarks.py --type retrieval

# Faster test with smaller index
python benchmarks/stress_test_runner.py --duration 300
```

**Measures:**
- Query latency on 1M entry index
- Throughput (queries/second)
- Index building time
- Memory usage

### 3. Stress Testing

System stability and breaking point identification.

```bash
# Progressive stress test
python benchmarks/stress_test_runner.py --duration 600 --max-concurrency 100

# Steady load test
python benchmarks/stress_test_runner.py --duration 300 --load-pattern steady

# Spike load test
python benchmarks/stress_test_runner.py --duration 600 --load-pattern spike
```

**Stress patterns:**
- **Steady**: Constant load
- **Ramp**: Linear increase to max load
- **Spike**: Regular high-load bursts
- **Resource pressure**: Memory and CPU stress

### 4. Regression Detection

Compare current performance against baseline.

```bash
# Create baseline
python benchmarks/run_benchmarks.py --type rtx_4090 --output-dir baseline_results

# Compare against baseline
python benchmarks/run_benchmarks.py --type regression --baseline-file baseline_results/rtx_4090_summary.json
```

## Configuration

### Custom Configuration File

Create `benchmark_config.yaml`:

```yaml
target_metrics:
  tok_s: 35.0  # Higher target
  ttft_ms: 4.0  # Tighter TTFT
  spike_rate_range: [3.0, 12.0]  # Tighter spike range
  retrieval_ms: 15.0  # Faster retrieval

test_params:
  iterations: 2000
  warmup_iterations: 200
  batch_sizes: [1, 4, 8, 16, 32, 64]

model:
  name: "microsoft/DialoGPT-large"  # Larger model

retrieval:
  index_size: 2000000  # 2M entries
  queries: 2000
```

### Command Line Options

```bash
# Common options
python benchmarks/run_benchmarks.py --help

python benchmarks/run_benchmarks.py \
    --type full \
    --iterations 1500 \
    --duration 900 \
    --output-dir custom_results \
    --baseline-file baseline.json
```

## Understanding Results

### Benchmark Output Structure

```json
{
  "timestamp": 1704067200.0,
  "benchmarks": {
    "token_throughput": {
      "tokens_per_second": 32.5,
      "p50_ms": 4.2,
      "p95_ms": 8.7,
      "spike_rate_percent": 8.3,
      "memory_peak_mb": 2048.0,
      "energy_consumption_joules": 1250.0,
      "target_scores": {
        "tok_s": true,
        "ttft": true,
        "spike_rate": true
      },
      "meets_targets": true
    }
  },
  "overall_assessment": {
    "performance_grade": "EXCELLENT",
    "targets_met": 4,
    "total_targets": 4
  }
}
```

### Performance Grades

- **🌟 Excellent**: ≥90% targets met, system performing optimally
- **👍 Good**: 80-89% targets met, system performing well
- **⚠️ Acceptable**: 60-79% targets met, system functional with issues
- **❌ Needs Improvement**: <60% targets met, significant issues

### HTML Reports

Generated automatically for full benchmarks:
- Interactive charts and graphs
- Target status indicators
- Performance trends
- System resource usage
- Alert summaries

## Monitoring Dashboard

The web dashboard provides real-time monitoring:

### Access
- URL: http://localhost:8080
- Auto-refreshes every 1 second
- Mobile-responsive design

### Features
- Real-time metrics display
- Status color coding (green/yellow/red)
- Alert notifications
- Performance summaries
- Data export capabilities

### Metrics Displayed
- Token throughput with target status
- TTFT measurements
- Spike rate monitoring
- System resource usage
- Recent alerts

## Common Use Cases

### 1. Development Testing

```bash
# Quick validation during development
python benchmarks/run_benchmarks.py --type quick

# Focus on specific performance area
python benchmarks/run_benchmarks.py --type tok_throughput --iterations 500
```

### 2. CI/CD Integration

```bash
# Fast validation for CI/CD
python benchmarks/run_benchmarks.py --type production

# Check for regressions
python benchmarks/run_benchmarks.py --type regression --baseline-file baseline.json
```

### 3. Production Monitoring

```bash
# Continuous monitoring
python benchmarks/performance_dashboard.py --monitor-only

# Alert setup (modify thresholds in config)
python benchmarks/performance_dashboard.py --baseline-file production_baseline.json
```

### 4. Performance Optimization

```bash
# Identify breaking points
python benchmarks/stress_test_runner.py --breaking-point --max-concurrency 200

# Analyze under sustained load
python benchmarks/stress_test_runner.py --duration 1800 --max-concurrency 100
```

## Troubleshooting

### Common Issues

1. **High TTFT (>5ms)**
   - Check model loading time
   - Verify GPU memory availability
   - Consider model quantization

2. **Low token throughput (<30 tok/s)**
   - Check GPU utilization
   - Verify batch size optimization
   - Monitor system resource usage

3. **High spike rate (>15%)**
   - Check for memory pressure
   - Monitor CPU/GPU thermal throttling
   - Identify resource bottlenecks

4. **Retrieval latency issues**
   - Verify index size vs. available memory
   - Check vector dimension optimization
   - Monitor I/O performance

### Performance Tips

1. **For RTX 4090 optimization:**
   - Use larger batch sizes (32-64)
   - Enable GPU memory optimization
   - Monitor temperature and power limits

2. **For memory efficiency:**
   - Use memory monitoring tools
   - Implement garbage collection optimization
   - Consider model parallelism

3. **For energy efficiency:**
   - Monitor power consumption
   - Optimize for performance/watt
   - Use power management features

### Log Files

Check log files for detailed information:
- `benchmark_suite.log`: Main benchmark execution log
- `monitoring.log`: Performance monitoring log
- `stress_test_results/`: Stress test output files

## Extending the Suite

### Adding Custom Benchmarks

```python
# Create custom benchmark function
async def my_custom_benchmark():
    # Your benchmark logic here
    return metrics

# Add to benchmark suite
results = await benchmark.run_custom_benchmark(my_custom_benchmark, "my_benchmark")
```

### Custom Metrics

Add new metrics by extending the `PerformanceMetrics` class:

```python
@dataclass
class CustomPerformanceMetrics(PerformanceMetrics):
    my_custom_metric: float = 0.0
```

### Integration with Other Tools

The benchmark suite can be integrated with:
- Prometheus/Grafana for monitoring
- Jenkins/GitHub Actions for CI/CD
- Custom alerting systems
- Performance regression pipelines

## Best Practices

1. **Always warm up** the system before measurements
2. **Run multiple iterations** for statistical significance
3. **Monitor system resources** during testing
4. **Use baselines** for regression detection
5. **Test at realistic load** levels
6. **Document configuration** changes
7. **Set up automated** monitoring for production

## Support and Contributing

For issues, feature requests, or contributions:
- Check existing logs and outputs
- Provide configuration details
- Include system specifications
- Report performance regression scenarios

This benchmarking suite provides comprehensive performance measurement and monitoring capabilities for AI systems, ensuring optimal performance and reliability in production environments.
