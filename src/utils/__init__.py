"""
Performance Optimization Utilities for mini-biai-1

This comprehensive utilities package provides performance optimization, monitoring,
and benchmarking tools specifically designed for brain-inspired AI systems. The
module combines low-level profiling capabilities with high-level performance
analysis to enable optimal system operation across diverse hardware configurations.

Performance Optimization Suite:
    The utilities module provides a complete performance engineering toolkit:

    ┌─────────────────────────────────────────────────────────────┐
    │                    Performance Utilities                     │
    └─────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Profiling  │ │Benchmarking │ │Optimization │
    │             │ │             │ │             │
    │ • Memory    │ │ • Hardware  │ │ • Memory    │
    │ • Latency   │ │   Assessment│ │ • Latency   │
    │ • Throughput│ │ • Benchmarks│ │ • Adaptive  │
    └─────────────┘ └─────────────┘ └─────────────┘

Core Components:

Performance Profiling:
    - PerformanceProfiler: Comprehensive performance monitoring and analysis
    - LatencyTracker: Real-time latency measurement and spike detection
    - MemoryOptimizer: Dynamic memory management and optimization
    - OptimizationConfig: Flexible configuration for different scenarios
    - MemoryContext: Context manager for memory-efficient operations

Benchmarking Suite:
    - BenchmarkSuite: Comprehensive benchmarking framework
    - HardwareAssessment: Automatic hardware capability detection
    - Performance validation across different configurations
    - Memory allocation and deallocation pattern analysis
    - Latency distribution measurement and spike detection

Advanced Features:
    - Graceful degradation for resource-constrained environments
    - Adaptive optimization based on runtime performance
    - Real-time performance monitoring with alerting
    - Hardware-aware optimization (CPU/CUDA/MPS)
    - Memory leak detection and prevention
    - Throughput optimization for high-load scenarios

Key Features:
    - Sub-millisecond profiling overhead for real-time monitoring
    - Memory usage tracking with leak detection
    - Automatic hardware detection and optimization
    - Comprehensive latency distribution analysis
    - Batch processing optimization for throughput
    - Thread-safe operations for concurrent workloads
    - Configurable monitoring granularity and sampling rates

Usage Examples:

Performance Profiling:
    >>> from src.utils import PerformanceProfiler, OptimizationConfig
    >>> import time
    >>> 
    >>> # Configure profiler for real-time monitoring
    >>> config = OptimizationConfig(
    ...     sample_rate_hz=1000,  # 1kHz sampling
    ...     memory_monitoring=True,
    ...     latency_tracking=True
    ... )
    >>> 
    >>> profiler = PerformanceProfiler(config)
    >>> profiler.start_monitoring()
    >>> 
    >>> # Monitor operations
    >>> for i in range(1000):
    ...     start_time = time.time()
    ...     result = some_ai_operation()
    ...     profiler.record_operation("ai_operation", time.time() - start_time)
    >>> 
    >>> stats = profiler.get_performance_summary()
    >>> print(f"Average latency: {stats['latency_ms']:.2f}ms")

Memory Optimization:
    >>> from src.utils import MemoryContext, optimize_memory
    >>> 
    >>> # Memory-efficient context manager
    >>> with MemoryContext(max_memory_mb=512) as ctx:
    ...     # Process large dataset in memory-constrained environment
    ...     for batch in large_dataset:
    ...         process_batch(batch, memory_limit=ctx.remaining_memory)

Benchmarking:
    >>> from src.utils import BenchmarkSuite
    >>> 
    >>> # Create benchmark suite
    >>> suite = BenchmarkSuite()
    >>> 
    >>> # Run comprehensive benchmarks
    >>> memory_results = suite.run_memory_benchmark(iterations=1000)
    >>> latency_results = suite.run_latency_benchmark([operation1, operation2])
    >>> 
    >>> # Get hardware assessment
    >>> hardware = suite.get_hardware_assessment()
    >>> print(f"Optimal batch size: {hardware.recommended_batch_size}")

Hardware-Aware Optimization:
    >>> from src.utils import get_profiler
    >>> 
    >>> # Get hardware-optimized profiler
    >>> profiler = get_profiler(
    ...     hardware_type="auto",  # Auto-detect optimal settings
    ...     optimization_level="aggressive"
    ... )
    >>> 
    >>> # Profile with hardware optimization
    >>> with profiler.profile_context("ai_inference"):
    ...     result = ai_model.inference(input_data)

Performance Monitoring:
    >>> from src.utils import GracefulDegradation
    >>> 
    >>> # Monitor performance and trigger degradation if needed
    >>> degrader = GracefulDegradation(
    ...     latency_threshold_ms=150,
    ...     memory_threshold_mb=1024
    ... )
    >>> 
    >>> # Process with degradation protection
    >>> result = degrader.process_with_monitoring(
    ...     operation=ai_inference,
    ...     input_data=data,
    ...     fallback_strategy="reduce_precision"
    ... )

Architecture Benefits:
    - Comprehensive performance engineering toolkit
    - Real-time monitoring with minimal overhead
    - Automatic hardware optimization and fallback
    - Memory leak detection and prevention
    - Latency spike detection and analysis
    - Graceful degradation for resource constraints
    - Benchmarking and validation framework
    - Production-ready monitoring and alerting

Performance Characteristics:
    - Profiling overhead: < 1% for typical operations
    - Memory monitoring: Sub-millisecond sampling
    - Latency tracking: Real-time spike detection
    - Hardware detection: < 100ms initialization time
    - Benchmark execution: Configurable duration (default: 60s)
    - Memory optimization: Automatic garbage collection hints

Hardware Support:
    - CPU: Universal compatibility, full feature support
    - CUDA: GPU-specific optimization and memory tracking
    - MPS: Apple Silicon optimization and performance hints
    - Cross-platform: Windows, Linux, macOS compatibility

Dependencies:
    - psutil: System and process monitoring
    - torch: Neural network performance profiling
    - numpy: Numerical performance optimization
    - time: High-resolution timing measurements
    - threading: Concurrent performance monitoring
    - statistics: Statistical performance analysis
    - json: Performance report serialization

Error Handling:
    The utilities implement comprehensive error handling:
    - Graceful fallback on monitoring system failures
    - Automatic cleanup on profiler destruction
    - Memory leak detection and automatic remediation
    - Performance degradation warnings and alerts
    - Hardware compatibility checks and fallbacks
    - Thread-safe operations with proper locking

Monitoring and Alerting:
    - Real-time performance dashboards
    - Configurable alert thresholds
    - Performance regression detection
    - Resource utilization warnings
    - Automatic optimization recommendations
    - Performance trend analysis

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .profiling import (
    PerformanceProfiler,
    OptimizationConfig,
    MemoryOptimizer,
    LatencyTracker,
    GracefulDegradation,
    get_profiler,
    profile_operation,
    get_memory_usage,
    optimize_memory,
    MemoryContext
)

from .benchmarking import (
    BenchmarkSuite,
    HardwareAssessment
)

__all__ = [
    'PerformanceProfiler',
    'OptimizationConfig', 
    'MemoryOptimizer',
    'LatencyTracker',
    'GracefulDegradation',
    'get_profiler',
    'profile_operation',
    'get_memory_usage',
    'optimize_memory',
    'MemoryContext',
    'BenchmarkSuite',
    'HardwareAssessment'
]

__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"