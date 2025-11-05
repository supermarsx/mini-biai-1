#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite
================================

A complete benchmarking framework for measuring tok/s, TTFT, memory usage, energy consumption,
spike rates, and retrieval performance with stress testing and regression detection.

Target Metrics:
- ≥30 tok/s on RTX 4090
- <5ms TTFT
- 5-15% spike rate
- <20ms retrieval on 1M entries

Usage:
    python comprehensive_benchmark_suite.py --config config.yaml --output results.json
    python comprehensive_benchmark_suite.py --quick-test --target tok_throughput
    python comprehensive_benchmark_suite.py --stress-test --duration 600
"""

import asyncio
import time
import json
import statistics
import numpy as np
import torch
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import logging
from collections import deque
import warnings
import gc
import os
import subprocess
import platform

# Optional imports
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    warnings.warn("GPUtil not available - GPU metrics will be limited")

try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("Transformers not available - tok/s tests will be limited")


@dataclass
class BenchmarkConfig:
    """Configuration for comprehensive benchmarking."""
    # Target metrics and thresholds
    target_tok_s: float = 30.0  # ≥30 tok/s on RTX 4090
    target_ttft_ms: float = 5.0  # <5ms TTFT
    target_spike_rate_range: Tuple[float, float] = (5.0, 15.0)  # 5-15% spike rate
    target_retrieval_ms: float = 20.0  # <20ms on 1M entries
    
    # Test parameters
    iterations: int = 1000
    warmup_iterations: int = 100
    timeout_seconds: int = 300
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    
    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    test_text_lengths: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    
    # Retrieval test configuration
    retrieval_index_size: int = 1000000  # 1M entries
    retrieval_queries: int = 1000
    retrieval_top_k: int = 10
    
    # Stress test parameters
    stress_concurrency_levels: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    stress_duration_seconds: int = 300
    
    # Monitoring configuration
    monitoring_interval_ms: int = 100
    memory_sample_interval_ms: int = 500
    
    # Energy monitoring
    enable_energy_monitoring: bool = True
    energy_sample_interval_ms: int = 1000
    
    # Output configuration
    output_dir: str = "benchmark_results"
    save_detailed_logs: bool = True
    generate_html_report: bool = True
    
    # Regression detection
    baseline_file: Optional[str] = None
    regression_threshold_percent: float = 10.0


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    # Timing metrics
    mean_ms: float = 0.0
    median_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0
    
    # Throughput metrics
    operations_per_second: float = 0.0
    tokens_per_second: float = 0.0
    throughput_rps: float = 0.0
    
    # Resource metrics
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    cpu_avg_percent: float = 0.0
    gpu_util_avg_percent: float = 0.0
    
    # Energy metrics
    energy_consumption_joules: float = 0.0
    power_avg_watts: float = 0.0
    
    # Quality metrics
    spike_rate_percent: float = 0.0
    error_rate_percent: float = 0.0
    
    # Target assessment
    meets_targets: bool = False
    target_scores: Dict[str, bool] = field(default_factory=dict)


class SystemMonitor:
    """Real-time system resource monitoring."""
    
    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.interval_seconds = interval_ms / 1000.0
        
        self.monitoring = False
        self.monitor_thread = None
        
        # Resource tracking
        self.memory_samples = deque(maxlen=10000)
        self.cpu_samples = deque(maxlen=10000)
        self.gpu_samples = deque(maxlen=10000) if HAS_GPUTIL else None
        self.power_samples = deque(maxlen=10000)
        
        # Process info
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info("System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logging.info("System monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Memory monitoring
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = self.process.memory_percent()
                
                self.memory_samples.append((time.time(), memory_mb, memory_percent))
                
                # CPU monitoring
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append((time.time(), cpu_percent))
                
                # GPU monitoring
                if HAS_GPUTIL and torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            gpu_util = gpu.load * 100
                            gpu_memory = gpu.memoryUsed
                            self.gpu_samples.append((time.time(), gpu_util, gpu_memory))
                    except Exception:
                        pass
                
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                logging.warning(f"Monitoring error: {e}")
                time.sleep(self.interval_seconds)
                
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        current_time = time.time()
        
        # Memory metrics
        if self.memory_samples:
            latest_memory = self.memory_samples[-1]
            memory_mb = latest_memory[1]
            memory_percent = latest_memory[2]
        else:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            memory_percent = self.process.memory_percent()
            
        # CPU metrics
        if self.cpu_samples:
            latest_cpu = self.cpu_samples[-1]
            cpu_percent = latest_cpu[1]
        else:
            cpu_percent = self.process.cpu_percent()
            
        # GPU metrics
        gpu_util = 0.0
        gpu_memory = 0.0
        if HAS_GPUTIL and torch.cuda.is_available() and self.gpu_samples:
            try:
                latest_gpu = self.gpu_samples[-1]
                gpu_util = latest_gpu[1]
                gpu_memory = latest_gpu[2]
            except Exception:
                pass
                
        return {
            'memory_mb': memory_mb,
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'gpu_util_percent': gpu_util,
            'gpu_memory_mb': gpu_memory
        }
        
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics over monitoring period."""
        if not self.memory_samples:
            return {}
            
        # Memory metrics
        memory_values = [sample[1] for sample in self.memory_samples]  # MB
        memory_percent_values = [sample[2] for sample in self.memory_samples]
        
        # CPU metrics
        cpu_values = [sample[1] for sample in self.cpu_samples] if self.cpu_samples else []
        
        # GPU metrics
        gpu_util_values = []
        gpu_memory_values = []
        if self.gpu_samples:
            gpu_util_values = [sample[1] for sample in self.gpu_samples]
            gpu_memory_values = [sample[2] for sample in self.gpu_samples]
            
        return {
            'memory_peak_mb': max(memory_values) if memory_values else 0.0,
            'memory_avg_mb': statistics.mean(memory_values) if memory_values else 0.0,
            'memory_std_mb': statistics.stdev(memory_values) if len(memory_values) > 1 else 0.0,
            'memory_avg_percent': statistics.mean(memory_percent_values) if memory_percent_values else 0.0,
            'cpu_avg_percent': statistics.mean(cpu_values) if cpu_values else 0.0,
            'cpu_peak_percent': max(cpu_values) if cpu_values else 0.0,
            'gpu_util_avg_percent': statistics.mean(gpu_util_values) if gpu_util_values else 0.0,
            'gpu_util_peak_percent': max(gpu_util_values) if gpu_util_values else 0.0,
            'gpu_memory_avg_mb': statistics.mean(gpu_memory_values) if gpu_memory_values else 0.0,
            'gpu_memory_peak_mb': max(gpu_memory_values) if gpu_memory_values else 0.0,
        }


class EnergyMonitor:
    """Energy consumption monitoring using system power sensors."""
    
    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.interval_seconds = interval_ms / 1000.0
        
        self.monitoring = False
        self.monitor_thread = None
        
        # Energy tracking
        self.power_samples = deque(maxlen=10000)
        self.energy_start = None
        
        # Platform-specific power monitoring
        self.power_method = self._detect_power_method()
        
    def _detect_power_method(self) -> str:
        """Detect available power monitoring method."""
        if platform.system() == "Linux":
            # Try to use /sys/class/power_supply
            if Path("/sys/class/power_supply/BAT0/capacity").exists():
                return "battery"
            # Try RAPL for Intel CPUs
            if Path("/sys/class/powercap/intel-rapl").exists():
                return "rapl"
        elif platform.system() == "Darwin":
            # macOS with Intel Power Gadget or similar
            try:
                result = subprocess.run(
                    ["which", "powermetrics"], 
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    return "powermetrics"
            except Exception:
                pass
                
        return "none"
        
    def start_monitoring(self):
        """Start energy monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.energy_start = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logging.info(f"Energy monitoring started using {self.power_method}")
        
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and calculate energy metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        if not self.power_samples:
            return {}
            
        end_time = time.time()
        total_duration = end_time - (self.energy_start or end_time)
        
        # Calculate aggregate metrics
        power_values = [sample[1] for sample in self.power_samples]
        avg_power = statistics.mean(power_values) if power_values else 0.0
        
        # Energy in joules (assuming power in watts)
        energy_joules = avg_power * total_duration
        
        logging.info(f"Energy monitoring stopped: {energy_joules:.2f}J over {total_duration:.1f}s")
        
        return {
            'energy_consumption_joules': energy_joules,
            'power_avg_watts': avg_power,
            'power_peak_watts': max(power_values) if power_values else 0.0,
            'power_min_watts': min(power_values) if power_values else 0.0,
            'monitoring_duration_seconds': total_duration
        }
        
    def _monitor_loop(self):
        """Main power monitoring loop."""
        while self.monitoring:
            try:
                power_watts = self._read_power_sensor()
                if power_watts > 0:
                    self.power_samples.append((time.time(), power_watts))
                    
                time.sleep(self.interval_seconds)
                
            except Exception as e:
                logging.warning(f"Energy monitoring error: {e}")
                time.sleep(self.interval_seconds)
                
    def _read_power_sensor(self) -> float:
        """Read power consumption from sensor."""
        if self.power_method == "battery":
            try:
                # Read battery discharge rate (approximate)
                discharge_rate = 0.0
                voltage_path = Path("/sys/class/power_supply/BAT0/voltage_now")
                current_path = Path("/sys/class/power_supply/BAT0/current_now")
                
                if voltage_path.exists() and current_path.exists():
                    voltage_v = float(voltage_path.read_text().strip()) / 1000000  # µV to V
                    current_a = float(current_path.read_text().strip()) / 1000000  # µA to A
                    discharge_rate = voltage_v * current_a
                    
                return discharge_rate
            except Exception:
                return 0.0
                
        elif self.power_method == "rapl":
            try:
                # Read RAPL energy for CPU package
                rapl_path = Path("/sys/class/powercap/intel-rapl:0:0/energy_uj")
                if rapl_path.exists():
                    energy_uj = float(rapl_path.read_text().strip())
                    return energy_uj / 1000000  # µJ to J
            except Exception:
                pass
                
        elif self.power_method == "powermetrics":
            # This would require external tool integration
            pass
            
        # Fallback: estimate based on CPU usage
        try:
            cpu_percent = psutil.cpu_percent()
            # Rough estimate: modern CPUs can consume 15-100W
            return (cpu_percent / 100.0) * 50.0  # Assume 50W max
        except Exception:
            return 0.0


class ComprehensiveBenchmark:
    """Main comprehensive benchmarking suite."""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.setup_logging()
        
        # Initialize monitors
        self.system_monitor = SystemMonitor(self.config.monitoring_interval_ms)
        self.energy_monitor = EnergyMonitor(self.config.energy_sample_interval_ms)
        
        # Results storage
        self.results = {}
        self.baseline_results = None
        
        # Load baseline if specified
        if self.config.baseline_file and Path(self.config.baseline_file).exists():
            self.baseline_results = self._load_baseline()
            
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        logging.info("Comprehensive Benchmark Suite initialized")
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('benchmark_suite.log')
            ]
        )
        
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline results for regression testing."""
        try:
            with open(self.config.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load baseline: {e}")
            return {}
            
    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save results to file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to {filepath}")
        
    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentiles for a list of values."""
        if not values:
            return {}
            
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'mean_ms': statistics.mean(sorted_values),
            'median_ms': statistics.median(sorted_values),
            'std_ms': statistics.stdev(sorted_values) if n > 1 else 0.0,
            'min_ms': min(sorted_values),
            'max_ms': max(sorted_values),
            'p50_ms': np.percentile(sorted_values, 50),
            'p95_ms': np.percentile(sorted_values, 95),
            'p99_ms': np.percentile(sorted_values, 99),
            'p999_ms': np.percentile(sorted_values, 99.9)
        }
        
    def _calculate_spike_rate(self, latencies: List[float], threshold_multiplier: float = 3.0) -> float:
        """Calculate spike rate as percentage of outliers."""
        if not latencies or len(latencies) < 10:
            return 0.0
            
        mean_latency = statistics.mean(latencies)
        std_latency = statistics.stdev(latencies)
        threshold = mean_latency + (threshold_multiplier * std_latency)
        
        spikes = [lat for lat in latencies if lat > threshold]
        spike_rate = (len(spikes) / len(latencies)) * 100
        
        return min(spike_rate, 100.0)  # Cap at 100%
        
    def _assess_targets(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Assess if metrics meet target thresholds."""
        scores = {}
        
        # Tok/s assessment
        if 'tokens_per_second' in metrics:
            scores['tok_s'] = metrics['tokens_per_second'] >= self.config.target_tok_s
            
        # TTFT assessment
        if 'p50_ms' in metrics:
            scores['ttft'] = metrics['p50_ms'] <= self.config.target_ttft_ms
            
        # Spike rate assessment
        if 'spike_rate_percent' in metrics:
            min_rate, max_rate = self.config.target_spike_rate_range
            scores['spike_rate'] = min_rate <= metrics['spike_rate_percent'] <= max_rate
            
        # Retrieval assessment
        if 'p95_ms' in metrics and 'retrieval' in metrics:
            scores['retrieval'] = metrics['p95_ms'] <= self.config.target_retrieval_ms
            
        return scores
        
    async def benchmark_token_throughput(self) -> PerformanceMetrics:
        """Benchmark tokens per second performance."""
        logging.info("Starting token throughput benchmark...")
        
        if not HAS_TRANSFORMERS:
            logging.warning("Transformers not available - using synthetic token benchmark")
            return await self._synthetic_token_benchmark()
            
        # Initialize model and tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            model = AutoModel.from_pretrained(self.config.model_name)
            
            if torch.cuda.is_available():
                model = model.cuda()
                
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return await self._synthetic_token_benchmark()
            
        # Test data preparation
        test_texts = []
        for length in self.config.test_text_lengths:
            test_text = " ".join(["word"] * length)
            test_texts.append(test_text)
            
        latencies = []
        tokens_per_second_values = []
        
        # Start monitoring
        self.system_monitor.start_monitoring()
        if self.config.enable_energy_monitoring:
            self.energy_monitor.start_monitoring()
            
        # Warmup
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = model(**tokenizer(test_texts[0], return_tensors="pt").to(model.device))
                
        # Main benchmark
        for i, text in enumerate(test_texts):
            for _ in range(max(1, self.config.iterations // len(test_texts))):
                start_time = time.perf_counter()
                
                try:
                    with torch.no_grad():
                        inputs = tokenizer(text, return_tensors="pt")
                        if torch.cuda.is_available():
                            inputs = {k: v.cuda() for k, v in inputs.items()}
                            
                        outputs = model(**inputs)
                        
                    end_time = time.perf_counter()
                    latency_ms = (end_time - start_time) * 1000
                    
                    # Calculate tokens per second
                    tokens = len(tokenizer.encode(text))
                    tok_s = tokens / ((end_time - start_time))
                    
                    latencies.append(latency_ms)
                    tokens_per_second_values.append(tok_s)
                    
                except Exception as e:
                    logging.warning(f"Token benchmark iteration failed: {e}")
                    continue
                    
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        energy_metrics = {}
        if self.config.enable_energy_monitoring:
            energy_metrics = self.energy_monitor.stop_monitoring()
            
        # Calculate metrics
        latency_stats = self._calculate_percentiles(latencies)
        
        metrics = PerformanceMetrics()
        metrics.mean_ms = latency_stats.get('mean_ms', 0.0)
        metrics.median_ms = latency_stats.get('median_ms', 0.0)
        metrics.std_ms = latency_stats.get('std_ms', 0.0)
        metrics.min_ms = latency_stats.get('min_ms', 0.0)
        metrics.max_ms = latency_stats.get('max_ms', 0.0)
        metrics.p50_ms = latency_stats.get('p50_ms', 0.0)
        metrics.p95_ms = latency_stats.get('p95_ms', 0.0)
        metrics.p99_ms = latency_stats.get('p99_ms', 0.0)
        
        metrics.tokens_per_second = statistics.mean(tokens_per_second_values) if tokens_per_second_values else 0.0
        metrics.throughput_rps = len(latencies) / (max(latencies) / 1000) if latencies else 0.0
        
        # Spike rate
        metrics.spike_rate_percent = self._calculate_spike_rate(latencies)
        
        # Resource metrics
        system_metrics = self.system_monitor.get_aggregate_metrics()
        metrics.memory_peak_mb = system_metrics.get('memory_peak_mb', 0.0)
        metrics.memory_avg_mb = system_metrics.get('memory_avg_mb', 0.0)
        metrics.cpu_avg_percent = system_metrics.get('cpu_avg_percent', 0.0)
        metrics.gpu_util_avg_percent = system_metrics.get('gpu_util_avg_percent', 0.0)
        
        # Energy metrics
        metrics.energy_consumption_joules = energy_metrics.get('energy_consumption_joules', 0.0)
        metrics.power_avg_watts = energy_metrics.get('power_avg_watts', 0.0)
        
        # Target assessment
        metrics.target_scores = self._assess_targets(asdict(metrics))
        metrics.meets_targets = all(metrics.target_scores.values())
        
        logging.info(f"Token throughput: {metrics.tokens_per_second:.1f} tok/s, TTFT: {metrics.p50_ms:.2f}ms")
        
        return metrics
        
    async def _synthetic_token_benchmark(self) -> PerformanceMetrics:
        """Synthetic token benchmark when transformers is not available."""
        logging.info("Running synthetic token benchmark...")
        
        latencies = []
        
        self.system_monitor.start_monitoring()
        if self.config.enable_energy_monitoring:
            self.energy_monitor.start_monitoring()
            
        # Warmup
        for _ in range(self.config.warmup_iterations):
            tokens = sum(i * i for i in range(1000))
            
        # Main benchmark
        for _ in range(self.config.iterations):
            start_time = time.perf_counter()
            
            # Simulate token processing
            tokens = sum(i * i for i in range(500))
            synthetic_output = [tokens] * 100  # Simulate token generation
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            latencies.append(latency_ms)
            
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        energy_metrics = {}
        if self.config.enable_energy_monitoring:
            energy_metrics = self.energy_monitor.stop_monitoring()
            
        # Calculate synthetic tok/s based on computation
        avg_latency = statistics.mean(latencies) if latencies else 1.0
        synthetic_tok_s = 1000 / avg_latency * 500  # Approximate tokens per second
        
        latency_stats = self._calculate_percentiles(latencies)
        
        metrics = PerformanceMetrics()
        metrics.mean_ms = latency_stats.get('mean_ms', 0.0)
        metrics.median_ms = latency_stats.get('median_ms', 0.0)
        metrics.p50_ms = latency_stats.get('p50_ms', 0.0)
        metrics.p95_ms = latency_stats.get('p95_ms', 0.0)
        metrics.tokens_per_second = synthetic_tok_s
        
        system_metrics = self.system_monitor.get_aggregate_metrics()
        metrics.memory_peak_mb = system_metrics.get('memory_peak_mb', 0.0)
        metrics.cpu_avg_percent = system_metrics.get('cpu_avg_percent', 0.0)
        
        metrics.energy_consumption_joules = energy_metrics.get('energy_consumption_joules', 0.0)
        metrics.power_avg_watts = energy_metrics.get('power_avg_watts', 0.0)
        
        metrics.spike_rate_percent = self._calculate_spike_rate(latencies)
        
        return metrics
        
    async def benchmark_retrieval_performance(self) -> PerformanceMetrics:
        """Benchmark retrieval system performance on large index."""
        logging.info(f"Starting retrieval benchmark with {self.config.retrieval_index_size} entries...")
        
        # Simulate large retrieval index
        import random
        random.seed(42)
        
        # Create simulated index data
        index_data = []
        for i in range(self.config.retrieval_index_size):
            # Create random vectors and metadata
            vector = np.random.rand(768).astype(np.float32)  # 768-dim vectors
            metadata = {
                'id': i,
                'text': f"Document {i} with content " + " ".join([f"word{j}" for j in range(random.randint(10, 100))]),
                'category': random.choice(['science', 'tech', 'sports', 'news', 'finance']),
                'timestamp': time.time() - random.randint(0, 86400 * 365)
            }
            index_data.append((vector, metadata))
            
        # Simulate retrieval queries
        queries = []
        for i in range(self.config.retrieval_queries):
            query_vector = np.random.rand(768).astype(np.float32)
            queries.append(query_vector)
            
        latencies = []
        self.system_monitor.start_monitoring()
        if self.config.enable_energy_monitoring:
            self.energy_monitor.start_monitoring()
            
        # Simulate vector similarity search
        def simulate_retrieval(query_vector, top_k=10):
            # Simulate cosine similarity computation
            similarities = []
            for i, (vector, metadata) in enumerate(index_data):
                # Simulate dot product computation
                similarity = float(np.dot(query_vector, vector))
                similarities.append((similarity, metadata))
                
            # Sort by similarity and return top_k
            similarities.sort(reverse=True)
            return similarities[:top_k]
            
        # Warmup
        for _ in range(10):
            _ = simulate_retrieval(queries[0])
            
        # Main benchmark
        for i, query_vector in enumerate(queries):
            start_time = time.perf_counter()
            
            try:
                results = simulate_retrieval(query_vector, self.config.retrieval_top_k)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                latencies.append(latency_ms)
                
            except Exception as e:
                logging.warning(f"Retrieval query {i} failed: {e}")
                continue
                
        # Stop monitoring
        self.system_monitor.stop_monitoring()
        energy_metrics = {}
        if self.config.enable_energy_monitoring:
            energy_metrics = self.energy_monitor.stop_monitoring()
            
        # Calculate metrics
        latency_stats = self._calculate_percentiles(latencies)
        
        metrics = PerformanceMetrics()
        metrics.mean_ms = latency_stats.get('mean_ms', 0.0)
        metrics.median_ms = latency_stats.get('median_ms', 0.0)
        metrics.p50_ms = latency_stats.get('p50_ms', 0.0)
        metrics.p95_ms = latency_stats.get('p95_ms', 0.0)
        metrics.p99_ms = latency_stats.get('p99_ms', 0.0)
        metrics.max_ms = latency_stats.get('max_ms', 0.0)
        
        # Retrieval-specific throughput
        if latencies:
            total_queries = len(latencies)
            total_time = sum(latencies) / 1000.0  # Convert to seconds
            metrics.throughput_rps = total_queries / total_time if total_time > 0 else 0.0
            
        system_metrics = self.system_monitor.get_aggregate_metrics()
        metrics.memory_peak_mb = system_metrics.get('memory_peak_mb', 0.0)
        metrics.cpu_avg_percent = system_metrics.get('cpu_avg_percent', 0.0)
        
        metrics.energy_consumption_joules = energy_metrics.get('energy_consumption_joules', 0.0)
        metrics.power_avg_watts = energy_metrics.get('power_avg_watts', 0.0)
        
        metrics.spike_rate_percent = self._calculate_spike_rate(latencies)
        
        # Target assessment for retrieval
        metrics.target_scores = {'retrieval': metrics.p95_ms <= self.config.target_retrieval_ms}
        metrics.meets_targets = all(metrics.target_scores.values())
        
        logging.info(f"Retrieval performance: {metrics.p95_ms:.1f}ms p95, {metrics.throughput_rps:.1f} queries/sec")
        
        return metrics
        
    async def run_stress_test(self) -> Dict[str, Any]:
        """Run comprehensive stress test."""
        logging.info(f"Starting stress test for {self.config.stress_duration_seconds} seconds...")
        
        stress_results = {}
        self.system_monitor.start_monitoring()
        
        for concurrency in self.config.stress_concurrency_levels:
            logging.info(f"Testing concurrency level: {concurrency}")
            
            start_time = time.perf_counter()
            completed_requests = 0
            failed_requests = 0
            request_latencies = []
            
            # Simulate concurrent load
            async def simulate_request():
                nonlocal completed_requests, failed_requests
                request_start = time.perf_counter()
                
                try:
                    # Simulate variable workload
                    workload_time = np.random.exponential(0.01)  # Average 10ms
                    await asyncio.sleep(workload_time)
                    
                    # Add some CPU work
                    _ = sum(i * i for i in range(1000))
                    
                    request_end = time.perf_counter()
                    latency_ms = (request_end - request_start) * 1000
                    
                    completed_requests += 1
                    request_latencies.append(latency_ms)
                    
                except Exception as e:
                    failed_requests += 1
                    logging.warning(f"Request failed: {e}")
                    
            # Run concurrent requests
            tasks = []
            end_time = start_time + self.config.stress_duration_seconds
            
            while time.perf_counter() < end_time:
                # Create batch of concurrent tasks
                batch_tasks = [
                    simulate_request() for _ in range(min(concurrency, 100))
                ]
                
                if batch_tasks:
                    await asyncio.gather(*batch_tasks, return_exceptions=True)
                    tasks.extend(batch_tasks)
                    
                # Small delay between batches
                await asyncio.sleep(0.1)
                
            total_time = time.perf_counter() - start_time
            
            # Calculate metrics
            latency_stats = self._calculate_percentiles(request_latencies) if request_latencies else {}
            
            stress_results[f"concurrency_{concurrency}"] = {
                'concurrency_level': concurrency,
                'total_time_seconds': total_time,
                'completed_requests': completed_requests,
                'failed_requests': failed_requests,
                'total_requests': completed_requests + failed_requests,
                'throughput_rps': completed_requests / total_time if total_time > 0 else 0,
                'error_rate_percent': (failed_requests / (completed_requests + failed_requests) * 100) if (completed_requests + failed_requests) > 0 else 0,
                'latency_stats': latency_stats,
                'system_stable': (completed_requests / (completed_requests + failed_requests) > 0.95 if (completed_requests + failed_requests) > 0 else False)
            }
            
            logging.info(f"Concurrency {concurrency}: {completed_requests} completed, "
                        f"{stress_results[f'concurrency_{concurrency}']['throughput_rps']:.1f} req/s")
                        
        self.system_monitor.stop_monitoring()
        
        return stress_results
        
    async def run_regression_detection(self) -> Dict[str, Any]:
        """Run performance regression detection against baseline."""
        if not self.baseline_results:
            logging.warning("No baseline available for regression testing")
            return {'regression_detected': False, 'message': 'No baseline loaded'}
            
        logging.info("Running regression detection against baseline...")
        
        # Run current benchmarks
        current_token_metrics = await self.benchmark_token_throughput()
        current_retrieval_metrics = await self.benchmark_retrieval_performance()
        
        regression_results = {
            'timestamp': time.time(),
            'baseline_file': self.config.baseline_file,
            'regression_threshold_percent': self.config.regression_threshold_percent,
            'current_results': {
                'token_throughput': asdict(current_token_metrics),
                'retrieval_performance': asdict(current_retrieval_metrics)
            },
            'baseline_results': self.baseline_results,
            'regression_analysis': {}
        }
        
        # Compare token throughput
        if 'token_throughput' in self.baseline_results:
            baseline_tok_s = self.baseline_results['token_throughput'].get('tokens_per_second', 0)
            current_tok_s = current_token_metrics.tokens_per_second
            
            if baseline_tok_s > 0:
                change_percent = ((current_tok_s - baseline_tok_s) / baseline_tok_s) * 100
                regression_results['regression_analysis']['token_throughput'] = {
                    'baseline_tok_s': baseline_tok_s,
                    'current_tok_s': current_tok_s,
                    'change_percent': change_percent,
                    'regression_detected': change_percent < -self.config.regression_threshold_percent
                }
                
        # Compare retrieval performance
        if 'retrieval_performance' in self.baseline_results:
            baseline_p95 = self.baseline_results['retrieval_performance'].get('p95_ms', 0)
            current_p95 = current_retrieval_metrics.p95_ms
            
            if baseline_p95 > 0:
                change_percent = ((current_p95 - baseline_p95) / baseline_p95) * 100
                regression_results['regression_analysis']['retrieval_performance'] = {
                    'baseline_p95_ms': baseline_p95,
                    'current_p95_ms': current_p95,
                    'change_percent': change_percent,
                    'regression_detected': change_percent > self.config.regression_threshold_percent
                }
                
        # Overall assessment
        regressions = [
            analysis.get('regression_detected', False)
            for analysis in regression_results['regression_analysis'].values()
        ]
        
        regression_results['regression_detected'] = any(regressions)
        
        logging.info(f"Regression detection complete: {regression_results['regression_detected']}")
        
        return regression_results
        
    async def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Run complete comprehensive benchmark suite."""
        logging.info("Starting comprehensive benchmark suite...")
        
        suite_results = {
            'timestamp': time.time(),
            'config': asdict(self.config),
            'system_info': self._get_system_info(),
            'benchmarks': {},
            'stress_test': {},
            'regression_analysis': {},
            'overall_assessment': {}
        }
        
        try:
            # Core benchmarks
            logging.info("Running core performance benchmarks...")
            
            # Token throughput benchmark
            token_metrics = await self.benchmark_token_throughput()
            suite_results['benchmarks']['token_throughput'] = asdict(token_metrics)
            
            # Retrieval performance benchmark
            retrieval_metrics = await self.benchmark_retrieval_performance()
            suite_results['benchmarks']['retrieval_performance'] = asdict(retrieval_metrics)
            
            # Stress test
            logging.info("Running stress test...")
            stress_results = await self.run_stress_test()
            suite_results['stress_test'] = stress_results
            
            # Regression detection
            if self.baseline_results:
                logging.info("Running regression detection...")
                regression_results = await self.run_regression_detection()
                suite_results['regression_analysis'] = regression_results
                
            # Overall assessment
            suite_results['overall_assessment'] = self._assess_overall_performance(suite_results)
            
        except Exception as e:
            logging.error(f"Benchmark suite failed: {e}")
            suite_results['error'] = str(e)
            
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"comprehensive_benchmark_{timestamp}.json"
        self._save_results(suite_results, results_file)
        
        # Generate HTML report if requested
        if self.config.generate_html_report:
            await self._generate_html_report(suite_results, f"benchmark_report_{timestamp}.html")
            
        return suite_results
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'torch_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
        return info
        
    def _assess_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system performance."""
        assessment = {
            'targets_met': 0,
            'total_targets': 0,
            'performance_grade': 'unknown',
            'recommendations': []
        }
        
        # Check target metrics
        token_results = results.get('benchmarks', {}).get('token_throughput', {})
        retrieval_results = results.get('benchmarks', {}).get('retrieval_performance', {})
        
        # Tok/s target
        if 'tokens_per_second' in token_results:
            assessment['total_targets'] += 1
            if token_results['tokens_per_second'] >= self.config.target_tok_s:
                assessment['targets_met'] += 1
            else:
                assessment['recommendations'].append(
                    f"Token throughput {token_results['tokens_per_second']:.1f} tok/s below target {self.config.target_tok_s} tok/s"
                )
                
        # TTFT target
        if 'p50_ms' in token_results:
            assessment['total_targets'] += 1
            if token_results['p50_ms'] <= self.config.target_ttft_ms:
                assessment['targets_met'] += 1
            else:
                assessment['recommendations'].append(
                    f"TTFT {token_results['p50_ms']:.2f}ms above target {self.config.target_ttft_ms}ms"
                )
                
        # Spike rate target
        if 'spike_rate_percent' in token_results:
            assessment['total_targets'] += 1
            spike_rate = token_results['spike_rate_percent']
            if self.config.target_spike_rate_range[0] <= spike_rate <= self.config.target_spike_rate_range[1]:
                assessment['targets_met'] += 1
            else:
                assessment['recommendations'].append(
                    f"Spike rate {spike_rate:.1f}% outside target range {self.config.target_spike_rate_range[0]}-{self.config.target_spike_rate_range[1]}%"
                )
                
        # Retrieval target
        if 'p95_ms' in retrieval_results:
            assessment['total_targets'] += 1
            if retrieval_results['p95_ms'] <= self.config.target_retrieval_ms:
                assessment['targets_met'] += 1
            else:
                assessment['recommendations'].append(
                    f"Retrieval p95 {retrieval_results['p95_ms']:.1f}ms above target {self.config.target_retrieval_ms}ms"
                )
                
        # Calculate grade
        if assessment['total_targets'] > 0:
            pass_rate = assessment['targets_met'] / assessment['total_targets']
            if pass_rate >= 0.9:
                assessment['performance_grade'] = 'excellent'
            elif pass_rate >= 0.8:
                assessment['performance_grade'] = 'good'
            elif pass_rate >= 0.6:
                assessment['performance_grade'] = 'acceptable'
            else:
                assessment['performance_grade'] = 'needs_improvement'
                
        return assessment
        
    async def _generate_html_report(self, results: Dict[str, Any], filename: str):
        """Generate HTML report from results."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric h3 {{ margin-top: 0; color: #333; }}
        .target-met {{ background-color: #d4edda; }}
        .target-missed {{ background-color: #f8d7da; }}
        .grade-excellent {{ background-color: #d4edda; color: #155724; }}
        .grade-good {{ background-color: #d1ecf1; color: #0c5460; }}
        .grade-acceptable {{ background-color: #fff3cd; color: #856404; }}
        .grade-needs-improvement {{ background-color: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Comprehensive Benchmark Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
        
        # Add overall assessment
        assessment = results.get('overall_assessment', {})
        grade_class = f"grade-{assessment.get('performance_grade', 'unknown')}"
        html_content += f"""
    <div class="metric {grade_class}">
        <h2>Overall Performance Assessment</h2>
        <p><strong>Grade:</strong> {assessment.get('performance_grade', 'unknown').upper()}</p>
        <p><strong>Targets Met:</strong> {assessment.get('targets_met', 0)}/{assessment.get('total_targets', 0)}</p>
    </div>
"""
        
        # Add benchmark results
        benchmarks = results.get('benchmarks', {})
        for name, data in benchmarks.items():
            target_met = data.get('meets_targets', False)
            css_class = "target-met" if target_met else "target-missed"
            
            html_content += f"""
    <div class="metric {css_class}">
        <h3>{name.replace('_', ' ').title()}</h3>
        <p><strong>Tokens per second:</strong> {data.get('tokens_per_second', 0):.1f}</p>
        <p><strong>TTFT (p50):</strong> {data.get('p50_ms', 0):.2f}ms</p>
        <p><strong>Latency (p95):</strong> {data.get('p95_ms', 0):.2f}ms</p>
        <p><strong>Spike rate:</strong> {data.get('spike_rate_percent', 0):.1f}%</p>
        <p><strong>Peak memory:</strong> {data.get('memory_peak_mb', 0):.1f}MB</p>
        <p><strong>Energy consumption:</strong> {data.get('energy_consumption_joules', 0):.1f}J</p>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(html_content)
            
        logging.info(f"HTML report saved to {filepath}")


# CLI Interface
async def main():
    """Main CLI interface for comprehensive benchmarking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Suite")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test")
    parser.add_argument("--target", choices=["tok_throughput", "retrieval"], 
                       help="Run specific target benchmark")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test")
    parser.add_argument("--duration", type=int, default=300, help="Stress test duration")
    parser.add_argument("--baseline", help="Baseline file for regression testing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = BenchmarkConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
    # Override with command line args
    if args.output:
        config.output_dir = args.output
    if args.baseline:
        config.baseline_file = args.baseline
    if args.duration:
        config.stress_duration_seconds = args.duration
        
    # Initialize benchmark suite
    benchmark = ComprehensiveBenchmark(config)
    
    try:
        if args.quick_test:
            # Quick test mode
            if args.target == "tok_throughput":
                result = await benchmark.benchmark_token_throughput()
                print(f"Token throughput: {result.tokens_per_second:.1f} tok/s")
                print(f"TTFT: {result.p50_ms:.2f}ms")
                
            elif args.target == "retrieval":
                result = await benchmark.benchmark_retrieval_performance()
                print(f"Retrieval p95: {result.p95_ms:.2f}ms")
                
        elif args.stress_test:
            # Stress test mode
            result = await benchmark.run_stress_test()
            print("Stress test results:")
            for level, data in result.items():
                print(f"  {level}: {data['throughput_rps']:.1f} req/s, "
                      f"{data['error_rate_percent']:.1f}% errors")
                      
        else:
            # Full benchmark suite
            result = await benchmark.run_full_benchmark_suite()
            
            # Print summary
            print("\n" + "="*60)
            print("COMPREHENSIVE BENCHMARK SUITE RESULTS")
            print("="*60)
            
            assessment = result.get('overall_assessment', {})
            print(f"Overall Grade: {assessment.get('performance_grade', 'unknown').upper()}")
            print(f"Targets Met: {assessment.get('targets_met', 0)}/{assessment.get('total_targets', 0)}")
            
            benchmarks = result.get('benchmarks', {})
            for name, data in benchmarks.items():
                print(f"\n{name.replace('_', ' ').title()}:")
                if 'tokens_per_second' in data:
                    print(f"  Token Throughput: {data['tokens_per_second']:.1f} tok/s")
                    print(f"  TTFT (p50): {data['p50_ms']:.2f}ms")
                    print(f"  Spike Rate: {data['spike_rate_percent']:.1f}%")
                if 'p95_ms' in data:
                    print(f"  Latency (p95): {data['p95_ms']:.2f}ms")
                    
            print("\nDetailed results saved to output directory.")
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        logging.exception("Benchmark execution failed")
        raise


if __name__ == "__main__":
    asyncio.run(main())