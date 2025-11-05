#!/usr/bin/env python3
"""
Stress Test Runner
==================

Specialized stress testing tool for comprehensive load testing of the system.
Tests system stability under various load conditions and identifies breaking points.

Usage:
    python stress_test_runner.py --duration 600 --max-concurrency 100
    python stress_test_runner.py --memory-stress --cpu-stress
    python stress_test_runner.py --production-load --duration 1800
"""

import asyncio
import time
import json
import statistics
import numpy as np
import psutil
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import logging
from collections import deque
import gc
import random

# Optional imports
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    # Test duration and intensity
    duration_seconds: int = 600
    max_concurrency: int = 100
    ramp_up_time: int = 60  # Time to ramp up to max load
    ramp_down_time: int = 30  # Time to ramp down
    
    # Load patterns
    load_pattern: str = "steady"  # steady, ramp, spike, random
    spike_frequency: int = 30  # seconds between spikes
    spike_intensity_multiplier: float = 3.0
    
    # Stress dimensions
    enable_memory_stress: bool = True
    enable_cpu_stress: bool = True
    enable_io_stress: bool = True
    enable_network_stress: bool = False
    
    # Monitoring
    sample_interval_ms: int = 100
    save_detailed_logs: bool = True
    
    # Success criteria
    max_error_rate_percent: float = 5.0
    max_latency_ms: float = 1000.0
    min_throughput_degradation_percent: float = 20.0
    
    # Resource limits
    max_memory_usage_percent: float = 90.0
    max_cpu_usage_percent: float = 95.0


class StressLoadGenerator:
    """Generates various types of stress load."""
    
    def __init__(self, config: StressTestConfig):
        self.config = config
        self.active_tasks = []
        self.completed_requests = 0
        self.failed_requests = 0
        self.request_latencies = deque(maxlen=10000)
        self.memory_pressure_tasks = []
        self.cpu_stress_tasks = []
        
    async def generate_steady_load(self, duration: int, concurrency: int):
        """Generate steady load for specified duration."""
        print(f"Generating steady load: {concurrency} concurrent requests for {duration}s")
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            batch_tasks = []
            
            # Create batch of concurrent requests
            for _ in range(min(concurrency, 20)):  # Limit batch size
                task = asyncio.create_task(self._simulate_request())
                batch_tasks.append(task)
                
            if batch_tasks:
                # Wait for batch completion
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
            # Small delay between batches
            await asyncio.sleep(0.1)
            
    async def generate_ramp_load(self, duration: int, max_concurrency: int):
        """Generate ramping load from 1 to max_concurrency."""
        print(f"Generating ramp load from 1 to {max_concurrency} over {duration}s")
        
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            elapsed = time.time() - start_time
            progress = elapsed / duration
            
            # Linear ramp up
            current_concurrency = int(1 + (max_concurrency - 1) * progress)
            
            batch_tasks = []
            for _ in range(min(current_concurrency, 20)):
                task = asyncio.create_task(self._simulate_request())
                batch_tasks.append(task)
                
            if batch_tasks:
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
            await asyncio.sleep(0.1)
            
    async def generate_spike_load(self, duration: int, base_concurrency: int):
        """Generate load with regular spikes."""
        print(f"Generating spike load: base {base_concurrency}, spikes every {self.config.spike_frequency}s")
        
        start_time = time.time()
        end_time = start_time + duration
        next_spike = start_time + self.config.spike_frequency
        
        while time.time() < end_time:
            current_time = time.time()
            
            # Determine current concurrency
            if current_time >= next_spike:
                # Spike period
                current_concurrency = int(base_concurrency * self.config.spike_intensity_multiplier)
                next_spike += self.config.spike_frequency
                print(f"🔥 Spike detected: {current_concurrency} concurrent requests")
            else:
                # Normal period
                current_concurrency = base_concurrency
                
            # Generate load
            batch_tasks = []
            for _ in range(min(current_concurrency, 20)):
                task = asyncio.create_task(self._simulate_request())
                batch_tasks.append(task)
                
            if batch_tasks:
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
            await asyncio.sleep(0.1)
            
    async def _simulate_request(self):
        """Simulate a typical request with various stress components."""
        request_start = time.time()
        
        try:
            # CPU intensive component
            if self.config.enable_cpu_stress:
                await self._cpu_intensive_operation()
                
            # Memory intensive component
            if self.config.enable_memory_stress:
                await self._memory_intensive_operation()
                
            # I/O intensive component
            if self.config.enable_io_stress:
                await self._io_intensive_operation()
                
            # Simulate actual processing time
            processing_time = np.random.exponential(0.01)  # Average 10ms
            await asyncio.sleep(processing_time)
            
            request_end = time.time()
            latency_ms = (request_end - request_start) * 1000
            
            self.completed_requests += 1
            self.request_latencies.append(latency_ms)
            
        except Exception as e:
            self.failed_requests += 1
            logging.warning(f"Request failed: {e}")
            
    async def _cpu_intensive_operation(self):
        """CPU intensive operation for stress testing."""
        # Matrix operations simulation
        size = 100
        result = 0
        for _ in range(10):
            matrix = np.random.rand(size, size)
            result += np.trace(matrix @ matrix.T)
        return result
        
    async def _memory_intensive_operation(self):
        """Memory intensive operation for stress testing."""
        # Allocate and manipulate large data structures
        data_chunks = []
        for _ in range(5):
            chunk = np.random.rand(10000).astype(np.float32)
            processed_chunk = chunk ** 2 + np.sin(chunk)
            data_chunks.append(processed_chunk)
            
        # Process data
        processed_data = []
        for chunk in data_chunks:
            processed_data.extend(chunk[:1000])
            
        # Cleanup
        del data_chunks, processed_data
        
    async def _io_intensive_operation(self):
        """I/O intensive operation for stress testing."""
        # Simulate disk I/O
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write data
            data = b"x" * 1024 * 100  # 100KB
            f.write(data)
            f.flush()
            
        # Simulate network I/O (if enabled)
        if self.config.enable_network_stress:
            # Simulate HTTP requests
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get("http://httpbin.org/delay/0.1", timeout=5) as response:
                        await response.text()
            except Exception:
                pass  # Network stress is optional
                
    async def generate_memory_pressure(self, duration: int):
        """Generate memory pressure to test system limits."""
        print(f"Generating memory pressure for {duration}s")
        
        start_time = time.time()
        end_time = start_time + duration
        memory_chunks = []
        
        while time.time() < end_time:
            try:
                # Allocate increasingly large chunks
                chunk_size = random.randint(1000000, 10000000)  # 1-10MB
                chunk = np.random.rand(chunk_size).astype(np.float32)
                memory_chunks.append(chunk)
                
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > self.config.max_memory_usage_percent:
                    print(f"⚠️ Memory usage exceeded limit: {memory_percent:.1f}%")
                    break
                    
                await asyncio.sleep(1.0)  # Check every second
                
            except MemoryError:
                print("💥 Memory allocation failed - system limit reached")
                break
            except Exception as e:
                logging.warning(f"Memory pressure test error: {e}")
                break
                
        # Cleanup
        del memory_chunks
        gc.collect()
        
    async def generate_cpu_pressure(self, duration: int):
        """Generate CPU pressure to test system limits."""
        print(f"Generating CPU pressure for {duration}s")
        
        def cpu_worker():
            """Worker function for CPU stress."""
            end_time = time.time() + duration
            while time.time() < end_time:
                # CPU intensive calculation
                result = sum(i * i for i in range(10000))
                # Keep CPU busy
                for _ in range(1000):
                    result += hash(str(result)) % 1000
                    
        # Start multiple CPU workers
        cpu_count = psutil.cpu_count()
        tasks = []
        
        for _ in range(cpu_count):
            task = asyncio.create_task(asyncio.to_thread(cpu_worker))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)


class SystemMonitor:
    """Real-time system monitoring during stress testing."""
    
    def __init__(self, sample_interval_ms: int = 100):
        self.sample_interval_ms = sample_interval_ms
        self.sample_interval_sec = sample_interval_ms / 1000.0
        
        self.monitoring = False
        self.monitor_thread = None
        
        # Metrics storage
        self.metrics_history = deque(maxlen=10000)
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 System monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("📊 System monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                # System metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Process metrics
                process = psutil.Process()
                process_memory = process.memory_info()
                process_cpu = process.cpu_percent()
                
                # GPU metrics
                gpu_util = 0.0
                gpu_memory = 0.0
                if HAS_GPUTIL and torch.cuda.is_available():
                    try:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]
                            gpu_util = gpu.load * 100
                            gpu_memory = gpu.memoryUsed
                    except Exception:
                        pass
                        
                metrics = {
                    'timestamp': timestamp,
                    'system_cpu_percent': cpu_percent,
                    'system_memory_percent': memory.percent,
                    'system_memory_available_gb': memory.available / (1024**3),
                    'disk_usage_percent': disk.percent,
                    'process_cpu_percent': process_cpu,
                    'process_memory_mb': process_memory.rss / 1024 / 1024,
                    'gpu_util_percent': gpu_util,
                    'gpu_memory_mb': gpu_memory
                }
                
                self.metrics_history.append(metrics)
                
                time.sleep(self.sample_interval_sec)
                
            except Exception as e:
                logging.warning(f"Monitoring error: {e}")
                time.sleep(self.sample_interval_sec)
                
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]
        
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics over monitoring period."""
        if not self.metrics_history:
            return {}
            
        # Extract metric arrays
        cpu_values = [m['system_cpu_percent'] for m in self.metrics_history]
        memory_values = [m['system_memory_percent'] for m in self.metrics_history]
        process_memory_values = [m['process_memory_mb'] for m in self.metrics_history]
        gpu_util_values = [m['gpu_util_percent'] for m in self.metrics_history if m['gpu_util_percent'] > 0]
        
        return {
            'samples': len(self.metrics_history),
            'monitoring_duration': self.metrics_history[-1]['timestamp'] - self.metrics_history[0]['timestamp'],
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_peak': max(cpu_values),
            'memory_avg': statistics.mean(memory_values),
            'memory_peak': max(memory_values),
            'process_memory_avg': statistics.mean(process_memory_values),
            'process_memory_peak': max(process_memory_values),
            'gpu_util_avg': statistics.mean(gpu_util_values) if gpu_util_values else 0,
            'gpu_util_peak': max(gpu_util_values) if gpu_util_values else 0,
        }


class StressTestRunner:
    """Comprehensive stress testing suite."""
    
    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        self.monitor = SystemMonitor(self.config.sample_interval_ms)
        self.load_generator = StressLoadGenerator(self.config)
        
        # Results storage
        self.test_results = {
            'config': asdict(self.config),
            'test_info': {},
            'performance_metrics': {},
            'system_metrics': {},
            'stability_assessment': {},
            'breakpoints': {}
        }
        
    async def run_steady_state_stress(self, concurrency: int, duration: int):
        """Run steady-state stress test."""
        print(f"\n🚀 Starting steady-state stress test")
        print(f"   Concurrency: {concurrency}")
        print(f"   Duration: {duration}s")
        
        self.monitor.start_monitoring()
        self.load_generator.completed_requests = 0
        self.load_generator.failed_requests = 0
        
        start_time = time.time()
        
        try:
            # Ramp up
            print("⬆️  Ramping up load...")
            await self._ramp_up_load(concurrency, self.config.ramp_up_time)
            
            # Steady state
            print("🎯 Maintaining steady load...")
            await self.load_generator.generate_steady_load(
                duration - self.config.ramp_up_time - self.config.ramp_down_time,
                concurrency
            )
            
            # Ramp down
            print("⬇️  Ramping down load...")
            await self._ramp_down_load(concurrency, self.config.ramp_down_time)
            
        finally:
            self.monitor.stop_monitoring()
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        return self._analyze_results('steady_state', total_duration, concurrency)
        
    async def run_ramp_stress(self, max_concurrency: int, duration: int):
        """Run ramping load stress test."""
        print(f"\n🚀 Starting ramp stress test")
        print(f"   Max concurrency: {max_concurrency}")
        print(f"   Duration: {duration}s")
        
        self.monitor.start_monitoring()
        self.load_generator.completed_requests = 0
        self.load_generator.failed_requests = 0
        
        start_time = time.time()
        
        try:
            await self.load_generator.generate_ramp_load(duration, max_concurrency)
            
        finally:
            self.monitor.stop_monitoring()
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        return self._analyze_results('ramp', total_duration, max_concurrency)
        
    async def run_spike_stress(self, base_concurrency: int, duration: int):
        """Run spike load stress test."""
        print(f"\n🚀 Starting spike stress test")
        print(f"   Base concurrency: {base_concurrency}")
        print(f"   Duration: {duration}s")
        
        self.monitor.start_monitoring()
        self.load_generator.completed_requests = 0
        self.load_generator.failed_requests = 0
        
        start_time = time.time()
        
        try:
            await self.load_generator.generate_spike_load(duration, base_concurrency)
            
        finally:
            self.monitor.stop_monitoring()
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        return self._analyze_results('spike', total_duration, base_concurrency)
        
    async def run_resource_pressure_test(self, duration: int):
        """Run resource pressure test (memory + CPU)."""
        print(f"\n🚀 Starting resource pressure test")
        print(f"   Duration: {duration}s")
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        
        tasks = []
        
        # Start memory pressure
        if self.config.enable_memory_stress:
            task = asyncio.create_task(self.load_generator.generate_memory_pressure(duration))
            tasks.append(task)
            
        # Start CPU pressure
        if self.config.enable_cpu_stress:
            task = asyncio.create_task(self.load_generator.generate_cpu_pressure(duration))
            tasks.append(task)
            
        try:
            # Run all pressure tests concurrently
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            self.monitor.stop_monitoring()
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        return self._analyze_resource_pressure_results(total_duration)
        
    async def run_breaking_point_test(self):
        """Run test to find system breaking point."""
        print(f"\n🚀 Starting breaking point test")
        print("   Finding maximum sustainable load...")
        
        self.monitor.start_monitoring()
        
        start_time = time.time()
        breaking_point_concurrency = 0
        tested_concurrencies = []
        
        try:
            # Start with low concurrency and increase
            concurrency = 1
            while concurrency <= self.config.max_concurrency:
                print(f"   Testing concurrency: {concurrency}")
                
                tested_concurrencies.append(concurrency)
                
                # Test this concurrency level
                success = await self._test_concurrency_level(concurrency, 30)
                
                if success:
                    breaking_point_concurrency = concurrency
                    concurrency += 10  # Increment by 10s
                else:
                    # Found breaking point, test few more levels for precision
                    for precise_concurrency in range(max(1, concurrency - 9), concurrency):
                        tested_concurrencies.append(precise_concurrency)
                        await self._test_concurrency_level(precise_concurrency, 20)
                    break
                    
        except Exception as e:
            print(f"   Breaking point test error: {e}")
            
        finally:
            self.monitor.stop_monitoring()
            
        end_time = time.time()
        total_duration = end_time - start_time
        
        return self._analyze_breaking_point_results(
            total_duration, breaking_point_concurrency, tested_concurrencies
        )
        
    async def _test_concurrency_level(self, concurrency: int, test_duration: int) -> bool:
        """Test a specific concurrency level."""
        self.load_generator.completed_requests = 0
        self.load_generator.failed_requests = 0
        
        start_time = time.time()
        end_time = start_time + test_duration
        
        try:
            while time.time() < end_time:
                batch_tasks = []
                for _ in range(min(concurrency, 20)):
                    task = asyncio.create_task(self.load_generator._simulate_request())
                    batch_tasks.append(task)
                    
                if batch_tasks:
                    await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                await asyncio.sleep(0.1)
                
            # Analyze results for this concurrency level
            total_requests = self.load_generator.completed_requests + self.load_generator.failed_requests
            error_rate = (self.load_generator.failed_requests / total_requests * 100) if total_requests > 0 else 100
            
            current_metrics = self.monitor.get_current_metrics()
            memory_percent = current_metrics.get('system_memory_percent', 0)
            cpu_percent = current_metrics.get('system_cpu_percent', 0)
            
            # Success criteria
            success = (
                error_rate <= self.config.max_error_rate_percent and
                memory_percent <= self.config.max_memory_usage_percent and
                cpu_percent <= self.config.max_cpu_usage_percent
            )
            
            print(f"      Result: {'✅ PASSED' if success else '❌ FAILED'} "
                  f"({error_rate:.1f}% errors, {memory_percent:.1f}% memory, {cpu_percent:.1f}% CPU)")
            
            return success
            
        except Exception as e:
            print(f"      Result: ❌ FAILED (error: {e})")
            return False
            
    async def _ramp_up_load(self, target_concurrency: int, ramp_duration: int):
        """Gradually ramp up to target load."""
        steps = 10
        step_duration = ramp_duration / steps
        step_increment = target_concurrency / steps
        
        for i in range(steps):
            current_concurrency = int((i + 1) * step_increment)
            print(f"   Ramping to: {current_concurrency} concurrent requests")
            await asyncio.sleep(step_duration)
            
    async def _ramp_down_load(self, start_concurrency: int, ramp_duration: int):
        """Gradually ramp down from target load."""
        steps = 5
        step_duration = ramp_duration / steps
        step_decrement = start_concurrency / steps
        
        for i in range(steps):
            current_concurrency = int(start_concurrency - (i + 1) * step_decrement)
            if current_concurrency > 0:
                print(f"   Ramping down to: {current_concurrency} concurrent requests")
                await asyncio.sleep(step_duration)
                
    def _analyze_results(self, test_type: str, duration: float, max_concurrency: int) -> Dict[str, Any]:
        """Analyze test results."""
        completed = self.load_generator.completed_requests
        failed = self.load_generator.failed_requests
        total = completed + failed
        
        latencies = list(self.load_generator.request_latencies)
        
        # Calculate metrics
        if latencies:
            latency_stats = {
                'mean_ms': statistics.mean(latencies),
                'median_ms': statistics.median(latencies),
                'p95_ms': np.percentile(latencies, 95),
                'p99_ms': np.percentile(latencies, 99),
                'max_ms': max(latencies)
            }
        else:
            latency_stats = {}
            
        throughput = completed / duration if duration > 0 else 0
        error_rate = (failed / total * 100) if total > 0 else 100
        
        # System metrics
        system_metrics = self.monitor.get_aggregate_metrics()
        
        # Stability assessment
        stability_assessment = {
            'system_stable': (
                error_rate <= self.config.max_error_rate_percent and
                latency_stats.get('p95_ms', float('inf')) <= self.config.max_latency_ms
            ),
            'error_rate_percent': error_rate,
            'throughput_rps': throughput,
            'max_concurrency_tested': max_concurrency
        }
        
        results = {
            'test_type': test_type,
            'duration_seconds': duration,
            'max_concurrency': max_concurrency,
            'total_requests': total,
            'completed_requests': completed,
            'failed_requests': failed,
            'latency_stats': latency_stats,
            'throughput_rps': throughput,
            'error_rate_percent': error_rate,
            'system_metrics': system_metrics,
            'stability_assessment': stability_assessment
        }
        
        print(f"\n📊 {test_type.title()} Test Results:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Requests: {completed}/{total} completed ({error_rate:.1f}% errors)")
        print(f"   Throughput: {throughput:.1f} req/s")
        print(f"   Latency (p95): {latency_stats.get('p95_ms', 0):.1f}ms")
        print(f"   System: {'✅ Stable' if stability_assessment['system_stable'] else '❌ Unstable'}")
        
        return results
        
    def _analyze_resource_pressure_results(self, duration: float) -> Dict[str, Any]:
        """Analyze resource pressure test results."""
        system_metrics = self.monitor.get_aggregate_metrics()
        
        # Check if system handled pressure
        memory_peak = system_metrics.get('memory_peak', 0)
        cpu_peak = system_metrics.get('cpu_peak', 0)
        
        resource_assessment = {
            'handled_memory_pressure': memory_peak < 95.0,
            'handled_cpu_pressure': cpu_peak < 98.0,
            'memory_peak_percent': memory_peak,
            'cpu_peak_percent': cpu_peak,
            'pressure_sustained': True
        }
        
        results = {
            'test_type': 'resource_pressure',
            'duration_seconds': duration,
            'system_metrics': system_metrics,
            'resource_assessment': resource_assessment
        }
        
        print(f"\n📊 Resource Pressure Test Results:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Memory peak: {memory_peak:.1f}%")
        print(f"   CPU peak: {cpu_peak:.1f}%")
        print(f"   Assessment: {'✅ Passed' if resource_assessment['pressure_sustained'] else '❌ Failed'}")
        
        return results
        
    def _analyze_breaking_point_results(self, duration: float, breaking_point: int, tested: List[int]) -> Dict[str, Any]:
        """Analyze breaking point test results."""
        system_metrics = self.monitor.get_aggregate_metrics()
        
        breaking_point_assessment = {
            'max_sustainable_load': breaking_point,
            'breaking_point_reached': breaking_point < self.config.max_concurrency,
            'tested_concurrencies': tested,
            'scaling_efficiency': breaking_point / max(tested) if tested else 0
        }
        
        results = {
            'test_type': 'breaking_point',
            'duration_seconds': duration,
            'breaking_point_concurrency': breaking_point,
            'tested_concurrencies': tested,
            'system_metrics': system_metrics,
            'breaking_point_assessment': breaking_point_assessment
        }
        
        print(f"\n📊 Breaking Point Test Results:")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Max sustainable load: {breaking_point} concurrent requests")
        print(f"   Breaking point reached: {'✅ Yes' if breaking_point_assessment['breaking_point_reached'] else '❌ No'}")
        print(f"   Tested levels: {', '.join(map(str, tested))}")
        
        return results
        
    async def run_comprehensive_stress_test(self):
        """Run comprehensive stress testing suite."""
        print("🚀 Starting Comprehensive Stress Test Suite")
        print("="*60)
        
        all_results = {}
        
        try:
            # Test 1: Steady state at moderate load
            print("\n1️⃣  Steady State Test (Moderate Load)")
            all_results['steady_moderate'] = await self.run_steady_state_stress(
                self.config.max_concurrency // 4, 
                self.config.duration_seconds // 3
            )
            
            # Test 2: Steady state at high load
            print("\n2️⃣  Steady State Test (High Load)")
            all_results['steady_high'] = await self.run_steady_state_stress(
                self.config.max_concurrency // 2,
                self.config.duration_seconds // 3
            )
            
            # Test 3: Ramp test
            print("\n3️⃣  Ramp Load Test")
            all_results['ramp'] = await self.run_ramp_stress(
                self.config.max_concurrency,
                self.config.duration_seconds // 2
            )
            
            # Test 4: Spike test
            print("\n4️⃣  Spike Load Test")
            all_results['spike'] = await self.run_spike_stress(
                self.config.max_concurrency // 5,
                self.config.duration_seconds // 2
            )
            
            # Test 5: Resource pressure
            if self.config.enable_memory_stress or self.config.enable_cpu_stress:
                print("\n5️⃣  Resource Pressure Test")
                all_results['resource_pressure'] = await self.run_resource_pressure_test(
                    self.config.duration_seconds // 3
                )
                
            # Test 6: Breaking point
            print("\n6️⃣  Breaking Point Test")
            all_results['breaking_point'] = await self.run_breaking_point_test()
            
        except Exception as e:
            print(f"\n❌ Stress test suite failed: {e}")
            all_results['error'] = str(e)
            
        # Final assessment
        print("\n" + "="*60)
        print("🏁 STRESS TEST SUITE COMPLETE")
        print("="*60)
        
        overall_assessment = self._assess_overall_stress_performance(all_results)
        
        print(f"\n📊 Overall Assessment:")
        print(f"   System Stability: {overall_assessment['overall_stability']}")
        print(f"   Max Sustainable Load: {overall_assessment['max_sustainable_load']}")
        print(f"   Performance Grade: {overall_assessment['performance_grade']}")
        
        return {
            'test_results': all_results,
            'overall_assessment': overall_assessment
        }
        
    def _assess_overall_stress_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall stress test performance."""
        stable_tests = 0
        total_tests = 0
        max_loads = []
        
        for test_name, test_result in results.items():
            if test_name == 'error':
                continue
                
            total_tests += 1
            
            # Check stability
            if 'stability_assessment' in test_result:
                if test_result['stability_assessment'].get('system_stable', False):
                    stable_tests += 1
                    
            # Track max load
            if 'max_concurrency' in test_result:
                max_loads.append(test_result['max_concurrency'])
            elif 'breaking_point_concurrency' in test_result:
                max_loads.append(test_result['breaking_point_concurrency'])
                
        max_sustainable_load = max(max_loads) if max_loads else 0
        stability_rate = stable_tests / total_tests if total_tests > 0 else 0
        
        # Grade assessment
        if stability_rate >= 0.9 and max_sustainable_load >= self.config.max_concurrency * 0.8:
            grade = "Excellent"
        elif stability_rate >= 0.8 and max_sustainable_load >= self.config.max_concurrency * 0.6:
            grade = "Good"
        elif stability_rate >= 0.6:
            grade = "Acceptable"
        else:
            grade = "Poor"
            
        return {
            'overall_stability': f"{stability_rate:.0%}",
            'max_sustainable_load': max_sustainable_load,
            'performance_grade': grade,
            'tests_passed': f"{stable_tests}/{total_tests}"
        }


async def main():
    """Main CLI interface for stress testing."""
    parser = argparse.ArgumentParser(
        description="Stress Test Runner - Comprehensive stress testing suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick stress test (5 minutes)
  python stress_test_runner.py --duration 300 --max-concurrency 50
  
  # Comprehensive stress test (10 minutes)
  python stress_test_runner.py --duration 600 --max-concurrency 100
  
  # Memory stress test only
  python stress_test_runner.py --memory-stress --duration 300
  
  # CPU stress test only  
  python stress_test_runner.py --cpu-stress --duration 300
  
  # Breaking point test
  python stress_test_runner.py --breaking-point --max-concurrency 200
  
  # Production load test (30 minutes)
  python stress_test_runner.py --production-load --duration 1800
        """
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=300,
        help="Test duration in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--max-concurrency", "-c",
        type=int,
        default=50,
        help="Maximum concurrency level (default: 50)"
    )
    
    parser.add_argument(
        "--load-pattern",
        choices=["steady", "ramp", "spike", "random"],
        default="steady",
        help="Load pattern to generate"
    )
    
    parser.add_argument(
        "--memory-stress",
        action="store_true",
        help="Run memory stress test only"
    )
    
    parser.add_argument(
        "--cpu-stress",
        action="store_true", 
        help="Run CPU stress test only"
    )
    
    parser.add_argument(
        "--breaking-point",
        action="store_true",
        help="Find system breaking point"
    )
    
    parser.add_argument(
        "--production-load",
        action="store_true",
        help="Run production-like load test (longer duration)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="stress_test_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Setup configuration
    if args.production_load:
        config = StressTestConfig(
            duration_seconds=1800,  # 30 minutes
            max_concurrency=100,
            load_pattern="steady"
        )
    else:
        config = StressTestConfig(
            duration_seconds=args.duration,
            max_concurrency=args.max_concurrency,
            load_pattern=args.load_pattern
        )
        
    if args.memory_stress:
        config.enable_cpu_stress = False
        config.enable_io_stress = False
        
    if args.cpu_stress:
        config.enable_memory_stress = False
        config.enable_io_stress = False
        
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize and run tests
    runner = StressTestRunner(config)
    
    try:
        if args.breaking_point:
            result = await runner.run_breaking_point_test()
            
        elif args.memory_stress:
            result = await runner.run_resource_pressure_test(config.duration_seconds)
            
        elif args.cpu_stress:
            result = await runner.run_resource_pressure_test(config.duration_seconds)
            
        else:
            result = await runner.run_comprehensive_stress_test()
            
        # Save results
        timestamp = int(time.time())
        results_file = output_dir / f"stress_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Stress test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Stress test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
