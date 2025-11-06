#!/usr/bin/env python3
"""
SSM Performance Optimization Utilities

Performance optimization tools for the SSM/Linear Attention system
including benchmarking, memory optimization, and hardware-specific tuning.

Author: mini-biai-1 Team
License: MIT
"""

import time
import logging
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.profiler
import psutil

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.language.ssm_backbone import SSMBackbone, SSMConfig, SSMType, HardwareType
from src.language.linear_attention import LinearAttentionConfig, LinearAttentionType
from src.language.hybrid_processor import HybridProcessorConfig, HybridProcessingMode


@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    name: str
    forward_time: float
    throughput: float
    memory_mb: float
    flops: float
    latency_ms: float
    stability_score: float  # 0-1, measures consistency


class SSMPerformanceOptimizer:
    """Performance optimizer for SSM system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.device = self._setup_device()
        self.benchmark_history = []
        
    def _setup_logging(self):
        """Setup logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _setup_device(self):
        """Setup optimal device"""
        if torch.cuda.is_available():
            self.logger.info("CUDA available, using GPU acceleration")
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.logger.info("MPS available, using Apple Silicon GPU")
            return torch.device("mps")
        else:
            self.logger.info("Using CPU")
            return torch.device("cpu")
    
    def benchmark_ssm_configuration(self, 
                                  config: SSMConfig, 
                                  test_cases: List[Tuple[int, int]],
                                  num_runs: int = 5) -> BenchmarkResult:
        """Benchmark a specific SSM configuration"""
        
        # Create model
        model = SSMBackbone(config)
        model.to(self.device)
        model.eval()
        
        results = []
        
        for batch_size, seq_len in test_cases:
            # Generate test data
            x = torch.randn(batch_size, seq_len, config.hidden_size, device=self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(x)
            
            # Benchmark
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(x)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed = (end_time - start_time) / num_runs
            
            # Memory usage
            if self.device.type == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                torch.cuda.reset_peak_memory_stats()
            else:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024**2)
            
            # Calculate metrics
            tokens_per_sec = (batch_size * seq_len) / elapsed
            latency_ms = elapsed * 1000
            
            # FLOPS estimation (approximate)
            estimated_flops = self._estimate_flops(config, batch_size, seq_len)
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'time': elapsed,
                'throughput': tokens_per_sec,
                'memory_mb': memory_mb,
                'latency_ms': latency_ms,
                'flops': estimated_flops
            })
        
        # Calculate aggregate metrics
        avg_time = np.mean([r['time'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_memory = np.mean([r['memory_mb'] for r in results])
        avg_latency = np.mean([r['latency_ms'] for r in results])
        
        # Stability score (inverse of coefficient of variation)
        times = [r['time'] for r in results]
        stability = 1.0 / (1.0 + np.std(times) / np.mean(times)) if np.mean(times) > 0 else 0.0
        
        total_flops = np.mean([r['flops'] for r in results])
        
        return BenchmarkResult(
            name=f"SSM_{config.hidden_size}_{config.state_size}",
            forward_time=avg_time,
            throughput=avg_throughput,
            memory_mb=avg_memory,
            flops=total_flops,
            latency_ms=avg_latency,
            stability_score=stability
        )
    
    def benchmark_linear_attention_configuration(
                                               config: LinearAttentionConfig,
                                               test_cases: List[Tuple[int, int]],
                                               num_runs: int = 5) -> BenchmarkResult:
        """Benchmark a specific linear attention configuration"""
        
        # Create model
        from src.language.linear_attention import MultiHeadLinearAttention
        model = MultiHeadLinearAttention(config)
        model.to(self.device)
        model.eval()
        
        results = []
        
        for batch_size, seq_len in test_cases:
            # Generate Q, K, V
            query = torch.randn(batch_size, seq_len, config.hidden_size, device=self.device)
            key = torch.randn(batch_size, seq_len, config.hidden_size, device=self.device)
            value = torch.randn(batch_size, seq_len, config.hidden_size, device=self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(query, key, value)
            
            # Benchmark
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(query, key, value)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed = (end_time - start_time) / num_runs
            
            # Memory usage
            if self.device.type == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                torch.cuda.reset_peak_memory_stats()
            else:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024**2)
            
            # Calculate metrics
            tokens_per_sec = (batch_size * seq_len) / elapsed
            latency_ms = elapsed * 1000
            estimated_flops = self._estimate_attention_flops(config, batch_size, seq_len)
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'time': elapsed,
                'throughput': tokens_per_sec,
                'memory_mb': memory_mb,
                'latency_ms': latency_ms,
                'flops': estimated_flops
            })
        
        # Calculate aggregate metrics
        avg_time = np.mean([r['time'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_memory = np.mean([r['memory_mb'] for r in results])
        avg_latency = np.mean([r['latency_ms'] for r in results])
        
        # Stability score
        times = [r['time'] for r in results]
        stability = 1.0 / (1.0 + np.std(times) / np.mean(times)) if np.mean(times) > 0 else 0.0
        
        total_flops = np.mean([r['flops'] for r in results])
        
        return BenchmarkResult(
            name=f"LinearAttn_{config.hidden_size}_{config.num_attention_heads}",
            forward_time=avg_time,
            throughput=avg_throughput,
            memory_mb=avg_memory,
            flops=total_flops,
            latency_ms=avg_latency,
            stability_score=stability
        )
    
    def benchmark_hybrid_configuration(
                                     config: HybridProcessorConfig,
                                     test_cases: List[Tuple[int, int]],
                                     num_runs: int = 5) -> BenchmarkResult:
        """Benchmark a specific hybrid processor configuration"""
        
        # Create model
        from src.language.hybrid_processor import HybridProcessor
        model = HybridProcessor(config)
        model.to(self.device)
        model.eval()
        
        results = []
        
        for batch_size, seq_len in test_cases:
            # Generate test data
            x = torch.randn(batch_size, seq_len, config.hidden_size, device=self.device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(x)
            
            # Benchmark
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    output = model(x)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            elapsed = (end_time - start_time) / num_runs
            
            # Memory usage
            if self.device.type == 'cuda':
                memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                torch.cuda.reset_peak_memory_stats()
            else:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024**2)
            
            # Calculate metrics
            tokens_per_sec = (batch_size * seq_len) / elapsed
            latency_ms = elapsed * 1000
            estimated_flops = self._estimate_hybrid_flops(config, batch_size, seq_len)
            
            results.append({
                'batch_size': batch_size,
                'seq_len': seq_len,
                'time': elapsed,
                'throughput': tokens_per_sec,
                'memory_mb': memory_mb,
                'latency_ms': latency_ms,
                'flops': estimated_flops
            })
        
        # Calculate aggregate metrics
        avg_time = np.mean([r['time'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_memory = np.mean([r['memory_mb'] for r in results])
        avg_latency = np.mean([r['latency_ms'] for r in results])
        
        # Stability score
        times = [r['time'] for r in results]
        stability = 1.0 / (1.0 + np.std(times) / np.mean(times)) if np.mean(times) > 0 else 0.0
        
        total_flops = np.mean([r['flops'] for r in results])
        
        return BenchmarkResult(
            name=f"Hybrid_{config.hidden_size}_{config.processing_mode.value}",
            forward_time=avg_time,
            throughput=avg_throughput,
            memory_mb=avg_memory,
            flops=total_flops,
            latency_ms=avg_latency,
            stability_score=stability
        )
    
    def optimize_configuration(self, 
                             model_type: str,
                             target_workload: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for a target workload"""
        
        self.logger.info(f"Optimizing {model_type} configuration")
        
        best_config = None
        best_score = 0
        optimization_results = []
        
        # Parameter ranges to test
        if model_type == "ssm":
            hidden_sizes = [128, 256, 512, 1024]
            state_sizes = [64, 128, 256, 512]
            num_layers_list = [2, 4, 6, 8]
            
            for hidden_size in hidden_sizes:
                for state_size in state_sizes:
                    for num_layers in num_layers_list:
                        if state_size > hidden_size:
                            continue
                        
                        config = SSMConfig(
                            hidden_size=hidden_size,
                            state_size=state_size,
                            num_layers=num_layers,
                            hardware_type=HardwareType.CUDA if self.device.type == 'cuda' else HardwareType.CPU
                        )
                        
                        # Benchmark
                        test_cases = [(1, 64), (2, 128), (1, 256)]
                        result = self.benchmark_ssm_configuration(config, test_cases)
                        
                        # Score based on workload requirements
                        score = self._score_configuration(result, target_workload)
                        
                        optimization_results.append({
                            'config': config,
                            'result': result,
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_config = config
        
        elif model_type == "linear_attention":
            hidden_sizes = [256, 512, 1024]
            num_heads_list = [4, 8, 16, 32]
            attention_types = [
                LinearAttentionType.PERFORMER,
                LinearAttentionType.LINEAR_TRANSFORMER,
                LinearAttentionType.SLIDING_WINDOW
            ]
            
            for hidden_size in hidden_sizes:
                for num_heads in num_heads_list:
                    if hidden_size % num_heads != 0:
                        continue
                    
                    for attention_type in attention_types:
                        config = LinearAttentionConfig(
                            hidden_size=hidden_size,
                            num_attention_heads=num_heads,
                            attention_type=attention_type,
                            hardware_type=HardwareType.CUDA if self.device.type == 'cuda' else HardwareType.CPU
                        )
                        
                        # Benchmark
                        test_cases = [(1, 64), (2, 128), (1, 256)]
                        result = self.benchmark_linear_attention_configuration(config, test_cases)
                        
                        # Score
                        score = self._score_configuration(result, target_workload)
                        
                        optimization_results.append({
                            'config': config,
                            'result': result,
                            'score': score
                        })
                        
                        if score > best_score:
                            best_score = score
                            best_config = config
        
        elif model_type == "hybrid":
            hidden_sizes = [256, 512, 1024]
            processing_modes = [
                HybridProcessingMode.SEQUENTIAL,
                HybridProcessingMode.PARALLEL,
                HybridProcessingMode.WEIGHTED
            ]
            
            for hidden_size in hidden_sizes:
                for processing_mode in processing_modes:
                    config = HybridProcessorConfig(
                        hidden_size=hidden_size,
                        processing_mode=processing_mode,
                        hardware_type=HardwareType.CUDA if self.device.type == 'cuda' else HardwareType.CPU
                    )
                    
                    # Benchmark
                    test_cases = [(1, 64), (2, 128), (1, 256)]
                    result = self.benchmark_hybrid_configuration(config, test_cases)
                    
                    # Score
                    score = self._score_configuration(result, target_workload)
                    
                    optimization_results.append({
                        'config': config,
                        'result': result,
                        'score': score
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_config = config
        
        return {
            'best_config': best_config,
            'best_score': best_score,
            'all_results': optimization_results
        }
    
    def _estimate_flops(self, config: SSMConfig, batch_size: int, seq_len: int) -> float:
        """Estimate FLOPs for SSM"""
        # Approximate FLOPs calculation
        hidden_size = config.hidden_size
        state_size = config.state_size
        
        # SSM matrix operations
        flops = 0
        flops += batch_size * seq_len * hidden_size * state_size * 2  # A*x + B*u
        flops += batch_size * seq_len * hidden_size * state_size * 2  # C*x + D*u
        flops *= config.num_layers
        
        return flops
    
    def _estimate_attention_flops(self, config: LinearAttentionConfig, batch_size: int, seq_len: int) -> float:
        """Estimate FLOPs for linear attention"""
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        num_heads = config.num_attention_heads
        
        # Linear attention FLOPs (O(N) complexity)
        flops = 0
        flops += batch_size * seq_len * hidden_size * 2  # Q, K, V projections
        flops += batch_size * seq_len * head_dim * num_heads * 2  # Attention computation
        
        return flops
    
    def _estimate_hybrid_flops(self, config: HybridProcessorConfig, batch_size: int, seq_len: int) -> float:
        """Estimate FLOPs for hybrid processor"""
        hidden_size = config.hidden_size
        
        # Combine SSM and attention FLOPs
        ssm_flops = self._estimate_flops(config.ssm_config or SSMConfig(hidden_size=hidden_size), batch_size, seq_len)
        attention_flops = self._estimate_attention_flops(config.attention_config or LinearAttentionConfig(hidden_size=hidden_size), batch_size, seq_len)
        
        return ssm_flops + attention_flops
    
    def _score_configuration(self, result: BenchmarkResult, workload: Dict[str, Any]) -> float:
        """Score configuration based on workload requirements"""
        
        score = 0.0
        weights = workload.get('weights', {})
        
        # Performance metrics
        if 'throughput_weight' in weights:
            score += weights['throughput_weight'] * result.throughput
        
        if 'latency_weight' in weights:
            score += weights['latency_weight'] * (1.0 / max(result.latency_ms, 0.001))
        
        if 'memory_weight' in weights:
            max_memory = workload.get('max_memory_mb', 1000)
            memory_score = max(0, 1.0 - result.memory_mb / max_memory)
            score += weights['memory_weight'] * memory_score
        
        if 'stability_weight' in weights:
            score += weights['stability_weight'] * result.stability_score
        
        # FLOPs efficiency
        if 'flops_efficiency_weight' in weights:
            flops_per_sec = result.flops / max(result.forward_time, 0.001)
            score += weights['flops_efficiency_weight'] * flops_per_sec / 1e9  # Normalize to billions
        
        return score
    
    def generate_optimization_report(self, optimization_results: Dict[str, Any]) -> str:
        """Generate optimization report"""
        
        report = []
        report.append("SSM Performance Optimization Report")
        report.append("=" * 50)
        
        # Hardware info
        report.append(f"Device: {self.device}")
        report.append(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            report.append(f"CUDA Device: {torch.cuda.get_device_name()}")
            report.append(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimization results
        for model_type, results in optimization_results.items():
            report.append(f"\n{model_type.upper()} Optimization:")
            report.append("-" * 30)
            
            best_result = max(results['all_results'], key=lambda x: x['score'])
            
            report.append(f"Best Score: {best_result['score']:.2f}")
            report.append(f"Best Config: {best_result['config']}")
            report.append(f"Best Performance:")
            report.append(f"  Throughput: {best_result['result'].throughput:.1f} tokens/sec")
            report.append(f"  Latency: {best_result['result'].latency_ms:.2f} ms")
            report.append(f"  Memory: {best_result['result'].memory_mb:.1f} MB")
            report.append(f"  Stability: {best_result['result'].stability_score:.2f}")
        
        return "\n".join(report)
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark suite"""
        
        self.logger.info("Running comprehensive benchmark suite")
        
        # Standard test configurations
        test_cases = [(1, 32), (1, 64), (2, 128), (1, 256)]
        num_runs = 5
        
        results = {}
        
        # Benchmark SSM
        self.logger.info("Benchmarking SSM configurations...")
        ssm_configs = [
            SSMConfig(hidden_size=128, state_size=64),
            SSMConfig(hidden_size=256, state_size=128),
            SSMConfig(hidden_size=512, state_size=256),
        ]
        
        ssm_results = []
        for config in ssm_configs:
            result = self.benchmark_ssm_configuration(config, test_cases, num_runs)
            ssm_results.append(result)
        
        results['ssm'] = ssm_results
        
        # Benchmark Linear Attention
        self.logger.info("Benchmarking Linear Attention configurations...")
        linear_configs = [
            LinearAttentionConfig(hidden_size=128, num_attention_heads=4),
            LinearAttentionConfig(hidden_size=256, num_attention_heads=8),
            LinearAttentionConfig(hidden_size=512, num_attention_heads=16),
        ]
        
        linear_results = []
        for config in linear_configs:
            result = self.benchmark_linear_attention_configuration(config, test_cases, num_runs)
            linear_results.append(result)
        
        results['linear_attention'] = linear_results
        
        # Benchmark Hybrid
        self.logger.info("Benchmarking Hybrid configurations...")
        hybrid_configs = [
            HybridProcessorConfig(hidden_size=128),
            HybridProcessorConfig(hidden_size=256),
            HybridProcessorConfig(hidden_size=512),
        ]
        
        hybrid_results = []
        for config in hybrid_configs:
            result = self.benchmark_hybrid_configuration(config, test_cases, num_runs)
            hybrid_results.append(result)
        
        results['hybrid'] = hybrid_results
        
        return results
    
    def print_benchmark_results(self, results: Dict[str, Any]):
        """Print benchmark results in a formatted table"""
        
        print("\n" + "=" * 80)
        print("SSM/Linear Attention Benchmark Results")
        print("=" * 80)
        
        for model_type, model_results in results.items():
            print(f"\n{model_type.upper()} RESULTS:")
            print("-" * 40)
            
            for result in model_results:
                print(f"Configuration: {result.name}")
                print(f"  Forward Time: {result.forward_time:.4f}s")
                print(f"  Throughput: {result.throughput:.1f} tokens/sec")
                print(f"  Memory: {result.memory_mb:.1f} MB")
                print(f"  Latency: {result.latency_ms:.2f} ms")
                print(f"  Stability: {result.stability_score:.2f}")
                print(f"  FLOPs: {result.flops:.1f}")
                print()


def main():
    """Main optimization function"""
    optimizer = SSMPerformanceOptimizer()
    
    # Run comprehensive benchmark
    results = optimizer.run_comprehensive_benchmark()
    
    # Print results
    optimizer.print_benchmark_results(results)
    
    # Example workload optimization
    target_workload = {
        'weights': {
            'throughput_weight': 0.4,
            'latency_weight': 0.3,
            'memory_weight': 0.2,
            'stability_weight': 0.1
        },
        'max_memory_mb': 200
    }
    
    print("\nOptimizing for target workload...")
    optimization_results = {}
    
    for model_type in ['ssm', 'linear_attention', 'hybrid']:
        opt_result = optimizer.optimize_configuration(model_type, target_workload)
        optimization_results[model_type] = opt_result
    
    # Generate and print report
    report = optimizer.generate_optimization_report(optimization_results)
    print("\n" + report)
    
    print("\nBenchmark and optimization completed!")


if __name__ == "__main__":
    main()