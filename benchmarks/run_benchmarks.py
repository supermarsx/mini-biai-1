#!/usr/bin/env python3
"""
Benchmark Runner
================

Quick runner script for different benchmark types with preset configurations.
Target metrics: ≥30 tok/s on RTX 4090, <5ms TTFT, 5-15% spike rate, <20ms retrieval on 1M entries.

Usage:
    python run_benchmarks.py --type quick
    python run_benchmarks.py --type full
    python run_benchmarks.py --type stress --duration 600
    python run_benchmarks.py --type tok_throughput
    python run_benchmarks.py --type regression --baseline results/baseline.json
"""

import asyncio
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

# Import our comprehensive benchmark suite
from comprehensive_benchmark_suite import (
    ComprehensiveBenchmark, 
    BenchmarkConfig, 
    PerformanceMetrics
)


class BenchmarkRunner:
    """Runner for different benchmark types with preset configurations."""
    
    def __init__(self):
        self.output_dir = Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def get_presets(self) -> Dict[str, BenchmarkConfig]:
        """Get preset configurations for different benchmark types."""
        return {
            "quick": self._quick_config(),
            "tok_throughput": self._tok_throughput_config(),
            "retrieval": self._retrieval_config(),
            "full": self._full_config(),
            "stress": self._stress_config(),
            "regression": self._regression_config(),
            "rtx_4090": self._rtx_4090_config(),
            "production": self._production_config()
        }
        
    def _quick_config(self) -> BenchmarkConfig:
        """Quick benchmark configuration (5 minutes)."""
        return BenchmarkConfig(
            iterations=100,
            warmup_iterations=20,
            timeout_seconds=60,
            batch_sizes=[1, 4, 8],
            test_text_lengths=[10, 50, 100],
            retrieval_index_size=100000,  # 100K instead of 1M
            retrieval_queries=100,
            stress_concurrency_levels=[1, 5, 10],
            stress_duration_seconds=60,
            enable_energy_monitoring=False,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _tok_throughput_config(self) -> BenchmarkConfig:
        """Token throughput focused configuration."""
        return BenchmarkConfig(
            iterations=1000,
            warmup_iterations=100,
            timeout_seconds=120,
            batch_sizes=[1, 4, 8, 16, 32],
            test_text_lengths=[10, 50, 100, 500, 1000],
            retrieval_index_size=50000,  # Smaller index for speed
            retrieval_queries=50,
            enable_energy_monitoring=True,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _retrieval_config(self) -> BenchmarkConfig:
        """Retrieval performance focused configuration."""
        return BenchmarkConfig(
            iterations=200,
            warmup_iterations=50,
            timeout_seconds=180,
            batch_sizes=[1, 2, 4],
            test_text_lengths=[50, 100, 200],
            retrieval_index_size=1000000,  # Full 1M entries
            retrieval_queries=1000,
            stress_concurrency_levels=[1, 5, 10, 20],
            stress_duration_seconds=300,
            enable_energy_monitoring=True,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _full_config(self) -> BenchmarkConfig:
        """Full comprehensive benchmark configuration."""
        return BenchmarkConfig(
            iterations=1000,
            warmup_iterations=100,
            timeout_seconds=300,
            batch_sizes=[1, 4, 8, 16, 32],
            test_text_lengths=[10, 50, 100, 500, 1000],
            retrieval_index_size=1000000,
            retrieval_queries=1000,
            stress_concurrency_levels=[1, 5, 10, 20, 50],
            stress_duration_seconds=300,
            enable_energy_monitoring=True,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _stress_config(self) -> BenchmarkConfig:
        """Stress test focused configuration."""
        return BenchmarkConfig(
            iterations=500,
            warmup_iterations=50,
            timeout_seconds=600,
            batch_sizes=[1, 4, 8, 16],
            test_text_lengths=[10, 50, 100],
            retrieval_index_size=500000,
            retrieval_queries=500,
            stress_concurrency_levels=[1, 5, 10, 20, 50, 100],
            stress_duration_seconds=600,  # Longer stress test
            enable_energy_monitoring=True,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _regression_config(self) -> BenchmarkConfig:
        """Regression testing configuration."""
        return BenchmarkConfig(
            iterations=500,
            warmup_iterations=50,
            timeout_seconds=180,
            batch_sizes=[1, 4, 8, 16],
            test_text_lengths=[10, 50, 100, 500],
            retrieval_index_size=500000,
            retrieval_queries=500,
            stress_concurrency_levels=[1, 5, 10, 20],
            stress_duration_seconds=300,
            enable_energy_monitoring=True,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _rtx_4090_config(self) -> BenchmarkConfig:
        """RTX 4090 optimized configuration targeting ≥30 tok/s."""
        return BenchmarkConfig(
            target_tok_s=30.0,
            target_ttft_ms=5.0,
            target_spike_rate_range=(5.0, 15.0),
            target_retrieval_ms=20.0,
            iterations=1500,
            warmup_iterations=150,
            timeout_seconds=300,
            batch_sizes=[1, 4, 8, 16, 32, 64],  # Larger batch sizes for GPU
            test_text_lengths=[10, 50, 100, 500, 1000, 2000],
            retrieval_index_size=1000000,
            retrieval_queries=1500,
            stress_concurrency_levels=[1, 5, 10, 20, 50, 100],
            stress_duration_seconds=600,
            enable_energy_monitoring=True,
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    def _production_config(self) -> BenchmarkConfig:
        """Production monitoring configuration for ongoing regression testing."""
        return BenchmarkConfig(
            iterations=100,
            warmup_iterations=20,
            timeout_seconds=120,
            batch_sizes=[1, 4, 8],
            test_text_lengths=[10, 50, 100],
            retrieval_index_size=200000,  # Smaller for faster CI/CD
            retrieval_queries=200,
            stress_concurrency_levels=[1, 5, 10],
            stress_duration_seconds=180,
            enable_energy_monitoring=False,  # Skip energy for speed
            save_detailed_logs=True,
            generate_html_report=True
        )
        
    async def run_benchmark(self, benchmark_type: str, **kwargs) -> Dict[str, Any]:
        """Run benchmark with specified type."""
        presets = self.get_presets()
        
        if benchmark_type not in presets:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
            
        config = presets[benchmark_type]
        
        # Override config with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        # Create and run benchmark
        benchmark = ComprehensiveBenchmark(config)
        
        print(f"🚀 Starting {benchmark_type} benchmark...")
        print(f"📊 Target metrics:")
        print(f"   • Tok/s: ≥{config.target_tok_s}")
        print(f"   • TTFT: <{config.target_ttft_ms}ms")
        print(f"   • Spike rate: {config.target_spike_rate_range[0]}-{config.target_spike_rate_range[1]}%")
        print(f"   • Retrieval: <{config.target_retrieval_ms}ms")
        print()
        
        start_time = time.time()
        
        try:
            if benchmark_type == "tok_throughput":
                result = await benchmark.benchmark_token_throughput()
                return {
                    'type': 'tok_throughput',
                    'metrics': self._extract_key_metrics(result),
                    'passed': result.meets_targets
                }
                
            elif benchmark_type == "retrieval":
                result = await benchmark.benchmark_retrieval_performance()
                return {
                    'type': 'retrieval',
                    'metrics': self._extract_key_metrics(result),
                    'passed': result.meets_targets
                }
                
            elif benchmark_type == "stress":
                result = await benchmark.run_stress_test()
                return {
                    'type': 'stress',
                    'metrics': self._extract_stress_metrics(result),
                    'max_sustainable_load': self._find_max_sustainable_load(result)
                }
                
            elif benchmark_type in ["regression"]:
                # Set baseline file if provided
                if 'baseline_file' in kwargs:
                    config.baseline_file = kwargs['baseline_file']
                    
                result = await benchmark.run_regression_detection()
                return {
                    'type': 'regression',
                    'regression_detected': result.get('regression_detected', False),
                    'analysis': result.get('regression_analysis', {})
                }
                
            else:
                # Run full suite
                result = await benchmark.run_full_benchmark_suite()
                return {
                    'type': 'full_suite',
                    'results': result,
                    'assessment': result.get('overall_assessment', {})
                }
                
        finally:
            elapsed_time = time.time() - start_time
            print(f"\n⏱️  Benchmark completed in {elapsed_time:.1f} seconds")
            
    def _extract_key_metrics(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """Extract key metrics from PerformanceMetrics object."""
        return {
            'tokens_per_second': metrics.tokens_per_second,
            'p50_ms': metrics.p50_ms,
            'p95_ms': metrics.p95_ms,
            'spike_rate_percent': metrics.spike_rate_percent,
            'memory_peak_mb': metrics.memory_peak_mb,
            'energy_consumption_joules': metrics.energy_consumption_joules
        }
        
    def _extract_stress_metrics(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from stress test results."""
        return {
            'concurrency_levels': list(stress_results.keys()),
            'max_throughput': max(r.get('throughput_rps', 0) for r in stress_results.values()),
            'stable_concurrency_levels': [
                k for k, v in stress_results.items() 
                if v.get('system_stable', False)
            ]
        }
        
    def _find_max_sustainable_load(self, stress_results: Dict[str, Any]) -> int:
        """Find maximum sustainable load from stress results."""
        stable_levels = [
            int(k.split('_')[1]) for k in stress_results.keys()
            if stress_results[k].get('system_stable', False)
        ]
        return max(stable_levels) if stable_levels else 0
        
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of benchmark results."""
        print("\n" + "="*60)
        print("🎯 BENCHMARK RESULTS SUMMARY")
        print("="*60)
        
        if results['type'] == 'tok_throughput':
            metrics = results['metrics']
            print(f"📈 Token Throughput Performance:")
            print(f"   • Tokens/second: {metrics['tokens_per_second']:.1f}")
            print(f"   • TTFT (p50): {metrics['p50_ms']:.2f}ms")
            print(f"   • Latency (p95): {metrics['p95_ms']:.2f}ms")
            print(f"   • Spike rate: {metrics['spike_rate_percent']:.1f}%")
            print(f"   • Peak memory: {metrics['memory_peak_mb']:.1f}MB")
            
            # Check targets
            passed = results['passed']
            print(f"   • Target Status: {'✅ PASSED' if passed else '❌ FAILED'}")
            
        elif results['type'] == 'retrieval':
            metrics = results['metrics']
            print(f"🔍 Retrieval Performance:")
            print(f"   • Latency (p95): {metrics['p95_ms']:.2f}ms")
            print(f"   • Mean latency: {metrics['p50_ms']:.2f}ms")
            print(f"   • Peak memory: {metrics['memory_peak_mb']:.1f}MB")
            print(f"   • Target Status: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
            
        elif results['type'] == 'stress':
            metrics = results['metrics']
            print(f"💪 Stress Test Results:")
            print(f"   • Max throughput: {metrics['max_throughput']:.1f} req/s")
            print(f"   • Max sustainable load: {results['max_sustainable_load']} concurrent requests")
            print(f"   • Stable concurrency levels: {', '.join(map(str, metrics['stable_concurrency_levels']))}")
            
        elif results['type'] == 'regression':
            reg_detected = results['regression_detected']
            print(f"📊 Regression Analysis:")
            print(f"   • Regression detected: {'⚠️ YES' if reg_detected else '✅ NO'}")
            
            if results['analysis']:
                for metric, analysis in results['analysis'].items():
                    change = analysis.get('change_percent', 0)
                    print(f"   • {metric}: {change:+.1f}% change")
                    
        elif results['type'] == 'full_suite':
            assessment = results.get('assessment', {})
            print(f"🏆 Overall Assessment:")
            grade = assessment.get('performance_grade', 'unknown').upper()
            targets_met = assessment.get('targets_met', 0)
            total_targets = assessment.get('total_targets', 0)
            
            grade_emoji = {
                'EXCELLENT': '🌟',
                'GOOD': '👍',
                'ACCEPTABLE': '⚠️',
                'NEEDS_IMPROVEMENT': '❌'
            }.get(grade, '❓')
            
            print(f"   • Performance Grade: {grade_emoji} {grade}")
            print(f"   • Targets Met: {targets_met}/{total_targets}")
            
            # Print individual benchmarks
            benchmarks = results['results'].get('benchmarks', {})
            for name, data in benchmarks.items():
                print(f"\n   📊 {name.replace('_', ' ').title()}:")
                if 'tokens_per_second' in data:
                    print(f"      • Token throughput: {data['tokens_per_second']:.1f} tok/s")
                    print(f"      • TTFT: {data['p50_ms']:.2f}ms")
                    print(f"      • Spike rate: {data['spike_rate_percent']:.1f}%")
                if 'p95_ms' in data:
                    print(f"      • Retrieval latency: {data['p95_ms']:.2f}ms")
                    
        print("="*60)


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark Runner - Run performance benchmarks with preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick performance check (5 minutes)
  python run_benchmarks.py --type quick
  
  # Full RTX 4090 benchmark targeting ≥30 tok/s
  python run_benchmarks.py --type rtx_4090
  
  # Token throughput benchmark only
  python run_benchmarks.py --type tok_throughput
  
  # Retrieval performance benchmark
  python run_benchmarks.py --type retrieval
  
  # Stress test for 10 minutes
  python run_benchmarks.py --type stress --duration 600
  
  # Regression testing against baseline
  python run_benchmarks.py --type regression --baseline-file results/baseline.json
  
  # Production monitoring (fast CI/CD friendly)
  python run_benchmarks.py --type production
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["quick", "tok_throughput", "retrieval", "full", "stress", "regression", "rtx_4090", "production"],
        default="quick",
        help="Benchmark type to run (default: quick)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        help="Duration for stress tests in seconds"
    )
    
    parser.add_argument(
        "--baseline-file", "-b",
        type=str,
        help="Baseline file for regression testing"
    )
    
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        help="Number of iterations for benchmarks"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="benchmark_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-energy",
        action="store_true",
        help="Disable energy monitoring for faster execution"
    )
    
    args = parser.parse_args()
    
    # Setup
    runner = BenchmarkRunner()
    kwargs = {}
    
    if args.duration:
        kwargs['stress_duration_seconds'] = args.duration
    if args.baseline_file:
        kwargs['baseline_file'] = args.baseline_file
    if args.iterations:
        kwargs['iterations'] = args.iterations
    if args.no_energy:
        kwargs['enable_energy_monitoring'] = False
        
    try:
        # Run benchmark
        results = await runner.run_benchmark(args.type, **kwargs)
        
        # Print summary
        runner.print_results_summary(results)
        
        # Save results summary
        summary_file = Path(args.output_dir) / f"{args.type}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Results saved to: {summary_file}")
        
        # Exit with appropriate code
        if args.type in ['tok_throughput', 'retrieval']:
            exit_code = 0 if results['passed'] else 1
        elif args.type == 'regression':
            exit_code = 1 if results['regression_detected'] else 0
        else:
            exit_code = 0
            
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)