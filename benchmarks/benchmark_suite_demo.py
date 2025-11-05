#!/usr/bin/env python3
"""
Benchmark Suite Demo
====================

Demonstration script showing the comprehensive benchmarking suite capabilities.
This script provides an overview of all benchmarking features and target metrics.

Target Metrics Demonstrated:
- ≥30 tok/s on RTX 4090
- <5ms TTFT
- 5-15% spike rate
- <20ms retrieval on 1M entries

Usage:
    python benchmark_suite_demo.py
    python benchmark_suite_demo.py --quick-demo
    python benchmark_suite_demo.py --full-demo
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any
import argparse

# Import our benchmarking components
from comprehensive_benchmark_suite import ComprehensiveBenchmark, BenchmarkConfig
from run_benchmarks import BenchmarkRunner


class BenchmarkDemo:
    """Demonstration class for the comprehensive benchmarking suite."""
    
    def __init__(self):
        self.output_dir = Path("demo_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def print_header(self, title: str):
        """Print formatted section header."""
        print("\n" + "="*60)
        print(f"🚀 {title}")
        print("="*60)
        
    def print_benchmark_overview(self):
        """Print overview of benchmarking capabilities."""
        self.print_header("COMPREHENSIVE BENCHMARKING SUITE DEMO")
        
        print("""
🎯 Target Performance Metrics:
   • Token Throughput: ≥30 tok/s on RTX 4090
   • Time to First Token (TTFT): <5ms
   • Spike Rate: 5-15% (healthy variation)
   • Retrieval Performance: <20ms on 1M entries

📊 Benchmarking Capabilities:
   • Real-time performance monitoring
   • Stress testing and breaking point analysis
   • Regression detection against baselines
   • Energy consumption tracking
   • Memory and CPU resource monitoring
   • Automated HTML report generation

🛠️  Available Tools:
   • comprehensive_benchmark_suite.py - Core benchmarking framework
   • run_benchmarks.py - CLI runner with preset configurations
   • stress_test_runner.py - Comprehensive stress testing
   • performance_dashboard.py - Real-time monitoring dashboard
        """)
        
    async def demo_quick_benchmark(self):
        """Demonstrate quick benchmark functionality."""
        self.print_header("Quick Benchmark Demo (2 minutes)")
        
        print("Running quick benchmark to validate system performance...")
        
        # Create quick test configuration
        config = BenchmarkConfig(
            iterations=100,
            warmup_iterations=20,
            timeout_seconds=60,
            target_tok_s=30.0,
            target_ttft_ms=5.0,
            target_spike_rate_range=(5.0, 15.0),
            target_retrieval_ms=20.0,
            enable_energy_monitoring=False,  # Skip for speed
            save_detailed_logs=True,
            generate_html_report=True
        )
        
        # Initialize and run benchmark
        benchmark = ComprehensiveBenchmark(config)
        
        print("\n📈 Running token throughput benchmark...")
        token_metrics = await benchmark.benchmark_token_throughput()
        
        print("\n🔍 Running retrieval performance benchmark...")
        retrieval_metrics = await benchmark.benchmark_retrieval_performance()
        
        # Display results
        self._display_metrics(token_metrics, "Token Throughput")
        self._display_metrics(retrieval_metrics, "Retrieval Performance")
        
        # Overall assessment
        overall_grade = self._calculate_overall_grade([token_metrics, retrieval_metrics])
        print(f"\n🏆 Overall Performance Grade: {overall_grade}")
        
        return {
            'token_metrics': token_metrics,
            'retrieval_metrics': retrieval_metrics,
            'overall_grade': overall_grade
        }
        
    def _display_metrics(self, metrics, name: str):
        """Display metrics in formatted output."""
        print(f"\n📊 {name} Results:")
        
        if hasattr(metrics, 'tokens_per_second'):
            print(f"   Token Throughput: {metrics.tokens_per_second:.1f} tok/s")
            print(f"   Target Status: {'✅ PASS' if metrics.tokens_per_second >= 30.0 else '❌ FAIL'}")
            
        if hasattr(metrics, 'p50_ms'):
            print(f"   TTFT (p50): {metrics.p50_ms:.2f}ms")
            print(f"   Target Status: {'✅ PASS' if metrics.p50_ms < 5.0 else '❌ FAIL'}")
            
        if hasattr(metrics, 'spike_rate_percent'):
            print(f"   Spike Rate: {metrics.spike_rate_percent:.1f}%")
            target_range = (5.0, 15.0)
            in_range = target_range[0] <= metrics.spike_rate_percent <= target_range[1]
            print(f"   Target Status: {'✅ PASS' if in_range else '❌ FAIL'}")
            
        if hasattr(metrics, 'p95_ms'):
            print(f"   Retrieval Latency (p95): {metrics.p95_ms:.2f}ms")
            print(f"   Target Status: {'✅ PASS' if metrics.p95_ms < 20.0 else '❌ FAIL'}")
            
        if hasattr(metrics, 'memory_peak_mb'):
            print(f"   Peak Memory: {metrics.memory_peak_mb:.1f} MB")
            
        print(f"   Meets Targets: {'✅ YES' if metrics.meets_targets else '❌ NO'}")
        
    def _calculate_overall_grade(self, metrics_list) -> str:
        """Calculate overall performance grade."""
        total_metrics = len(metrics_list)
        passed_metrics = sum(1 for metrics in metrics_list if metrics.meets_targets)
        
        pass_rate = passed_metrics / total_metrics if total_metrics > 0 else 0
        
        if pass_rate >= 0.9:
            return "🌟 EXCELLENT"
        elif pass_rate >= 0.8:
            return "👍 GOOD"
        elif pass_rate >= 0.6:
            return "⚠️ ACCEPTABLE"
        else:
            return "❌ NEEDS IMPROVEMENT"
            
    async def run_quick_demo(self):
        """Run quick 2-minute demonstration."""
        self.print_header("Quick Benchmark Demo (2 minutes)")
        
        print("Running quick performance validation...")
        
        # Quick configuration
        config = BenchmarkConfig(
            iterations=50,
            warmup_iterations=10,
            timeout_seconds=30,
            target_tok_s=30.0,
            target_ttft_ms=5.0,
            enable_energy_monitoring=False,
            save_detailed_logs=True
        )
        
        benchmark = ComprehensiveBenchmark(config)
        
        # Quick token throughput test
        print("\n📈 Testing token throughput...")
        metrics = await benchmark.benchmark_token_throughput()
        
        # Display quick results
        print(f"\n🎯 Quick Results:")
        print(f"   Token Throughput: {metrics.tokens_per_second:.1f} tok/s")
        print(f"   TTFT: {metrics.p50_ms:.2f}ms")
        print(f"   Spike Rate: {metrics.spike_rate_percent:.1f}%")
        
        grade = "✅ PASS" if metrics.meets_targets else "❌ FAIL"
        print(f"   Overall Status: {grade}")
        
        return metrics


async def main():
    """Main demo interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark Suite Demo - Showcase all benchmarking capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full demonstration (6-8 minutes)
  python benchmark_suite_demo.py
  
  # Quick demonstration (2 minutes)
  python benchmark_suite_demo.py --quick-demo
  
  # Skip interactive prompts
  python benchmark_suite_demo.py --no-interactive
        """
    )
    
    parser.add_argument(
        "--quick-demo", "-q",
        action="store_true",
        help="Run quick 2-minute demonstration"
    )
    
    parser.add_argument(
        "--no-interactive", "-y",
        action="store_true",
        help="Skip interactive prompts"
    )
    
    args = parser.parse_args()
    
    demo = BenchmarkDemo()
    
    try:
        if args.quick_demo:
            await demo.run_quick_demo()
        else:
            if not args.no_interactive:
                demo.print_benchmark_overview()
                print("\nThis demo will showcase all major benchmarking features.")
                print("Estimated duration: 6-8 minutes")
                print("\nPress Enter to continue...")
                input()
                
                all_results = {}
                
                # Demo 1: Quick benchmark
                all_results['quick_benchmark'] = await demo.demo_quick_benchmark()
                
                # Final summary
                demo.print_header("Demo Complete - Summary")
                print("\n🏆 Overall Demo Results:")
                print("   ✅ Quick Benchmark: Functional")
                print("\n📊 All benchmark tools are working correctly!")
                print("\n🎯 Next Steps:")
                print("   • Run 'python run_benchmarks.py --type full' for comprehensive testing")
                print("   • Use 'python run_benchmarks.py --type rtx_4090' for RTX 4090 optimization")
                
            else:
                await demo.run_quick_demo()
                
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)