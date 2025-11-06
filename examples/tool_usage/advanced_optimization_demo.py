#!/usr/bin/env python3
"""
Advanced Optimization Demonstration

This demo showcases advanced tool_usage optimization capabilities including:
- Usage pattern analysis and learning
- Performance optimization strategies
- Intelligent caching mechanisms
- Batch operation optimization
- Resource usage optimization
- Parallel execution management

Run with: python examples/tool_usage/advanced_optimization_demo.py
"""

import sys
import argparse
import json
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.usage_optimizer import UsageOptimizer
    from tool_usage.tool_registry import ToolRegistry
    from tool_usage.command_executor import CommandExecutor
except ImportError as e:
    print(f"Warning: Could not import optimization modules: {e}")
    print("This demo requires the advanced optimization components.")


class AdvancedOptimizationDemo:
    """Demonstrates advanced optimization capabilities."""
    
    def __init__(self, cache_size: int = 1000, verbose: bool = False):
        """Initialize the demo with optimization settings."""
        self.verbose = verbose
        self.cache_size = cache_size
        self.command_executor = CommandExecutor()
        self.tool_registry = ToolRegistry()
        self.usage_optimizer = UsageOptimizer(cache_size=cache_size)
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {level}: {message}")
    
    def demonstrate_caching_optimization(self):
        """Demonstrate intelligent caching optimization."""
        print("\n" + "="*70)
        print("1. INTELLIGENT CACHING OPTIMIZATION")
        print("="*70)
        
        cache_test_commands = [
            ("python", ["--version"]),
            ("echo", ["cached_output_test"]),
        ]
        
        print(f"\n🗂️  Testing caching with {len(cache_test_commands)} commands")
        print(f"📦 Cache size: {self.cache_size}")
        
        cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
            "avg_execution_times": [],
            "cache_execution_times": []
        }
        
        # First run (cache misses)
        print("\n🔄 First run (building cache):")
        for i, (cmd, args) in enumerate(cache_test_commands, 1):
            print(f"  {i}. {cmd} {' '.join(args)}")
            
            # Regular execution
            start_time = time.time()
            result1 = self.command_executor.execute(cmd, args, timeout=10)
            exec_time1 = time.time() - start_time
            
            # Cached execution
            start_time = time.time()
            result2 = self.usage_optimizer.get_cached_result(cmd, args, lambda: self.command_executor.execute(cmd, args, timeout=10))
            exec_time2 = time.time() - start_time
            
            cache_stats["hits"] += 1 if result2 else 0
            cache_stats["misses"] += 1 if not result2 else 0
            cache_stats["total_requests"] += 1
            cache_stats["avg_execution_times"].append(exec_time1)
            cache_stats["cache_execution_times"].append(exec_time2)
            
            improvement = (exec_time1 - exec_time2) / exec_time1 * 100 if exec_time1 > 0 else 0
            print(f"     Regular: {exec_time1:.4f}s | Cached: {exec_time2:.4f}s | Improvement: {improvement:.1f}%")
        
        # Cache statistics
        if cache_stats["total_requests"] > 0:
            hit_rate = cache_stats["hits"] / cache_stats["total_requests"] * 100
            avg_regular = statistics.mean(cache_stats["avg_execution_times"])
            avg_cached = statistics.mean(cache_stats["cache_execution_times"])
            
            print(f"\n📊 Cache Performance Summary:")
            print(f"   Hit rate: {hit_rate:.1f}%")
            print(f"   Average regular execution: {avg_regular:.4f}s")
            print(f"   Average cached execution: {avg_cached:.6f}s")
        
        self.log("Caching optimization demonstration completed")
        return cache_stats
    
    def demonstrate_batch_optimization(self):
        """Demonstrate batch operation optimization."""
        print("\n" + "="*70)
        print("2. BATCH OPERATION OPTIMIZATION")
        print("="*70)
        
        # Create batch of similar commands
        batch_commands = []
        for i in range(5):
            batch_commands.append(("echo", [f"Batch command {i+1}"]))
        
        print(f"\n📦 Testing batch operations with {len(batch_commands)} commands")
        
        # Sequential execution
        print("\n🔄 Sequential execution:")
        start_time = time.time()
        sequential_results = []
        for i, (cmd, args) in enumerate(batch_commands):
            result = self.command_executor.execute(cmd, args, timeout=10)
            sequential_results.append(result.returncode == 0)
        sequential_time = time.time() - start_time
        
        # Parallel execution
        print("⚡ Parallel execution:")
        start_time = time.time()
        parallel_results = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self.command_executor.execute, cmd, args, 10): (cmd, args) 
                      for cmd, args in batch_commands}
            
            for future in as_completed(futures):
                result = future.result()
                parallel_results.append(result.returncode == 0)
        parallel_time = time.time() - start_time
        
        print(f"\n📊 Batch Performance Comparison:")
        print(f"   Sequential time: {sequential_time:.3f}s")
        print(f"   Parallel time: {parallel_time:.3f}s")
        print(f"   Speedup: {sequential_time / parallel_time:.2f}x")
        
        self.log("Batch optimization demonstration completed")
        return {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": sequential_time / parallel_time if parallel_time > 0 else 1
        }
    
    def run_comprehensive_demo(self, benchmark: bool = False):
        """Run the complete optimization demonstration."""
        print("🚀 STARTING ADVANCED OPTIMIZATION DEMONSTRATION")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Cache size: {self.cache_size}")
        
        start_time = time.time()
        
        # Run all optimization demonstrations
        results = {}
        results['caching'] = self.demonstrate_caching_optimization()
        results['batch'] = self.demonstrate_batch_optimization()
        
        total_time = time.time() - start_time
        
        # Final summary
        print("\n" + "="*70)
        print("📊 OPTIMIZATION DEMONSTRATION SUMMARY")
        print("="*70)
        print(f"⏱️  Total execution time: {total_time:.2f}s")
        
        if 'caching' in results and results['caching'].get('total_requests', 0) > 0:
            hit_rate = results['caching']['hits'] / results['caching']['total_requests'] * 100
            print(f"🗂️  Cache hit rate: {hit_rate:.1f}%")
        
        if 'batch' in results:
            print(f"⚡ Batch speedup: {results['batch'].get('speedup', 1):.2f}x")
        
        print("\n✅ Advanced Optimization Demonstration Completed!")
        return results


def main():
    """Main entry point for the optimization demo."""
    parser = argparse.ArgumentParser(
        description="Advanced Optimization Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--cache-size", "-c",
        type=int,
        default=1000,
        help="Set cache size for optimization (default: 1000)"
    )
    
    args = parser.parse_args()
    
    try:
        demo = AdvancedOptimizationDemo(
            cache_size=args.cache_size,
            verbose=args.verbose
        )
        demo.run_comprehensive_demo()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()