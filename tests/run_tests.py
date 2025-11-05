#!/usr/bin/env python3
"""
Test Runner Script for mini-biai-1 Testing Suite
===============================================

Provides convenient commands for running different types of tests
with appropriate configurations and reporting.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path


def run_command(cmd, description=None, check=True):
    """Run a command and handle output"""
    if description:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {cmd}")
        print('='*60)
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=False)
        elapsed = time.time() - start_time
        
        if description:
            print(f"\n✓ {description} completed in {elapsed:.2f}s")
        
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {description} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="mini-biai-1 Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s all              # Run all tests
  %(prog)s unit             # Run unit tests only
  %(prog)s integration      # Run integration tests
  %(prog)s performance      # Run performance tests
  %(prog)s stress           # Run stress tests (slow)
  %(prog)s fast             # Run fast tests only
  %(prog)s faiss            # Run FAISS tests
  %(prog)s spiking          # Run spiking tests
  %(prog)s memory           # Run memory tests
  %(prog)s pipeline         # Run pipeline tests
  %(prog)s coverage         # Run tests with coverage
  %(prog)s benchmark        # Run performance benchmarks
  %(prog)s quality          # Run code quality checks
  %(prog)s ci               # Run CI-compatible test suite
        """
    )
    
    parser.add_argument(
        'test_type',
        choices=[
            'all', 'unit', 'integration', 'performance', 'stress', 'fast',
            'faiss', 'spiking', 'memory', 'pipeline',
            'coverage', 'benchmark', 'quality', 'ci', 'help'
        ],
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Run tests in parallel (requires pytest-xdist)'
    )
    
    parser.add_argument(
        '--no-cov',
        action='store_true',
        help='Skip coverage reporting'
    )
    
    parser.add_argument(
        '--fail-fast',
        action='store_true',
        help='Stop on first test failure'
    )
    
    args = parser.parse_args()
    
    if args.test_type == 'help':
        parser.print_help()
        return 0
    
    # Base pytest command
    base_cmd = "python -m pytest"
    
    # Add verbosity
    if args.verbose:
        base_cmd += " -v"
    else:
        base_cmd += " -q"
    
    # Add parallel execution
    if args.parallel:
        base_cmd += " -n auto"
    
    # Add fail fast
    if args.fail_fast:
        base_cmd += " -x"
    
    # Test type configurations
    test_configs = {
        'all': {
            'cmd': f"{base_cmd} tests/ --cov=src --cov-report=term-missing",
            'description': 'All tests',
            'marker': None
        },
        'unit': {
            'cmd': f"{base_cmd} tests/ -m 'unit'",
            'description': 'Unit tests',
            'marker': 'unit'
        },
        'integration': {
            'cmd': f"{base_cmd} tests/ -m 'integration'",
            'description': 'Integration tests',
            'marker': 'integration'
        },
        'performance': {
            'cmd': f"{base_cmd} tests/ -m 'performance' --benchmark-json=benchmark.json --timeout=600",
            'description': 'Performance tests and benchmarks',
            'marker': 'performance'
        },
        'faiss': {
            'cmd': f"{base_cmd} tests/test_faiss.py",
            'description': 'FAISS vector database tests',
            'marker': 'faiss'
        },
        'spiking': {
            'cmd': f"{base_cmd} tests/test_spiking.py",
            'description': 'Spiking neural network tests',
            'marker': 'spiking'
        },
        'memory': {
            'cmd': f"{base_cmd} tests/test_memory.py",
            'description': 'Memory system tests',
            'marker': 'memory'
        },
        'pipeline': {
            'cmd': f"{base_cmd} tests/test_pipeline.py",
            'description': 'Pipeline integration tests',
            'marker': 'pipeline'
        },
        'coverage': {
            'cmd': f"{base_cmd} tests/ --cov=src --cov-report=html:htmlcov --cov-report=term-missing --cov-fail-under=80",
            'description': 'Tests with coverage report',
            'marker': None
        },
        'benchmark': {
            'cmd': f"{base_cmd} tests/ -m 'performance' --benchmark-json=benchmark.json --timeout=600",
            'description': 'Performance benchmarks',
            'marker': 'performance'
        },
        'stress': {
            'cmd': f"{base_cmd} tests/ -m 'slow' --timeout=900",
            'description': 'Stress tests and long-running tests',
            'marker': 'slow'
        },
        'quality': {
            'cmd': f"{base_cmd} --collect-only tests/ -q",
            'description': 'Code quality checks (collection only)',
            'marker': None
        },
        'fast': {
            'cmd': f"{base_cmd} tests/ -m 'not slow and not performance'",
            'description': 'Fast tests (unit + integration, excludes slow/performance)',
            'marker': 'not slow and not performance'
        },
        'ci': {
            'cmd': f"{base_cmd} tests/ --cov=src --cov-report=xml --cov-fail-under=80 -m 'not slow'",
            'description': 'CI-compatible test suite',
            'marker': None
        }
    }
    
    if args.test_type not in test_configs:
        print(f"Unknown test type: {args.test_type}")
        return 1
    
    config = test_configs[args.test_type]
    
    # Skip coverage if requested
    if args.no_cov and '--cov=' not in config['cmd']:
        config['cmd'] = config['cmd'].replace('tests/', 'tests/ --no-cov ')
    
    # Run the test
    success = run_command(config['cmd'], config['description'])
    
    if success:
        print(f"\n🎉 {config['description']} passed successfully!")
        
        # Additional actions for specific test types
        if args.test_type == 'coverage':
            print("\n📊 Coverage report generated:")
            print("   - Terminal: See output above")
            print("   - HTML: htmlcov/index.html")
            
        elif args.test_type == 'benchmark':
            print("\n⚡ Benchmark results:")
            print("   - JSON: benchmark.json")
            
        elif args.test_type == 'ci':
            print("\n✅ CI tests completed - ready for continuous integration!")
            
        return 0
    else:
        print(f"\n❌ {config['description']} failed!")
        print("\nTroubleshooting tips:")
        print("1. Check that all dependencies are installed: pip install -r tests/requirements.txt")
        print("2. Ensure mini-biai-1 source is in PYTHONPATH")
        print("3. Run with --verbose for more detailed output")
        print("4. Check individual test files for specific failures")
        return 1


if __name__ == "__main__":
    sys.exit(main())