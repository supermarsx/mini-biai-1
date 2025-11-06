#!/usr/bin/env python3
"""
Quick Test Runner for Tool Usage Module

This script runs all tests for the tool_usage module and provides
a quick validation that the test suite is working correctly.

Usage:
    python quick_test_runner.py [options]

Options:
    --coverage      Generate coverage report
    --verbose       Verbose output
    --parallel      Run tests in parallel
    --fail-fast     Stop on first failure
    --component     Run tests for specific component
    --help          Show this help message
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
import json


class QuickTestRunner:
    """Quick test runner for tool_usage module."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "tests" / "tool_usage"
        self.results = {}
    
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[VERBOSE] {message}")
    
    def run_command(self, command: str, capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run a command and return the result."""
        self.log(f"Running: {command}")
        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=300  # 5 minute timeout
                )
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.project_root,
                    timeout=300
                )
            return result
        except subprocess.TimeoutExpired:
            print(f"❌ Command timed out: {command}")
            return subprocess.CompletedProcess(command, -1, "", "Command timed out")
        except Exception as e:
            print(f"❌ Command failed: {command}")
            print(f"Error: {e}")
            return subprocess.CompletedProcess(command, -1, "", str(e))
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        required_packages = ["pytest", "pytest-cov", "unittest"]
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == "pytest":
                    import pytest
                elif package == "pytest-cov":
                    import pytest_cov
                elif package == "unittest":
                    import unittest
                self.log(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} is missing")
        
        if missing_packages:
            print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install pytest pytest-cov")
            return False
        
        # Check if source directory exists
        src_dir = self.project_root / "src" / "tool_usage"
        if not src_dir.exists():
            print(f"❌ Source directory not found: {src_dir}")
            print("Make sure you're running from the project root directory")
            return False
        
        self.log("✅ All dependencies checked")
        return True
    
    def run_basic_validation(self) -> bool:
        """Run basic validation tests."""
        print("\n🔬 Running basic validation...")
        
        tests = [
            {
                "name": "Import Tests",
                "command": "python -c \"from src.tool_usage.shell_detector import ShellDetector; from src.tool_usage.command_executor import CommandExecutor; from src.tool_usage.tool_registry import ToolRegistry; print('✅ All imports successful')\"",
            },
            {
                "name": "Test Directory Check",
                "command": f"python -c \"import os; test_dir='{self.test_dir}'; assert os.path.exists(test_dir), f'Test directory {{test_dir}} not found'; assert os.path.exists(os.path.join(test_dir, 'conftest.py')), 'conftest.py not found'; print('✅ Test directory structure valid')\"",
            },
            {
                "name": "Configuration Check",
                "command": f"python -c \"from tests.tool_usage.conftest import *; print('✅ Test configuration loaded')\"",
            }
        ]
        
        all_passed = True
        for test in tests:
            result = self.run_command(test["command"])
            if result.returncode == 0:
                print(f"✅ {test['name']}")
            else:
                print(f"❌ {test['name']}")
                if self.verbose:
                    print(f"Error: {result.stderr}")
                all_passed = False
        
        return all_passed
    
    def run_unit_tests(self, component: str = None) -> Dict[str, Any]:
        """Run unit tests for components."""
        print("\n🧪 Running unit tests...")
        
        if component:
            test_pattern = f"tests/tool_usage/unit/test_{component}.py"
            print(f"Running tests for: {component}")
        else:
            test_pattern = "tests/tool_usage/unit/"
            print("Running all unit tests")
        
        command = f"python -m pytest {test_pattern} -v --tb=short"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["unit_tests"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Unit tests passed")
        else:
            print("❌ Unit tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                print("STDERR:", result.stderr[-500:])
        
        return self.results["unit_tests"]
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests."""
        print("\n🔗 Running integration tests...")
        
        command = "python -m pytest tests/tool_usage/integration/ -v --tb=short"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["integration_tests"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["integration_tests"]
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests."""
        print("\n🔒 Running security tests...")
        
        command = "python -m pytest tests/tool_usage/security/ -v --tb=short"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["security_tests"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Security tests passed")
        else:
            print("❌ Security tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["security_tests"]
    
    def run_compatibility_tests(self) -> Dict[str, Any]:
        """Run compatibility tests."""
        print("\n🌐 Running compatibility tests...")
        
        command = "python -m pytest tests/tool_usage/compatibility/ -v --tb=short"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["compatibility_tests"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Compatibility tests passed")
        else:
            print("❌ Compatibility tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["compatibility_tests"]
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests."""
        print("\n⚡ Running performance tests...")
        
        command = "python -m pytest tests/tool_usage/performance/ -v --tb=short -s"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["performance_tests"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Performance tests passed")
        else:
            print("❌ Performance tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["performance_tests"]
    
    def run_all_tests_with_coverage(self) -> Dict[str, Any]:
        """Run all tests with coverage reporting."""
        print("\n📊 Running all tests with coverage...")
        
        command = "python -m pytest tests/tool_usage/ --cov=src.tool_usage --cov-report=term --cov-report=html -v"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["all_tests_with_coverage"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ All tests with coverage passed")
            # Extract coverage percentage from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'TOTAL' in line and '%' in line:
                    print(f"📈 Coverage: {line}")
                    break
        else:
            print("❌ Tests with coverage failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["all_tests_with_coverage"]
    
    def run_parallel_tests(self) -> Dict[str, Any]:
        """Run tests in parallel."""
        print("\n🚀 Running tests in parallel...")
        
        command = "python -m pytest tests/tool_usage/ -n auto -v"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["parallel_tests"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("✅ Parallel tests passed")
        else:
            print("❌ Parallel tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["parallel_tests"]
    
    def run_component_tests(self, component: str) -> Dict[str, Any]:
        """Run tests for a specific component."""
        print(f"\n🎯 Running tests for {component}...")
        
        test_files = {
            "shell-detector": "tests/tool_usage/unit/test_shell_detector.py",
            "command-executor": "tests/tool_usage/unit/test_command_executor.py",
            "tool-registry": "tests/tool_usage/unit/test_tool_registry.py",
            "usage-optimizer": "tests/tool_usage/unit/test_usage_optimizer.py",
            "platform-adapter": "tests/tool_usage/unit/test_platform_adapter.py"
        }
        
        if component not in test_files:
            print(f"❌ Unknown component: {component}")
            print(f"Available components: {', '.join(test_files.keys())}")
            return {"success": False}
        
        test_file = test_files[component]
        command = f"python -m pytest {test_file} -v --tb=short"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results[f"component_{component}"] = {
            "command": command,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print(f"✅ {component} tests passed")
        else:
            print(f"❌ {component} tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results[f"component_{component}"]
    
    def run_specific_test_pattern(self, pattern: str) -> Dict[str, Any]:
        """Run tests matching a specific pattern."""
        print(f"\n🔍 Running tests matching pattern: {pattern}")
        
        command = f"python -m pytest tests/tool_usage/ -k '{pattern}' -v"
        
        start_time = time.time()
        result = self.run_command(command)
        end_time = time.time()
        
        self.results["pattern_tests"] = {
            "command": command,
            "pattern": pattern,
            "return_code": result.returncode,
            "execution_time": end_time - start_time,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print(f"✅ Pattern '{pattern}' tests passed")
        else:
            print(f"❌ Pattern '{pattern}' tests failed")
            if self.verbose:
                print("STDOUT:", result.stdout[-500:])
                print("STDERR:", result.stderr[-500:])
        
        return self.results["pattern_tests"]
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of test results."""
        print("\n📋 Test Summary Report")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("success", False))
        failed_tests = total_tests - passed_tests
        
        print(f"Total test suites run: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n" + "=" * 60)
        print("Individual Test Results:")
        print("=" * 60)
        
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
            time_taken = result.get("execution_time", 0)
            print(f"{test_name:30} {status:12} ({time_taken:.2f}s)")
        
        return f"Summary: {passed_tests}/{total_tests} test suites passed"
    
    def save_results(self, filename: str = "quick_test_results.json"):
        """Save test results to JSON file."""
        results_file = Path(__file__).parent / filename
        
        output_data = {
            "timestamp": time.time(),
            "results": self.results,
            "summary": {
                "total_suites": len(self.results),
                "passed_suites": sum(1 for r in self.results.values() if r.get("success", False)),
                "failed_suites": sum(1 for r in self.results.values() if not r.get("success", False)),
                "total_execution_time": sum(r.get("execution_time", 0) for r in self.results.values())
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def run_all(self, coverage: bool = False, parallel: bool = False, 
                fail_fast: bool = False, component: str = None) -> bool:
        """Run the complete test suite."""
        print("🚀 Quick Test Runner for Tool Usage Module")
        print("=" * 60)
        
        # Check dependencies first
        if not self.check_dependencies():
            return False
        
        # Run basic validation
        if not self.run_basic_validation():
            print("❌ Basic validation failed")
            return False
        
        success = True
        
        # Run tests based on options
        if component:
            result = self.run_component_tests(component)
            success = success and result.get("success", False)
        else:
            # Unit tests
            self.run_unit_tests()
            success = success and self.results["unit_tests"].get("success", False)
            if fail_fast and not success:
                return False
            
            # Integration tests
            self.run_integration_tests()
            success = success and self.results["integration_tests"].get("success", False)
            if fail_fast and not success:
                return False
            
            # Security tests
            self.run_security_tests()
            success = success and self.results["security_tests"].get("success", False)
            if fail_fast and not success:
                return False
            
            # Compatibility tests
            self.run_compatibility_tests()
            success = success and self.results["compatibility_tests"].get("success", False)
            if fail_fast and not success:
                return False
            
            # Performance tests
            self.run_performance_tests()
            success = success and self.results["performance_tests"].get("success", False)
            
            # Coverage if requested
            if coverage:
                self.run_all_tests_with_coverage()
                success = success and self.results["all_tests_with_coverage"].get("success", False)
            
            # Parallel tests if requested
            if parallel:
                self.run_parallel_tests()
                success = success and self.results["parallel_tests"].get("success", False)
        
        # Generate summary
        self.generate_summary_report()
        self.save_results()
        
        return success


def main():
    """Main entry point for quick test runner."""
    parser = argparse.ArgumentParser(description="Quick Test Runner for Tool Usage Module")
    parser.add_argument("--coverage", action="store_true", 
                       help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", 
                       help="Verbose output")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run tests in parallel")
    parser.add_argument("--fail-fast", action="store_true", 
                       help="Stop on first failure")
    parser.add_argument("--component", 
                       choices=["shell-detector", "command-executor", "tool-registry", 
                               "usage-optimizer", "platform-adapter"],
                       help="Run tests for specific component")
    parser.add_argument("--pattern", 
                       help="Run tests matching pattern")
    
    args = parser.parse_args()
    
    # Check if running from correct directory
    test_dir = Path("tests/tool_usage")
    if not test_dir.exists():
        print("Error: tests/tool_usage directory not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Create runner
    runner = QuickTestRunner(verbose=args.verbose)
    
    # Run tests
    try:
        if args.pattern:
            success = runner.run_specific_test_pattern(args.pattern)
        else:
            success = runner.run_all(
                coverage=args.coverage,
                parallel=args.parallel,
                fail_fast=args.fail_fast,
                component=args.component
            )
        
        if success:
            print("\n🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Some tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTesting failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()