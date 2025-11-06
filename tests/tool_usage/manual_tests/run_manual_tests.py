#!/usr/bin/env python3
"""
Manual Testing Suite for Tool Usage Module

This script provides interactive testing scenarios for manual validation
of the tool_usage module components. It allows users to test real-world
scenarios that may be difficult to test with automated tests.

Usage:
    python run_manual_tests.py [options]

Options:
    --component   Test specific component (shell-detector, command-executor, etc.)
    --interactive Run in interactive mode with prompts
    --verbose     Enable verbose output
    --help        Show this help message
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from src.tool_usage.shell_detector import ShellDetector
    from src.tool_usage.command_executor import CommandExecutor
    from src.tool_usage.tool_registry import ToolRegistry
    from src.tool_usage.usage_optimizer import UsageOptimizer
    from src.tool_usage.platform_adapter import POSIXAdapter, WindowsAdapter
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)


class ManualTestRunner:
    """Interactive manual testing runner for tool_usage module."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.test_results = {}
        
    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[VERBOSE] {message}")
    
    def prompt_user(self, message: str, choices: Optional[List[str]] = None) -> str:
        """Prompt user for input with optional choices."""
        if choices:
            print(f"{message}")
            print(f"Choices: {', '.join(choices)}")
            while True:
                response = input("> ").strip()
                if response in choices:
                    return response
                print(f"Please choose from: {', '.join(choices)}")
        else:
            return input(f"{message} > ").strip()
    
    def test_shell_detector_manual(self):
        """Manual testing for ShellDetector component."""
        print("\n" + "="*60)
        print("SHELL DETECTOR MANUAL TESTING")
        print("="*60)
        
        detector = ShellDetector()
        
        print("\n1. Current Shell Detection")
        current_shell = detector.detect_current_shell()
        print(f"Detected current shell: {current_shell}")
        
        print("\n2. Available Shells Check")
        available = detector.get_available_shells()
        print(f"Available shells: {list(available.keys())}")
        
        print("\nShell Detector testing completed.")
        return True
    
    def test_command_executor_manual(self):
        """Manual testing for CommandExecutor component."""
        print("\n" + "="*60)
        print("COMMAND EXECUTOR MANUAL TESTING")
        print("="*60)
        
        executor = CommandExecutor()
        
        print("\n1. Safe Command Execution Test")
        safe_commands = ["echo 'Hello World'", "ls", "pwd", "date"]
        
        for cmd in safe_commands:
            print(f"\nTesting command: {cmd}")
            try:
                result = executor.execute(cmd, timeout=5)
                print(f"Return code: {result.returncode}")
                print(f"Stdout: {result.stdout[:200]}...")
                print(f"Stderr: {result.stderr[:200]}...")
                
                if self.verbose:
                    print(f"Execution time: {result.execution_time:.3f}s")
                
                self.prompt_user("Press Enter to continue...")
            except Exception as e:
                print(f"Error executing {cmd}: {e}")
        
        print("\nCommand Executor testing completed.")
        return True
    
    def run_all_tests(self, interactive: bool = True):
        """Run all manual tests."""
        print("Starting Manual Testing Suite for Tool Usage Module")
        print("=" * 60)
        
        if interactive:
            component = self.prompt_user(
                "Choose test component", 
                ['all', 'shell-detector', 'command-executor']
            )
        else:
            component = 'all'
        
        start_time = time.time()
        
        try:
            if component in ['all', 'shell-detector']:
                self.test_shell_detector_manual()
            
            if component in ['all', 'command-executor']:
                self.test_command_executor_manual()
            
            end_time = time.time()
            print(f"\nTesting completed in {end_time - start_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\nTesting interrupted by user.")
        except Exception as e:
            print(f"\n\nTesting failed with error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()


def main():
    """Main entry point for manual testing."""
    parser = argparse.ArgumentParser(description="Manual Testing Suite for Tool Usage Module")
    parser.add_argument("--component", help="Test specific component")
    parser.add_argument("--interactive", action="store_true", default=True, 
                       help="Run in interactive mode")
    parser.add_argument("--non-interactive", dest="interactive", action="store_false",
                       help="Run in non-interactive mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Run tests
    runner = ManualTestRunner(verbose=args.verbose)
    runner.run_all_tests(interactive=args.interactive)


if __name__ == "__main__":
    main()