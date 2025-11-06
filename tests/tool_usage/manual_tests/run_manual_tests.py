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
        
        print("\n3. Shell Capabilities")
        if current_shell in available:
            caps = detector.get_shell_capabilities(current_shell)
            print(f"Capabilities for {current_shell}: {caps}")
        
        print("\n4. Test Shell Version Detection")
        version = detector.get_shell_version(current_shell)
        print(f"{current_shell} version: {version}")
        
        # Interactive shell simulation test
        print("\n5. Shell Simulation Test")
        shell_to_sim = self.prompt_user(
            "Enter shell name to simulate (or 'skip'):", 
            list(available.keys()) + ['skip']
        )
        
        if shell_to_sim != 'skip':
            detector_sim = ShellDetector()
            detector_sim.shell_cache[shell_to_sim] = available[shell_to_sim]
            
            simulated_caps = detector_sim.get_shell_capabilities(shell_to_sim)
            print(f"Simulated {shell_to_sim} capabilities: {simulated_caps}")
        
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
        
        print("\n2. Timeout Test")
        timeout_cmd = self.prompt_user("Enter a command that should timeout:", ['sleep 10', 'timeout 5s yes'])
        try:
            result = executor.execute(timeout_cmd, timeout=2)
            print(f"Unexpected success: {result}")
        except Exception as e:
            print(f"Expected timeout error: {e}")
        
        print("\n3. Working Directory Test")
        test_dir = "/tmp/tool_usage_test"
        os.makedirs(test_dir, exist_ok=True)
        
        result = executor.execute("pwd", working_directory=test_dir)
        print(f"Working directory test result: {result.stdout.strip()}")
        
        print("\n4. Environment Variables Test")
        test_env = {"TEST_VAR": "test_value"}
        result = executor.execute("echo $TEST_VAR", environment=test_env)
        print(f"Environment variable test: {result.stdout.strip()}")
        
        print("\n5. Invalid Command Test")
        invalid_cmd = "this_command_does_not_exist_12345"
        try:
            result = executor.execute(invalid_cmd)
            print(f"Unexpected success: {result}")
        except Exception as e:
            print(f"Expected error for invalid command: {e}")
        
        print("\nCommand Executor testing completed.")
        return True
    
    def test_tool_registry_manual(self):
        """Manual testing for ToolRegistry component."""
        print("\n" + "="*60)
        print("TOOL REGISTRY MANUAL TESTING")
        print("="*60)
        
        registry = ToolRegistry()
        
        print("\n1. Tool Discovery Test")
        discovered = registry.discover_tools()
        print(f"Discovered {len(discovered)} tools")
        for tool in list(discovered.keys())[:5]:  # Show first 5
            print(f"  - {tool}: {discovered[tool].get('description', 'No description')}")
        
        print("\n2. Tool Registration Test")
        test_tool = {
            "name": "test_tool",
            "path": "/usr/bin/test",
            "version": "1.0.0",
            "description": "Test tool for manual testing",
            "capabilities": ["test", "manual"]
        }
        
        success = registry.register_tool(test_tool)
        print(f"Tool registration success: {success}")
        
        if success:
            registered = registry.get_tool_metadata("test_tool")
            print(f"Registered tool metadata: {registered}")
        
        print("\n3. Tool Search Test")
        search_term = self.prompt_user("Enter search term (or press Enter for 'echo'):", []) or "echo"
        results = registry.search_tools(search_term)
        print(f"Search results for '{search_term}': {len(results)} tools found")
        
        print("\n4. Dependency Analysis Test")
        if results:
            test_tool_name = list(results.keys())[0]
            deps = registry.analyze_dependencies(test_tool_name)
            print(f"Dependencies for {test_tool_name}: {deps}")
        
        print("\nTool Registry testing completed.")
        return True
    
    def test_usage_optimizer_manual(self):
        """Manual testing for UsageOptimizer component."""
        print("\n" + "="*60)
        print("USAGE OPTIMIZER MANUAL TESTING")
        print("="*60)
        
        optimizer = UsageOptimizer()
        
        print("\n1. Command Pattern Analysis")
        test_commands = [
            "ls -la /home/user/documents",
            "grep -r 'pattern' /src",
            "find . -name '*.py' -type f",
            "git status && git log --oneline -10",
            "docker ps && docker images"
        ]
        
        for i, cmd in enumerate(test_commands, 1):
            print(f"\n{i}. Analyzing: {cmd}")
            
            patterns = optimizer.analyze_patterns([cmd])
            print(f"Patterns found: {patterns}")
            
            optimizations = optimizer.optimize_command(cmd)
            print(f"Optimization suggestions: {optimizations}")
            
            self.prompt_user("Press Enter to continue...")
        
        print("\n2. Learning Test")
        print("Recording command patterns for learning...")
        
        user_commands = []
        print("Enter commands (type 'done' to finish):")
        while True:
            cmd = input("Command> ").strip()
            if cmd.lower() == 'done':
                break
            if cmd:
                user_commands.append(cmd)
                optimizer.record_command_usage(cmd)
        
        if user_commands:
            patterns = optimizer.analyze_patterns(user_commands)
            print(f"Learned patterns: {patterns}")
        
        print("\n3. Performance Prediction Test")
        test_cmd = "find / -name '*.log' -type f 2>/dev/null"
        prediction = optimizer.predict_performance(test_cmd)
        print(f"Performance prediction for '{test_cmd}': {prediction}")
        
        print("\nUsage Optimizer testing completed.")
        return True
    
    def test_platform_adapter_manual(self):
        """Manual testing for PlatformAdapter components."""
        print("\n" + "="*60)
        print("PLATFORM ADAPTER MANUAL TESTING")
        print("="*60)
        
        import platform
        current_os = platform.system()
        print(f"Current OS: {current_os}")
        
        print("\n1. Platform Detection Test")
        if current_os == "Windows":
            adapter = WindowsAdapter()
            print("Using WindowsAdapter")
        else:
            adapter = POSIXAdapter()
            print("Using POSIXAdapter")
        
        print(f"Adapter type: {type(adapter).__name__}")
        
        print("\n2. Path Handling Test")
        test_paths = [
            "/home/user/documents",
            "C:\\Users\\User\\Documents",
            "./relative/path",
            "~/home/path",
            "/path with spaces/file"
        ]
        
        for path in test_paths:
            print(f"\nTesting path: {path}")
            normalized = adapter.normalize_path(path)
            print(f"Normalized: {normalized}")
            
            absolute = adapter.get_absolute_path(path)
            print(f"Absolute: {absolute}")
        
        print("\n3. Command Translation Test")
        posix_commands = ["ls -la", "grep -r 'pattern' .", "find . -name '*.py'"]
        
        for cmd in posix_commands:
            translated = adapter.translate_command(cmd)
            print(f"POSIX: {cmd}")
            print(f"Translated: {translated}")
        
        print("\nPlatform Adapter testing completed.")
        return True
    
    def run_comprehensive_workflow(self):
        """Run a comprehensive end-to-end workflow test."""
        print("\n" + "="*60)
        print("COMPREHENSIVE WORKFLOW TEST")
        print("="*60)
        
        print("This test will demonstrate the integration of all components...")
        
        # Step 1: Detect shell
        print("\nStep 1: Shell Detection")
        detector = ShellDetector()
        current_shell = detector.detect_current_shell()
        print(f"Current shell: {current_shell}")
        
        # Step 2: Execute a command
        print("\nStep 2: Command Execution")
        executor = CommandExecutor()
        result = executor.execute("echo 'Integration test'", timeout=5)
        print(f"Command result: {result.stdout.strip()}")
        
        # Step 3: Register and discover tools
        print("\nStep 3: Tool Registry")
        registry = ToolRegistry()
        discovered = registry.discover_tools()
        print(f"Discovered {len(discovered)} tools")
        
        # Step 4: Optimize command
        print("\nStep 4: Command Optimization")
        optimizer = UsageOptimizer()
        optimization = optimizer.optimize_command("ls -la /tmp")
        print(f"Optimization suggestions: {optimization}")
        
        # Step 5: Platform adaptation
        print("\nStep 5: Platform Adaptation")
        import platform
        if platform.system() == "Windows":
            adapter = WindowsAdapter()
        else:
            adapter = POSIXAdapter()
        
        test_path = "./test"
        adapted = adapter.translate_command(f"ls {test_path}")
        print(f"Platform-adapted command: {adapted}")
        
        print("\nComprehensive workflow completed successfully!")
        return True
    
    def run_all_tests(self, interactive: bool = True):
        """Run all manual tests."""
        print("Starting Manual Testing Suite for Tool Usage Module")
        print("=" * 60)
        
        if interactive:
            component = self.prompt_user(
                "Choose test component:", 
                ['all', 'shell-detector', 'command-executor', 'tool-registry', 
                 'usage-optimizer', 'platform-adapter', 'workflow']
            )
        else:
            component = 'all'
        
        start_time = time.time()
        
        try:
            if component in ['all', 'shell-detector']:
                self.test_shell_detector_manual()
            
            if component in ['all', 'command-executor']:
                self.test_command_executor_manual()
            
            if component in ['all', 'tool-registry']:
                self.test_tool_registry_manual()
            
            if component in ['all', 'usage-optimizer']:
                self.test_usage_optimizer_manual()
            
            if component in ['all', 'platform-adapter']:
                self.test_platform_adapter_manual()
            
            if component == 'workflow':
                self.run_comprehensive_workflow()
            
            end_time = time.time()
            print(f"\nTesting completed in {end_time - start_time:.2f} seconds")
            
            # Save test results
            self.save_test_results()
            
        except KeyboardInterrupt:
            print("\n\nTesting interrupted by user.")
        except Exception as e:
            print(f"\n\nTesting failed with error: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
    
    def save_test_results(self):
        """Save test results to JSON file."""
        results_file = Path(__file__).parent / "test_results.json"
        results = {
            "timestamp": time.time(),
            "test_results": self.test_results,
            "summary": f"Manual testing completed at {time.ctime()}"
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTest results saved to: {results_file}")


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
    
    # Validate arguments
    if args.component and args.component not in [
        'shell-detector', 'command-executor', 'tool-registry', 
        'usage-optimizer', 'platform-adapter', 'workflow'
    ]:
        print(f"Invalid component: {args.component}")
        print("Valid components: shell-detector, command-executor, tool-registry, usage-optimizer, platform-adapter, workflow")
        sys.exit(1)
    
    # Check if running from correct directory
    if not Path("src/tool_usage").exists():
        print("Error: src/tool_usage directory not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Run tests
    runner = ManualTestRunner(verbose=args.verbose)
    
    if args.component:
        if args.component == 'shell-detector':
            runner.test_shell_detector_manual()
        elif args.component == 'command-executor':
            runner.test_command_executor_manual()
        elif args.component == 'tool-registry':
            runner.test_tool_registry_manual()
        elif args.component == 'usage-optimizer':
            runner.test_usage_optimizer_manual()
        elif args.component == 'platform-adapter':
            runner.test_platform_adapter_manual()
        elif args.component == 'workflow':
            runner.run_comprehensive_workflow()
    else:
        runner.run_all_tests(interactive=args.interactive)


if __name__ == "__main__":
    main()