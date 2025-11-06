#!/usr/bin/env python3
"""
Basic Tool Usage Demonstration

This demo showcases fundamental tool_usage capabilities including:
- Platform detection and shell identification
- Basic command execution with proper error handling
- Output formatting and result processing
- Interactive mode for command exploration
- Cross-platform compatibility basics

Run with: python examples/tool_usage/basic_usage_demo.py
"""

import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.shell_detector import ShellDetector
    from tool_usage.platform_adapter import PlatformAdapter
    from tool_usage.tool_registry import ToolRegistry
    from tool_usage.command_executor import CommandExecutor
except ImportError as e:
    print(f"Warning: Could not import tool_usage modules: {e}")
    print("This demo requires the tool_usage system to be properly installed.")
    sys.exit(1)


class BasicUsageDemo:
    """Demonstrates basic tool_usage functionality."""
    
    def __init__(self, verbose: bool = False):
        """Initialize the demo with verbose output."""
        self.verbose = verbose
        self.shell_detector = ShellDetector()
        self.platform_adapter = PlatformAdapter()
        self.tool_registry = ToolRegistry()
        self.command_executor = CommandExecutor()
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def demonstrate_platform_detection(self):
        """Demonstrate platform and shell detection capabilities."""
        print("\n" + "="*60)
        print("1. PLATFORM & SHELL DETECTION")
        print("="*60)
        
        # Detect platform
        platform_info = self.platform_adapter.get_platform_info()
        print(f"\n📱 Platform Information:")
        print(f"  OS: {platform_info.get('os', 'Unknown')}")
        print(f"  Version: {platform_info.get('version', 'Unknown')}")
        print(f"  Architecture: {platform_info.get('architecture', 'Unknown')}")
        
        # Detect shell
        shell_info = self.shell_detector.detect_current_shell()
        print(f"\n🐚 Shell Information:")
        print(f"  Type: {shell_info.get('type', 'Unknown')}")
        print(f"  Path: {shell_info.get('path', 'Unknown')}")
        print(f"  Version: {shell_info.get('version', 'Unknown')}")
        print(f"  Features: {', '.join(shell_info.get('features', []))}")
        
        # Available shells
        available_shells = self.shell_detector.get_available_shells()
        print(f"\n🛠️  Available Shells:")
        for shell_name, shell_path in available_shells.items():
            print(f"  {shell_name}: {shell_path}")
        
        self.log("Platform detection completed")
        return {
            "platform": platform_info,
            "shell": shell_info,
            "available_shells": available_shells
        }
    
    def demonstrate_command_execution(self):
        """Demonstrate basic command execution."""
        print("\n" + "="*60)
        print("2. COMMAND EXECUTION")
        print("="*60)
        
        # Test commands based on platform
        if self.platform_adapter.is_windows():
            test_commands = [
                ("echo", ["Hello from Windows!"]),
                ("dir", []),
                ("python", ["--version"]),
            ]
        else:
            test_commands = [
                ("echo", ["Hello from Unix/Linux!"]),
                ("ls", ["-la", str(Path.home())]),
                ("python3", ["--version"]),
            ]
        
        results = []
        
        for i, (cmd, args) in enumerate(test_commands, 1):
            print(f"\n🔧 Test {i}: {cmd} {' '.join(args)}")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = self.command_executor.execute(cmd, args, timeout=30)
                execution_time = time.time() - start_time
                
                print(f"  ✅ Success: {result.returncode == 0}")
                print(f"  ⏱️  Execution time: {execution_time:.3f}s")
                print(f"  📄 Output length: {len(result.stdout)} chars")
                print(f"  📝 Stderr length: {len(result.stderr)} chars")
                
                if result.stdout:
                    output_preview = result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout
                    print(f"  📋 Output preview:\n      {output_preview}")
                
                results.append({
                    "command": cmd,
                    "args": args,
                    "returncode": result.returncode,
                    "execution_time": execution_time,
                    "stdout_length": len(result.stdout),
                    "stderr_length": len(result.stderr)
                })
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results.append({
                    "command": cmd,
                    "args": args,
                    "error": str(e)
                })
        
        self.log("Command execution demonstration completed")
        return results
    
    def run_full_demo(self, interactive: bool = False):
        """Run the complete demonstration."""
        print("🚀 STARTING BASIC TOOL USAGE DEMONSTRATION")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Run all demonstrations
        results = {}
        results['platform'] = self.demonstrate_platform_detection()
        results['execution'] = self.demonstrate_command_execution()
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("📊 DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"⏱️  Total execution time: {total_time:.2f}s")
        print(f"🐚 Shell detected: {results['platform']['shell']['type']}")
        print(f"💻 Platform: {results['platform']['platform']['os']}")
        print(f"🔧 Commands tested: {len(results['execution'])}")
        
        print("\n✅ Basic Usage Demonstration Completed!")
        return results


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description="Basic Tool Usage Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basic_usage_demo.py                    # Run full demo
  python basic_usage_demo.py --verbose          # Run with verbose output
  python basic_usage_demo.py --interactive      # Include interactive mode
  python basic_usage_demo.py --command "ls -la" # Execute specific command
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enable interactive mode after demo"
    )
    
    parser.add_argument(
        "--command", "-c",
        type=str,
        help="Execute a specific command instead of running full demo"
    )
    
    args = parser.parse_args()
    
    try:
        demo = BasicUsageDemo(verbose=args.verbose)
        demo.run_full_demo(interactive=args.interactive)
            
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()