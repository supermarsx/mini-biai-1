#!/usr/bin/env python3
"""
Cross-Platform Demo

This demo showcases cross-platform compatibility features including:
- Platform detection and adaptation
- Shell compatibility across systems
- Command translation and normalization
- Path handling differences
- Environment variable differences

Run with: python examples/tool_usage/cross_platform_demo.py
"""

import sys
import os
import platform
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.platform_adapter import get_platform_adapter
    from tool_usage.shell_detector import ShellDetector
except ImportError as e:
    print(f"Warning: Could not import platform modules: {e}")
    print("This demo requires the platform adapter components.")


class CrossPlatformDemo:
    """Demonstrates cross-platform compatibility features."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.platform_adapter = get_platform_adapter()
        self.shell_detector = ShellDetector()
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def demonstrate_platform_detection(self):
        """Demonstrate platform detection capabilities."""
        print("\n" + "="*60)
        print("1. PLATFORM DETECTION")
        print("="*60)
        
        print(f"\n🖥️  Current System Information:")
        print(f"   OS: {platform.system()}")
        print(f"   Version: {platform.release()}")
        print(f"   Architecture: {platform.machine()}")
        print(f"   Processor: {platform.processor()}")
        
        # Platform adapter info
        platform_info = self.platform_adapter.get_platform_info()
        print(f"\n🔧 Platform Adapter Information:")
        print(f"   Type: {type(self.platform_adapter).__name__}")
        print(f"   Features: {', '.join(platform_info.get('features', []))}")
        
        # Path information
        path_info = self.platform_adapter.get_path_info()
        print(f"\n📁 Path Information:")
        print(f"   Separator: '{path_info.get('path_separator', 'N/A')}'")
        print(f"   Line ending: {repr(path_info.get('line_separator', 'N/A'))}")
        
        self.log("Platform detection demonstration completed")
        return platform_info
    
    def demonstrate_shell_compatibility(self):
        """Demonstrate shell compatibility across platforms."""
        print("\n" + "="*60)
        print("2. SHELL COMPATIBILITY")
        print("="*60)
        
        # Current shell info
        current_shell = self.shell_detector.detect_current_shell()
        print(f"\n🐚 Current Shell Information:")
        print(f"   Type: {current_shell.get('type', 'Unknown')}")
        print(f"   Path: {current_shell.get('path', 'Unknown')}")
        print(f"   Version: {current_shell.get('version', 'Unknown')}")
        
        # Available shells
        available_shells = self.shell_detector.get_available_shells()
        print(f"\n🛠️  Available Shells ({len(available_shells)}):")
        for shell_name, shell_path in available_shells.items():
            print(f"   {shell_name}: {shell_path}")
        
        self.log("Shell compatibility demonstration completed")
        return {"current": current_shell, "available": available_shells}
    
    def demonstrate_path_handling(self):
        """Demonstrate path handling differences."""
        print("\n" + "="*60)
        print("3. PATH HANDLING")
        print("="*60)
        
        # Test different path formats
        test_paths = [
            "/usr/bin/python",  # Unix-style
            "/home/user/documents",  # Unix home
            "./relative/path",  # Relative path
            "../parent/path",  # Parent directory
        ]
        
        if platform.system() == "Windows":
            test_paths.extend([
                "C:\\Windows\\System32",  # Windows absolute
                "C:\\Users\\User\\Documents",  # Windows user dir
                ".\\relative\\path",  # Windows relative
                "..\\parent\\path",  # Windows parent
            ])
        
        print(f"\n📁 Path Handling Tests:")
        for path in test_paths:
            try:
                normalized = self.platform_adapter.normalize_path(path)
                absolute = self.platform_adapter.get_absolute_path(path)
                print(f"   Original: {path}")
                print(f"   Normalized: {normalized}")
                print(f"   Absolute: {absolute}")
                print()
            except Exception as e:
                print(f"   Error processing {path}: {e}")
        
        self.log("Path handling demonstration completed")
        return test_paths
    
    def demonstrate_environment_variables(self):
        """Demonstrate environment variable handling."""
        print("\n" + "="*60)
        print("4. ENVIRONMENT VARIABLES")
        print("="*60)
        
        # Get environment variables
        env_vars = self.platform_adapter.get_environment_variables()
        
        print(f"\n🌍 Key Environment Variables:")
        key_vars = ['PATH', 'HOME', 'USER', 'SHELL', 'PWD']
        if platform.system() == "Windows":
            key_vars.extend(['USERPROFILE', 'HOMEPATH', 'WINDIR'])
        
        for var in key_vars:
            value = env_vars.get(var, 'Not set')
            if len(value) > 50:
                value = value[:47] + "..."
            print(f"   {var}: {value}")
        
        print(f"\n📊 Environment Summary:")
        print(f"   Total variables: {len(env_vars)}")
        print(f"   Platform-specific: {len([v for v in env_vars.keys() if v.upper() in ['USERPROFILE', 'HOMEPATH', 'WINDIR', 'SYSTEMROOT']])}")
        
        self.log("Environment variable demonstration completed")
        return env_vars
    
    def demonstrate_command_translation(self):
        """Demonstrate command translation across platforms."""
        print("\n" + "="*60)
        print("5. COMMAND TRANSLATION")
        print("="*60)
        
        # Commands that might need translation
        commands_to_translate = [
            "ls -la",
            "find . -name '*.txt'",
            "grep -r 'pattern' .",
            "cat file.txt",
        ]
        
        print(f"\n🔄 Command Translation Tests:")
        for cmd in commands_to_translate:
            try:
                translated = self.platform_adapter.translate_command(cmd)
                print(f"   Original: {cmd}")
                print(f"   Translated: {translated}")
                print()
            except Exception as e:
                print(f"   Error translating {cmd}: {e}")
        
        self.log("Command translation demonstration completed")
        return commands_to_translate
    
    def run_comprehensive_demo(self):
        """Run the complete cross-platform demonstration."""
        print("🌍 STARTING CROSS-PLATFORM DEMONSTRATION")
        print("=" * 60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Platform: {platform.system()}")
        
        start_time = datetime.now()
        
        # Run all demonstrations
        results = {}
        results['platform'] = self.demonstrate_platform_detection()
        results['shell'] = self.demonstrate_shell_compatibility()
        results['paths'] = self.demonstrate_path_handling()
        results['environment'] = self.demonstrate_environment_variables()
        results['commands'] = self.demonstrate_command_translation()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        print("\n" + "="*60)
        print("📊 CROSS-PLATFORM DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"⏱️  Total execution time: {duration:.2f}s")
        print(f"🖥️  Platform: {platform.system()}")
        print(f"🐚 Shells detected: {len(results['shell']['available'])}")
        print(f"📁 Path tests: {len(results['paths'])}")
        print(f"🌍 Environment variables: {len(results['environment'])}")
        print(f"🔄 Commands translated: {len(results['commands'])}")
        
        print("\n✅ Cross-Platform Demonstration Completed!")
        return results


def main():
    """Main entry point for the cross-platform demo."""
    parser = argparse.ArgumentParser(
        description="Cross-Platform Compatibility Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cross_platform_demo.py              # Run full demo
  python cross_platform_demo.py --verbose   # Run with verbose output
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    try:
        demo = CrossPlatformDemo(verbose=args.verbose)
        demo.run_comprehensive_demo()
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()