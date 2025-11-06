#!/usr/bin/env python3
"""
Run All Tool Usage Demos - Comprehensive demonstration runner

This script runs all tool usage demonstrations in sequence or parallel:
- Basic Usage Demo
- Advanced Optimization Demo
- Security Features Demo
- Cross-Platform Demo
- Integration Demo
- Learning Demo

Features:
- Sequential or parallel execution
- Individual demo selection
- Detailed reporting and results
- Performance benchmarking
- Cross-platform compatibility testing
- Interactive and batch modes

Usage:
    python run_all_tool_usage_demos.py [--parallel] [--demo DEMO_NAME] [--report]
    python run_all_tool_usage_demos.py --list-demos
    python run_all_tool_usage_demos.py --help
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Demo configurations
DEMO_CONFIGS = {
    "basic": {
        "name": "Basic Usage Demo",
        "file": "basic_usage_demo.py",
        "description": "Fundamental tool usage capabilities",
        "estimated_time": "2-3 minutes",
        "dependencies": ["core.platform_adapter", "core.shell_detector"],
        "category": "Core Functionality"
    },
    "advanced": {
        "name": "Advanced Optimization Demo",
        "file": "advanced_optimization_demo.py",
        "description": "Advanced optimization strategies and caching",
        "estimated_time": "3-5 minutes",
        "dependencies": ["core.usage_optimizer", "core.tool_registry"],
        "category": "Performance"
    },
    "security": {
        "name": "Security Features Demo",
        "file": "security_features_demo.py",
        "description": "Security validation and sandboxing",
        "estimated_time": "2-4 minutes",
        "dependencies": ["core.security_validator", "core.sandbox"],
        "category": "Security"
    },
    "cross_platform": {
        "name": "Cross-Platform Demo",
        "file": "cross_platform_demo.py",
        "description": "Cross-platform compatibility testing",
        "estimated_time": "3-4 minutes",
        "dependencies": ["core.platform_adapter"],
        "category": "Compatibility"
    },
    "integration": {
        "name": "Integration Demo",
        "file": "integration_demo.py",
        "description": "Integration with mini-biai-1 modules",
        "estimated_time": "4-6 minutes",
        "dependencies": ["integrations.affect", "integrations.memory"],
        "category": "Integration"
    },
    "learning": {
        "name": "Learning Demo",
        "file": "learning_demo.py",
        "description": "Machine learning and intelligence capabilities",
        "estimated_time": "5-7 minutes",
        "dependencies": ["learning.analytics", "learning.patterns"],
        "category": "Intelligence"
    }
}


class DemoRunner:
    """Comprehensive demo runner with parallel execution support"""
    
    def __init__(self, verbose: bool = False, output_dir: str = "demo_results"):
        self.verbose = verbose
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
        self.results = {}
        
        if self.verbose:
            logging.info("DemoRunner initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        
        # Create file handler
        log_file = self.output_dir / "demo_runner.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Configure formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self, demo_name: str) -> List[str]:
        """Check if demo dependencies are available"""
        if demo_name not in DEMO_CONFIGS:
            return [f"Demo '{demo_name}' not found"]
        
        config = DEMO_CONFIGS[demo_name]
        missing_deps = []
        
        for dep in config["dependencies"]:
            try:
                # Try to import dependency
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        return missing_deps
    
    def run_demo_script(self, demo_name: str, args: List[str] = None) -> Dict[str, Any]:
        """Run a specific demo script"""
        if demo_name not in DEMO_CONFIGS:
            return {
                "success": False,
                "error": f"Demo '{demo_name}' not found",
                "duration": 0
            }
        
        config = DEMO_CONFIGS[demo_name]
        script_path = Path(__file__).parent / config["file"]
        
        if not script_path.exists():
            return {
                "success": False,
                "error": f"Demo script not found: {script_path}",
                "duration": 0
            }
        
        # Check dependencies
        missing_deps = self.check_dependencies(demo_name)
        if missing_deps:
            return {
                "success": False,
                "error": f"Missing dependencies: {', '.join(missing_deps)}",
                "duration": 0,
                "dependencies": missing_deps
            }
        
        # Prepare command
        cmd = [sys.executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run the demo
        start_time = time.time()
        
        try:
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            success = result.returncode == 0
            
            return {
                "success": success,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Demo execution timeout (5 minutes)",
                "duration": 300
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def run_demo_async(self, demo_name: str, args: List[str] = None) -> Dict[str, Any]:
        """Run demo in async mode"""
        async def _run():
            return self.run_demo_script(demo_name, args)
        
        return asyncio.run(_run())
    
    def run_parallel_demos(self, demo_names: List[str], max_workers: int = 3) -> Dict[str, Dict[str, Any]]:
        """Run multiple demos in parallel"""
        self.logger.info(f"Running {len(demo_names)} demos in parallel (max_workers={max_workers})")
        
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all demo tasks
            future_to_demo = {
                executor.submit(self.run_demo_script, demo_name): demo_name 
                for demo_name in demo_names
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_demo):
                demo_name = future_to_demo[future]
                try:
                    result = future.result()
                    results[demo_name] = result
                    status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
                    duration = result.get("duration", 0)
                    self.logger.info(f"{status} {demo_name} ({duration:.1f}s)")
                except Exception as e:
                    results[demo_name] = {
                        "success": False,
                        "error": str(e),
                        "duration": 0
                    }
                    self.logger.error(f"✗ FAILED {demo_name}: {e}")
        
        return results
    
    def run_sequential_demos(self, demo_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Run multiple demos sequentially"""
        self.logger.info(f"Running {len(demo_names)} demos sequentially")
        
        results = {}
        total_start = time.time()
        
        for i, demo_name in enumerate(demo_names, 1):
            self.logger.info(f"Running demo {i}/{len(demo_names)}: {demo_name}")
            
            result = self.run_demo_script(demo_name)
            results[demo_name] = result
            
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            duration = result.get("duration", 0)
            self.logger.info(f"{status} {demo_name} ({duration:.1f}s)")
            
            if not result["success"]:
                self.logger.error(f"Demo {demo_name} failed: {result.get('error', 'Unknown error')}")
        
        total_duration = time.time() - total_start
        self.logger.info(f"All demos completed in {total_duration:.1f} seconds")
        
        return results
    
    def generate_report(self, results: Dict[str, Dict[str, Any]], output_file: str = None) -> str:
        """Generate comprehensive demo execution report"""
        if not output_file:
            output_file = self.output_dir / f"demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report_lines = [
            "# Tool Usage Demos - Execution Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Demos:** {len(results)}",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Calculate summary statistics
        successful = sum(1 for r in results.values() if r["success"])
        failed = len(results) - successful
        total_duration = sum(r.get("duration", 0) for r in results.values())
        
        report_lines.extend([
            f"- **Successful:** {successful}",
            f"- **Failed:** {failed}",
            f"- **Success Rate:** {successful/len(results)*100:.1f}%",
            f"- **Total Duration:** {total_duration:.1f} seconds",
            "",
            "## Demo Results",
            ""
        ])
        
        # Add individual demo results
        for demo_name, result in results.items():
            config = DEMO_CONFIGS.get(demo_name, {})
            
            status_emoji = "✅" if result["success"] else "❌"
            duration = result.get("duration", 0)
            
            report_lines.extend([
                f"### {status_emoji} {config.get('name', demo_name)}",
                "",
                f"- **Status:** {'Success' if result['success'] else 'Failed'}",
                f"- **Duration:** {duration:.1f} seconds",
                f"- **Category:** {config.get('category', 'Unknown')}",
                f"- **Description:** {config.get('description', 'N/A')}",
                ""
            ])
            
            if not result["success"]:
                error = result.get("error", "Unknown error")
                report_lines.extend([
                    f"- **Error:** {error}",
                    ""
                ])
                
                if "dependencies" in result:
                    deps = ", ".join(result["dependencies"])
                    report_lines.extend([
                        f"- **Missing Dependencies:** {deps}",
                        ""
                    ])
            
            if self.verbose and "stdout" in result and result["stdout"]:
                report_lines.extend([
                    "**Output:**",
                    "```",
                    result["stdout"][:1000] + ("..." if len(result["stdout"]) > 1000 else ""),
                    "```",
                    ""
                ])
        
        # Add recommendations
        if failed > 0:
            report_lines.extend([
                "## Recommendations",
                "",
                "### Failed Demos",
                ""
            ])
            
            for demo_name, result in results.items():
                if not result["success"]:
                    config = DEMO_CONFIGS.get(demo_name, {})
                    report_lines.extend([
                        f"#### {config.get('name', demo_name)}",
                        ""
                    ])
                    
                    if "dependencies" in result:
                        deps = result["dependencies"]
                        if deps:
                            report_lines.extend([
                                f"- Install missing dependencies: `pip install {' '.join(deps)}`",
                                ""
                            ])
                    
                    if "error" in result:
                        error = result["error"]
                        if "not found" in error.lower():
                            report_lines.extend([
                                "- Check that all required modules are properly installed",
                                "- Verify the demo script file exists and is executable"
                            ])
                        elif "timeout" in error.lower():
                            report_lines.extend([
                                "- Consider running the demo with more time or on a faster system",
                                "- Check for resource constraints or system load"
                            ])
                        else:
                            report_lines.extend([
                                f"- Review the specific error: {error}",
                                "- Check the detailed logs for more information"
                            ])
                    
                    report_lines.append("")
        
        # Add system information
        report_lines.extend([
            "## System Information",
            "",
            f"- **Python Version:** {sys.version}",
            f"- **Platform:** {sys.platform}",
            f"- **Working Directory:** {os.getcwd()}",
            f"- **Script Location:** {Path(__file__).absolute()}",
            ""
        ])
        
        # Write report
        report_content = "\n".join(report_lines)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Report generated: {output_file}")
        return str(output_file)
    
    def list_available_demos(self) -> None:
        """List all available demos with descriptions"""
        print("\nAvailable Tool Usage Demos:")
        print("=" * 60)
        
        # Group by category
        by_category = {}
        for name, config in DEMO_CONFIGS.items():
            category = config.get("category", "Other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, config))
        
        for category, demos in by_category.items():
            print(f"\n{category}:")
            for demo_name, config in demos:
                print(f"  • {demo_name:15} - {config['description']}")
                print(f"    {'':17} File: {config['file']}")
                print(f"    {'':17} Time: {config['estimated_time']}")
    
    def run_selected_demos(self, demo_names: List[str], parallel: bool = False, args: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Run selected demos with specified execution mode"""
        # Validate demo names
        valid_names = []
        invalid_names = []
        
        for name in demo_names:
            if name in DEMO_CONFIGS:
                valid_names.append(name)
            else:
                invalid_names.append(name)
        
        if invalid_names:
            print(f"Warning: Unknown demos: {', '.join(invalid_names)}")
            print(f"Available demos: {', '.join(DEMO_CONFIGS.keys())}")
        
        if not valid_names:
            print("No valid demos selected.")
            return {}
        
        print(f"\nRunning {len(valid_names)} demos {'in parallel' if parallel else 'sequentially'}...")
        
        # Run demos
        if parallel:
            results = self.run_parallel_demos(valid_names)
        else:
            results = self.run_sequential_demos(valid_names)
        
        # Store results
        self.results = results
        
        # Generate report
        report_file = self.generate_report(results)
        
        # Print summary
        successful = sum(1 for r in results.values() if r["success"])
        failed = len(results) - successful
        
        print(f"\n{'='*60}")
        print("EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {failed}/{len(results)}")
        print(f"Report generated: {report_file}")
        
        if failed > 0:
            print("\nFailed demos:")
            for demo_name, result in results.items():
                if not result["success"]:
                    print(f"  ✗ {demo_name}: {result.get('error', 'Unknown error')}")
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run All Tool Usage Demos - Comprehensive demonstration runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tool_usage_demos.py                    # Run all demos sequentially
  python run_all_tool_usage_demos.py --parallel         # Run all demos in parallel
  python run_all_tool_usage_demos.py --demo basic       # Run specific demo
  python run_all_tool_usage_demos.py --demo basic advanced  # Run multiple demos
  python run_all_tool_usage_demos.py --list-demos       # List available demos
  python run_all_tool_usage_demos.py --report           # Generate detailed report
        """
    )
    
    parser.add_argument(
        "--demo", 
        action="append",
        choices=list(DEMO_CONFIGS.keys()),
        help="Specific demo to run (can be used multiple times)"
    )
    
    parser.add_argument(
        "--list-demos", 
        action="store_true",
        help="List all available demos and exit"
    )
    
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run demos in parallel (default: sequential)"
    )
    
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate detailed execution report"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="demo_results",
        help="Output directory for results and reports (default: demo_results)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--max-workers", 
        type=int,
        default=3,
        help="Maximum number of parallel workers (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = DemoRunner(verbose=args.verbose, output_dir=args.output_dir)
    
    # Handle list demos
    if args.list_demos:
        runner.list_available_demos()
        return
    
    # Determine which demos to run
    if args.demo:
        demo_names = args.demo
    else:
        demo_names = list(DEMO_CONFIGS.keys())
    
    print(f"Starting demo execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target demos: {', '.join(demo_names)}")
    
    # Run demos
    try:
        results = runner.run_selected_demos(
            demo_names=demo_names,
            parallel=args.parallel,
            args=["--verbose"] if args.verbose else []
        )
        
        # Exit with appropriate code
        failed = sum(1 for r in results.values() if not r["success"])
        sys.exit(0 if failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\nDemo execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error during demo execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
