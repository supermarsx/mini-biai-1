#!/usr/bin/env python3
"""
Integration Demo - Demonstrates integration with mini-biai-1 modules

This script showcases how the tool_usage module integrates with:
- Affect System
- Memory System  
- Training System
- Coordinator
- Inference System

Features:
- Affect-aware command selection
- Memory-backed command history
- Training integration for optimization
- Coordinator routing
- Inference-driven suggestions

Usage:
    python integration_demo.py [--mode MODE] [--interactive] [--verbose]
    python integration_demo.py --help
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.platform_adapter import PlatformAdapter
    from core.shell_detector import ShellDetector
    from core.usage_optimizer import UsageOptimizer
    from core.tool_registry import ToolRegistry
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Running in fallback mode with mock integrations")

# Try importing mini-biai-1 integration modules
try:
    from integrations.affect import AffectIntegration
    from integrations.memory import MemoryIntegration
    from integrations.training import TrainingIntegration
    from integrations.coordinator import CoordinatorIntegration
    from integrations.inference import InferenceIntegration
    HAS_INTEGRATIONS = True
except ImportError:
    print("Integration modules not available, using fallback mode")
    HAS_INTEGRATIONS = False
    
    # Fallback classes for demonstration
    class AffectIntegration:
        def get_emotion_state(self):
            return "neutral"
        def should_execute_command(self, emotion, command):
            return True
    
    class MemoryIntegration:
        def store_command(self, command):
            pass
        def get_command_history(self):
            return []
    
    class TrainingIntegration:
        def record_optimization(self, before, after):
            pass
    
    class CoordinatorIntegration:
        def route_command(self, command):
            return command
    
    class InferenceIntegration:
        def suggest_optimizations(self, context):
            return []


class IntegrationDemo:
    """Demonstrates comprehensive integration capabilities"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize components
        self.platform = PlatformAdapter()
        self.shell = ShellDetector()
        self.optimizer = UsageOptimizer()
        self.registry = ToolRegistry()
        
        # Initialize integrations
        self.affect = AffectIntegration()
        self.memory = MemoryIntegration()
        self.training = TrainingIntegration()
        self.coordinator = CoordinatorIntegration()
        self.inference = InferenceIntegration()
        
        if self.verbose:
            logging.info("IntegrationDemo initialized with all components")
    
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('integration_demo.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def demonstrate_affect_integration(self):
        """Show affect-aware command execution"""
        print("\n" + "="*60)
        print("AFFECT INTEGRATION DEMONSTRATION")
        print("="*60)
        
        # Get current emotion state
        emotion = self.affect.get_emotion_state()
        print(f"Current emotional state: {emotion}")
        
        # Test affect-aware command routing
        test_commands = [
            "ls -la",
            "find /tmp -name '*.log'",
            "ps aux | grep python",
            "rm -rf /tmp/test",
            "sudo apt update"
        ]
        
        for command in test_commands:
            can_execute = self.affect.should_execute_command(emotion, command)
            status = "✓ EXECUTE" if can_execute else "✗ BLOCK"
            print(f"{status} | {command}")
            
            if can_execute:
                # Simulate command execution
                print(f"  → Executing with emotion consideration: {emotion}")
                
        return emotion
    
    async def demonstrate_memory_integration(self):
        """Show memory-backed command history"""
        print("\n" + "="*60)
        print("MEMORY INTEGRATION DEMONSTRATION")
        print("="*60)
        
        # Store sample commands
        commands = [
            "echo 'Hello World'",
            "pwd",
            "date",
            "whoami",
            "uname -a"
        ]
        
        print("Storing commands in memory...")
        for cmd in commands:
            self.memory.store_command(cmd)
            print(f"  ✓ Stored: {cmd}")
        
        # Retrieve history
        history = self.memory.get_command_history()
        print(f"\nRetrieved {len(history)} commands from memory:")
        for i, cmd in enumerate(history, 1):
            print(f"  {i}. {cmd}")
        
        return history
    
    async def demonstrate_training_integration(self):
        """Show training system integration"""
        print("\n" + "="*60)
        print("TRAINING INTEGRATION DEMONSTRATION")
        print("="*60)
        
        # Simulate optimization training
        optimizations = [
            {
                "before": "find . -name '*.txt' -exec grep -l 'pattern' {} \;",
                "after": "rg 'pattern' --type txt",
                "improvement": "3x faster execution"
            },
            {
                "before": "ls -la | grep '.py'",
                "after": "fd --type f --extension py",
                "improvement": "2x faster, better filtering"
            },
            {
                "before": "ps aux | grep python | grep -v grep",
                "after": "pgrep -f python",
                "improvement": "Simpler syntax, reliable matching"
            }
        ]
        
        print("Recording optimization improvements...")
        for opt in optimizations:
            self.training.record_optimization(opt["before"], opt["after"])
            print(f"  ✓ {opt['improvement']}")
            print(f"    Before: {opt['before']}")
            print(f"    After:  {opt['after']}")
        
        return optimizations
    
    async def demonstrate_coordinator_integration(self):
        """Show coordinator routing capabilities"""
        print("\n" + "="*60)
        print("COORDINATOR INTEGRATION DEMONSTRATION")
        print("="*60)
        
        # Test command routing
        test_commands = [
            "create_file",
            "execute_python",
            "web_search",
            "calculate",
            "analyze_data"
        ]
        
        print("Routing commands through coordinator...")
        for cmd in test_commands:
            routed = self.coordinator.route_command(cmd)
            print(f"  {cmd} → {routed}")
        
        return test_commands
    
    async def demonstrate_inference_integration(self):
        """Show inference-driven optimization suggestions"""
        print("\n" + "="*60)
        print("INFERENCE INTEGRATION DEMONSTRATION")
        print("="*60)
        
        # Context for inference
        context = {
            "current_directory": "/home/user/project",
            "recent_commands": ["ls", "cd src", "python main.py"],
            "available_tools": ["file_ops", "web_search", "code_analysis"],
            "user_skill_level": "intermediate"
        }
        
        # Get optimization suggestions
        suggestions = self.inference.suggest_optimizations(context)
        
        print(f"Context: {context}")
        print(f"\nGenerated {len(suggestions)} optimization suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        
        return suggestions
    
    async def demonstrate_full_workflow(self):
        """Execute a complete integrated workflow"""
        print("\n" + "="*60)
        print("FULL INTEGRATED WORKFLOW DEMONSTRATION")
        print("="*60)
        
        workflow_steps = [
            "1. Check emotional state",
            "2. Query memory for similar commands",
            "3. Route through coordinator",
            "4. Get inference suggestions",
            "5. Execute with training feedback"
        ]
        
        for step in workflow_steps:
            print(f"\n{step}")
            
            if "emotional" in step.lower():
                emotion = self.affect.get_emotion_state()
                print(f"  → Current state: {emotion}")
                
            elif "memory" in step.lower():
                history = self.memory.get_command_history()
                print(f"  → Found {len(history)} similar commands")
                
            elif "coordinator" in step.lower():
                print(f"  → Command routed successfully")
                
            elif "inference" in step.lower():
                suggestions = self.inference.suggest_optimizations({})
                print(f"  → {len(suggestions)} suggestions generated")
                
            elif "training" in step.lower():
                self.training.record_optimization("old_way", "optimized_way")
                print(f"  → Training feedback recorded")
            
            # Simulate processing time
            await asyncio.sleep(0.5)
        
        print("\n✓ Full workflow completed successfully")
    
    async def run_interactive_demo(self):
        """Run interactive demonstration mode"""
        print("\n" + "="*60)
        print("INTERACTIVE INTEGRATION DEMO")
        print("="*60)
        print("Commands: affect, memory, training, coordinator, inference, workflow, quit")
        
        while True:
            try:
                command = input("\nDemo> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("Goodbye!")
                    break
                elif command == "affect":
                    await self.demonstrate_affect_integration()
                elif command == "memory":
                    await self.demonstrate_memory_integration()
                elif command == "training":
                    await self.demonstrate_training_integration()
                elif command == "coordinator":
                    await self.demonstrate_coordinator_integration()
                elif command == "inference":
                    await self.demonstrate_inference_integration()
                elif command == "workflow":
                    await self.demonstrate_full_workflow()
                elif command == "help":
                    print("Available commands:")
                    for cmd in ["affect", "memory", "training", "coordinator", "inference", "workflow", "quit"]:
                        print(f"  - {cmd}")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                print("\nGoodbye!")
                break
    
    async def run_comprehensive_demo(self):
        """Run all demonstrations in sequence"""
        print("\n" + "="*80)
        print("COMPREHENSIVE INTEGRATION DEMONSTRATION")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run all demos
            await self.demonstrate_affect_integration()
            await asyncio.sleep(1)
            
            await self.demonstrate_memory_integration()
            await asyncio.sleep(1)
            
            await self.demonstrate_training_integration()
            await asyncio.sleep(1)
            
            await self.demonstrate_coordinator_integration()
            await asyncio.sleep(1)
            
            await self.demonstrate_inference_integration()
            await asyncio.sleep(1)
            
            await self.demonstrate_full_workflow()
            
            print("\n" + "="*80)
            print("DEMONSTRATION COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error during demonstration: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Integration Demo - Demonstrates mini-biai-1 module integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integration_demo.py                    # Run all demos
  python integration_demo.py --interactive      # Interactive mode
  python integration_demo.py --mode affect      # Specific demo mode
  python integration_demo.py --verbose          # Verbose output
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["affect", "memory", "training", "coordinator", "inference", "workflow", "all"],
        default="all",
        help="Demo mode to run (default: all)"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = IntegrationDemo(verbose=args.verbose)
    
    # Run appropriate demo mode
    if args.interactive:
        # Run interactive demo
        asyncio.run(demo.run_interactive_demo())
    elif args.mode == "all":
        # Run comprehensive demo
        asyncio.run(demo.run_comprehensive_demo())
    else:
        # Run specific demo
        method_name = f"demonstrate_{args.mode}_integration"
        if hasattr(demo, method_name):
            asyncio.run(getattr(demo, method_name)())
        else:
            print(f"Error: Demo mode '{args.mode}' not found")
            sys.exit(1)


if __name__ == "__main__":
    main()
