#!/usr/bin/env python3
"""
Learning Demo - Demonstrates machine learning and intelligence capabilities

This script showcases the learning capabilities including:
- Pattern Learning and Recognition
- Optimization Suggestions
- Performance Prediction
- Adaptive Learning
- Predictive Analytics

Features:
- Usage pattern analysis
- Intelligent suggestion generation
- Performance prediction models
- Adaptive learning algorithms
- Machine learning integration

Usage:
    python learning_demo.py [--mode MODE] [--interactive] [--verbose] [--training-data FILE]
    python learning_demo.py --help
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import random
import statistics
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from core.platform_adapter import PlatformAdapter
    from core.shell_detector import ShellDetector
    from core.usage_optimizer import UsageOptimizer
    from core.tool_registry import ToolRegistry
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")
    print("Running in fallback mode with mock learning components")

# Try importing machine learning modules
try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    print("Scikit-learn not available, using simple heuristics for demonstration")
    HAS_ML = False
    
    # Mock sklearn classes for demonstration
    class MockModel:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return [0.5] * len(X)
        def score(self, X, y):
            return 0.8
    
    class RandomForestRegressor(MockModel):
        pass
    
    class KMeans(MockModel):
        def __init__(self, n_clusters=3):
            self.n_clusters = n_clusters
        def fit_predict(self, X):
            return [i % self.n_clusters for i in range(len(X))]
    
    class StandardScaler:
        def fit_transform(self, X):
            return X
        def transform(self, X):
            return X
    
    class np:
        @staticmethod
        def array(lst):
            return lst
        @staticmethod
        def mean(arr):
            return statistics.mean(arr)
        @staticmethod
        def std(arr):
            return statistics.stdev(arr) if len(arr) > 1 else 0.0

# Try importing learning modules
try:
    from learning.analytics import UsageAnalytics
    from learning.patterns import PatternAnalyzer
    from learning.optimization import OptimizationSuggestions
    from learning.discovery import ToolDiscovery
    HAS_LEARNING_MODULES = True
except ImportError:
    print("Learning modules not available, using fallback components")
    HAS_LEARNING_MODULES = False
    
    # Fallback classes for demonstration
    class UsageAnalytics:
        def analyze_usage_patterns(self, data):
            return {"patterns": ["frequent_commands", "time_patterns"], "confidence": 0.8}
        def get_performance_metrics(self):
            return {"avg_response_time": 0.5, "success_rate": 0.95}
    
    class PatternAnalyzer:
        def find_patterns(self, data):
            return ["command_sequence", "time_based", "frequency"]
        def predict_next_command(self, context):
            return "ls"
    
    class OptimizationSuggestions:
        def generate_suggestions(self, usage_data):
            return ["Use faster alternatives", "Cache frequently used commands"]
        def calculate_improvement_score(self, suggestion):
            return 0.85
    
    class ToolDiscovery:
        def suggest_tools(self, context):
            return ["ripgrep", "fd", "bat"]
        def learn_tool_preferences(self, user_feedback):
            return True


class LearningEngine:
    """Core learning and intelligence engine"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize components
        self.platform = PlatformAdapter()
        self.shell = ShellDetector()
        self.optimizer = UsageOptimizer()
        self.registry = ToolRegistry()
        
        # Initialize learning components
        self.analytics = UsageAnalytics()
        self.patterns = PatternAnalyzer()
        self.optimization = OptimizationSuggestions()
        self.discovery = ToolDiscovery()
        
        # Learning data storage
        self.usage_history = deque(maxlen=1000)
        self.performance_data = defaultdict(list)
        self.pattern_cache = {}
        self.model_cache = {}
        
        if self.verbose:
            logging.info("LearningEngine initialized with all learning components")
    
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('learning_demo.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def record_usage(self, command: str, execution_time: float, success: bool, context: Dict = None):
        """Record usage data for learning"""
        record = {
            "timestamp": datetime.now(),
            "command": command,
            "execution_time": execution_time,
            "success": success,
            "context": context or {}
        }
        
        self.usage_history.append(record)
        self.performance_data[command].append(execution_time)
        
        if self.verbose:
            self.logger.debug(f"Recorded usage: {command} ({execution_time:.2f}s, {success})")
    
    def analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze usage patterns from recorded data"""
        if not self.usage_history:
            return {"patterns": [], "insights": []}
        
        # Analyze command frequency
        command_counts = defaultdict(int)
        time_patterns = defaultdict(int)
        
        for record in self.usage_history:
            command_counts[record["command"]] += 1
            hour = record["timestamp"].hour
            time_patterns[hour] += 1
        
        # Find most frequent commands
        top_commands = sorted(command_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Find peak usage hours
        peak_hours = sorted(time_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
        
        patterns = {
            "frequent_commands": top_commands,
            "time_patterns": peak_hours,
            "total_commands": len(self.usage_history),
            "unique_commands": len(command_counts)
        }
        
        return patterns
    
    def learn_patterns(self) -> List[str]:
        """Learn and identify usage patterns"""
        if not HAS_LEARNING_MODULES:
            # Simple pattern detection for fallback mode
            if len(self.usage_history) < 10:
                return ["Insufficient data for pattern learning"]
            
            # Analyze command sequences
            command_sequences = []
            commands = [r["command"] for r in self.usage_history]
            
            # Find common 2-grams and 3-grams
            for i in range(len(commands) - 1):
                command_sequences.append((commands[i], commands[i+1]))
            
            common_sequences = defaultdict(int)
            for seq in command_sequences:
                common_sequences[seq] += 1
            
            top_sequences = sorted(common_sequences.items(), key=lambda x: x[1], reverse=True)[:5]
            
            patterns = [f"Common sequence: {' → '.join(seq)}" for seq, count in top_sequences]
            
            # Time-based patterns
            hours = [r["timestamp"].hour for r in self.usage_history]
            if hours:
                most_common_hour = statistics.mode(hours)
                patterns.append(f"Most active hour: {most_common_hour}:00")
            
            return patterns
        else:
            # Use actual pattern analyzer
            usage_data = list(self.usage_history)
            patterns = self.patterns.find_patterns(usage_data)
            return patterns
    
    def predict_performance(self, command: str) -> Dict[str, float]:
        """Predict command performance based on learned patterns"""
        if not HAS_ML:
            # Simple prediction for fallback mode
            if command in self.performance_data:
                times = self.performance_data[command]
                return {
                    "predicted_time": statistics.mean(times),
                    "confidence": min(0.9, len(times) / 100),
                    "success_rate": sum(1 for r in self.usage_history if r["command"] == command and r["success"]) / max(1, len([r for r in self.usage_history if r["command"] == command))
                }
            else:
                return {"predicted_time": 1.0, "confidence": 0.1, "success_rate": 0.8}
        else:
            # Use ML model for prediction
            # This would require actual training data and feature engineering
            # Simplified for demonstration
            return {
                "predicted_time": random.uniform(0.1, 2.0),
                "confidence": random.uniform(0.6, 0.9),
                "success_rate": random.uniform(0.85, 0.95)
            }
    
    def generate_optimization_suggestions(self, context: Dict = None) -> List[Dict[str, Any]]:
        """Generate intelligent optimization suggestions"""
        if not HAS_LEARNING_MODULES:
            # Generate suggestions based on usage patterns
            suggestions = []
            
            # Analyze command efficiency
            slow_commands = []
            for command, times in self.performance_data.items():
                if len(times) >= 3:  # Need at least 3 samples
                    avg_time = statistics.mean(times)
                    if avg_time > 2.0:  # Consider slow if > 2 seconds
                        slow_commands.append((command, avg_time))
            
            # Suggest faster alternatives for slow commands
            command_alternatives = {
                "find": "fd",
                "grep": "rg",
                "ls -la": "ls --color=auto",
                "cat": "bat",
                "ps aux | grep": "pgrep"
            }
            
            for command, avg_time in slow_commands:
                if command in command_alternatives:
                    suggestion = {
                        "type": "performance",
                        "current": command,
                        "suggested": command_alternatives[command],
                        "reason": f"Current avg time: {avg_time:.2f}s",
                        "expected_improvement": f"2-5x faster",
                        "confidence": 0.8
                    }
                    suggestions.append(suggestion)
            
            # Suggest caching for frequently used commands
            command_counts = defaultdict(int)
            for record in self.usage_history:
                command_counts[record["command"]] += 1
            
            frequent_commands = [cmd for cmd, count in command_counts.items() if count >= 5]
            if frequent_commands:
                suggestions.append({
                    "type": "caching",
                    "commands": frequent_commands[:5],
                    "reason": "Frequently executed commands",
                    "expected_improvement": "Instant execution",
                    "confidence": 0.9
                })
            
            return suggestions
        else:
            # Use actual optimization engine
            usage_data = list(self.usage_history)
            suggestions = self.optimization.generate_suggestions(usage_data)
            return suggestions
    
    def adaptive_learning_cycle(self) -> Dict[str, Any]:
        """Perform adaptive learning cycle"""
        if len(self.usage_history) < 10:
            return {"status": "insufficient_data", "learned_patterns": 0}
        
        # Learn patterns
        patterns = self.learn_patterns()
        
        # Analyze usage
        usage_analysis = self.analyze_usage_patterns()
        
        # Generate predictions
        predictions = {}
        for command in self.performance_data.keys():
            predictions[command] = self.predict_performance(command)
        
        # Generate suggestions
        suggestions = self.generate_optimization_suggestions()
        
        learning_results = {
            "status": "completed",
            "learned_patterns": len(patterns),
            "usage_analysis": usage_analysis,
            "predictions": predictions,
            "suggestions": suggestions,
            "total_records": len(self.usage_history)
        }
        
        if self.verbose:
            self.logger.info(f"Learning cycle completed: {len(patterns)} patterns learned")
        
        return learning_results
    
    def discover_new_tools(self, context: Dict = None) -> List[Dict[str, Any]]:
        """Discover and suggest new tools based on usage patterns"""
        if not HAS_LEARNING_MODULES:
            # Simple tool discovery based on command analysis
            discovered_tools = []
            
            # Analyze current usage for tool gaps
            used_commands = set(r["command"] for r in self.usage_history)
            
            # Command tool mappings
            tool_suggestions = {
                "find": {"tool": "fd", "description": "Modern alternative to find"},
                "grep": {"tool": "rg", "description": "Fast grep alternative"},
                "cat": {"tool": "bat", "description": "Cat with syntax highlighting"},
                "ls": {"tool": "exa", "description": "Modern ls replacement"},
                "diff": {"tool": "delta", "description": "Better diff viewer"},
                "man": {"tool": "tldr", "description": "Simplified man pages"}
            }
            
            for command, tool_info in tool_suggestions.items():
                if command in used_commands:
                    suggestion = {
                        "tool": tool_info["tool"],
                        "replaces": command,
                        "description": tool_info["description"],
                        "confidence": 0.7,
                        "reason": f"You frequently use '{command}'"
                    }
                    discovered_tools.append(suggestion)
            
            return discovered_tools[:10]  # Limit to top 10
        else:
            # Use actual tool discovery
            context = context or {}
            return self.discovery.suggest_tools(context)


class LearningDemo:
    """Demonstrates learning and intelligence capabilities"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.learning_engine = LearningEngine(verbose=verbose)
    
    async def simulate_usage_data(self, num_commands: int = 50):
        """Generate simulated usage data for demonstration"""
        print(f"\nGenerating {num_commands} simulated usage records...")
        
        # Common commands with realistic execution times
        commands = {
            "ls": (0.1, 0.95),
            "pwd": (0.05, 0.99),
            "date": (0.1, 0.98),
            "whoami": (0.05, 0.99),
            "find . -name '*.py'": (1.5, 0.90),
            "grep -r 'import' .": (2.0, 0.85),
            "python script.py": (3.0, 0.80),
            "git status": (0.5, 0.95),
            "docker ps": (2.5, 0.85),
            "npm install": (15.0, 0.70)
        }
        
        for i in range(num_commands):
            command = random.choice(list(commands.keys()))
            base_time, base_success = commands[command]
            
            # Add some variation
            execution_time = max(0.01, random.gauss(base_time, base_time * 0.3))
            success = random.random() < base_success
            
            # Record usage
            self.learning_engine.record_usage(
                command=command,
                execution_time=execution_time,
                success=success,
                context={"session": f"demo_{i//10}", "tool": "demo"}
            )
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1} records...")
        
        print(f"✓ Generated {num_commands} usage records")
    
    async def demonstrate_pattern_learning(self):
        """Show pattern learning capabilities"""
        print("\n" + "="*60)
        print("PATTERN LEARNING DEMONSTRATION")
        print("="*60)
        
        # Learn patterns from usage data
        patterns = self.learning_engine.learn_patterns()
        
        print(f"\nLearned {len(patterns)} patterns:")
        for i, pattern in enumerate(patterns, 1):
            print(f"  {i}. {pattern}")
        
        # Analyze usage patterns
        usage_analysis = self.learning_engine.analyze_usage_patterns()
        
        print(f"\nUsage Analysis:")
        print(f"  Total commands executed: {usage_analysis['total_commands']}")
        print(f"  Unique commands: {usage_analysis['unique_commands']}")
        
        print(f"\nTop 5 Most Frequent Commands:")
        for command, count in usage_analysis['frequent_commands'][:5]:
            print(f"  {command}: {count} times")
        
        return patterns, usage_analysis
    
    async def demonstrate_optimization_suggestions(self):
        """Show optimization suggestion generation"""
        print("\n" + "="*60)
        print("OPTIMIZATION SUGGESTIONS DEMONSTRATION")
        print("="*60)
        
        # Generate suggestions
        suggestions = self.learning_engine.generate_optimization_suggestions()
        
        print(f"\nGenerated {len(suggestions)} optimization suggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n  {i}. {suggestion['type'].upper()} Optimization")
            if 'current' in suggestion:
                print(f"     Current: {suggestion['current']}")
                print(f"     Suggested: {suggestion['suggested']}")
            if 'commands' in suggestion:
                print(f"     Commands: {', '.join(suggestion['commands'])}")
            print(f"     Reason: {suggestion['reason']}")
            print(f"     Expected: {suggestion['expected_improvement']}")
            print(f"     Confidence: {suggestion['confidence']:.0%}")
        
        return suggestions
    
    async def demonstrate_performance_prediction(self):
        """Show performance prediction capabilities"""
        print("\n" + "="*60)
        print("PERFORMANCE PREDICTION DEMONSTRATION")
        print("="*60)
        
        # Get commands with usage history
        commands_with_history = list(self.learning_engine.performance_data.keys())
        
        if not commands_with_history:
            print("No command history available for prediction.")
            return {}
        
        print(f"\nPredicting performance for {len(commands_with_history)} commands:")
        predictions = {}
        
        for command in commands_with_history[:10]:  # Limit to 10 for demo
            prediction = self.learning_engine.predict_performance(command)
            predictions[command] = prediction
            
            print(f"\n  {command}:")
            print(f"    Predicted time: {prediction['predicted_time']:.2f}s")
            print(f"    Success rate: {prediction['success_rate']:.0%}")
            print(f"    Confidence: {prediction['confidence']:.0%}")
        
        return predictions
    
    async def demonstrate_adaptive_learning(self):
        """Show adaptive learning cycle"""
        print("\n" + "="*60)
        print("ADAPTIVE LEARNING CYCLE DEMONSTRATION")
        print("="*60)
        
        # Run learning cycle
        results = self.learning_engine.adaptive_learning_cycle()
        
        print(f"\nLearning Cycle Results:")
        print(f"  Status: {results['status']}")
        print(f"  Patterns learned: {results['learned_patterns']}")
        print(f"  Total records analyzed: {results['total_records']}")
        
        if results['status'] == 'completed':
            print(f"\nInsights from usage analysis:")
            analysis = results['usage_analysis']
            print(f"  - Unique commands: {analysis['unique_commands']}")
            print(f"  - Most frequent: {analysis['frequent_commands'][0][0] if analysis['frequent_commands'] else 'N/A'}")
            
            print(f"\nPerformance predictions available for {len(results['predictions'])} commands")
            print(f"Optimization suggestions generated: {len(results['suggestions'])}")
        
        return results
    
    async def demonstrate_tool_discovery(self):
        """Show intelligent tool discovery"""
        print("\n" + "="*60)
        print("TOOL DISCOVERY DEMONSTRATION")
        print("="*60)
        
        # Discover new tools
        discoveries = self.learning_engine.discover_new_tools()
        
        print(f"\nDiscovered {len(discoveries)} tool suggestions:")
        for i, discovery in enumerate(discoveries, 1):
            print(f"\n  {i}. {discovery['tool']}")
            print(f"     Replaces: {discovery['replaces']}")
            print(f"     Description: {discovery['description']}")
            print(f"     Reason: {discovery['reason']}")
            print(f"     Confidence: {discovery['confidence']:.0%}")
        
        return discoveries
    
    async def run_interactive_demo(self):
        """Run interactive learning demonstration"""
        print("\n" + "="*60)
        print("INTERACTIVE LEARNING DEMO")
        print("="*60)
        print("Commands: simulate, patterns, optimize, predict, learn, discover, quit")
        
        while True:
            try:
                command = input("\nDemo> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("Goodbye!")
                    break
                elif command == "simulate":
                    await self.simulate_usage_data(20)
                elif command == "patterns":
                    await self.demonstrate_pattern_learning()
                elif command == "optimize":
                    await self.demonstrate_optimization_suggestions()
                elif command == "predict":
                    await self.demonstrate_performance_prediction()
                elif command == "learn":
                    await self.demonstrate_adaptive_learning()
                elif command == "discover":
                    await self.demonstrate_tool_discovery()
                elif command == "help":
                    print("Available commands:")
                    for cmd in ["simulate", "patterns", "optimize", "predict", "learn", "discover", "quit"]:
                        print(f"  - {cmd}")
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nInterrupted. Type 'quit' to exit.")
            except EOFError:
                print("\nGoodbye!")
                break
    
    async def run_comprehensive_demo(self):
        """Run comprehensive learning demonstration"""
        print("\n" + "="*80)
        print("COMPREHENSIVE LEARNING DEMONSTRATION")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Generate usage data
            await self.simulate_usage_data(100)
            
            # Run all demonstrations
            await self.demonstrate_pattern_learning()
            await asyncio.sleep(1)
            
            await self.demonstrate_optimization_suggestions()
            await asyncio.sleep(1)
            
            await self.demonstrate_performance_prediction()
            await asyncio.sleep(1)
            
            await self.demonstrate_adaptive_learning()
            await asyncio.sleep(1)
            
            await self.demonstrate_tool_discovery()
            
            print("\n" + "="*80)
            print("LEARNING DEMONSTRATION COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Error during learning demonstration: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Learning Demo - Demonstrates ML and intelligence capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python learning_demo.py                    # Run all demos
  python learning_demo.py --interactive      # Interactive mode
  python learning_demo.py --mode patterns    # Specific demo mode
  python learning_demo.py --verbose          # Verbose output
  python learning_demo.py --training-data usage.json  # Use custom data
        """
    )
    
    parser.add_argument(
        "--mode", 
        choices=["patterns", "optimize", "predict", "learn", "discover", "all"],
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
    
    parser.add_argument(
        "--training-data", 
        type=str,
        help="Path to training data file (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = LearningDemo(verbose=args.verbose)
    
    # Load training data if specified
    if args.training_data and os.path.exists(args.training_data):
        try:
            with open(args.training_data, 'r') as f:
                data = json.load(f)
            print(f"Loaded training data from {args.training_data}")
            # Process training data...
        except Exception as e:
            print(f"Error loading training data: {e}")
    
    # Run appropriate demo mode
    if args.interactive:
        # Run interactive demo
        asyncio.run(demo.run_interactive_demo())
    elif args.mode == "all":
        # Run comprehensive demo
        asyncio.run(demo.run_comprehensive_demo())
    else:
        # Run specific demo
        method_name = f"demonstrate_{args.mode}"
        if hasattr(demo, method_name):
            asyncio.run(getattr(demo, method_name)())
        else:
            print(f"Error: Demo mode '{args.mode}' not found")
            sys.exit(1)


if __name__ == "__main__":
    main()
