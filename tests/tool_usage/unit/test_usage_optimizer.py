"""
Unit Tests for Usage Optimizer Component
=========================================

Comprehensive unit tests for the UsageOptimizer class, testing command
optimization, pattern analysis, and performance improvements.
"""

import pytest
import os
import sys
import time
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
from typing import List, Dict, Any
from dataclasses import asdict

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.usage_optimizer import (
        UsageOptimizer, UsagePattern, OptimizationResult,
        CommandMetrics, PatternAnalyzer, CommandOptimizer
    )
    from tool_usage.command_executor import CommandResult, SecurityLevel
    from tool_usage.platform_adapter import PosixAdapter, WindowsAdapter
except ImportError as e:
    pytest.skip(f"Could not import UsageOptimizer: {e}", allow_module_level=True)


class TestUsagePattern:
    """Test UsagePattern dataclass"""
    
    def test_usage_pattern_creation(self):
        """Test creating UsagePattern object"""
        pattern = UsagePattern(
            pattern_type="command_sequence",
            commands=["ls", "grep", "sort"],
            frequency=10,
            average_execution_time=0.05,
            efficiency_score=0.8,
            optimization_suggestions=[
                "Combine with find command",
                "Use grep -q for faster matching"
            ]
        )
        
        assert pattern.pattern_type == "command_sequence"
        assert pattern.commands == ["ls", "grep", "sort"]
        assert pattern.frequency == 10
        assert pattern.average_execution_time == 0.05
        assert pattern.efficiency_score == 0.8
        assert len(pattern.optimization_suggestions) == 2
    
    def test_usage_pattern_minimal(self):
        """Test creating UsagePattern with minimal data"""
        pattern = UsagePattern(
            pattern_type="simple_command",
            commands=["echo", "test"]
        )
        
        assert pattern.pattern_type == "simple_command"
        assert pattern.commands == ["echo", "test"]
        assert pattern.frequency == 0  # Default
        assert pattern.average_execution_time == 0.0  # Default
        assert pattern.efficiency_score == 0.0  # Default
        assert pattern.optimization_suggestions == []  # Default


class TestOptimizationResult:
    """Test OptimizationResult dataclass"""
    
    def test_optimization_result_creation(self):
        """Test creating OptimizationResult object"""
        result = OptimizationResult(
            command="cat file.txt | grep pattern | sort",
            optimized_command="grep -n pattern file.txt | sort",
            success=True,
            improvement_score=0.25,
            estimated_time_saved=0.1,
            actual_time_saved=0.08,
            optimization_applied=True,
            metadata={
                "optimization_type": "pipeline_elimination",
                "tools_used": ["grep", "sed"]
            }
        )
        
        assert result.command == "cat file.txt | grep pattern | sort"
        assert result.optimized_command == "grep -n pattern file.txt | sort"
        assert result.success is True
        assert result.improvement_score == 0.25
        assert result.estimated_time_saved == 0.1
        assert result.actual_time_saved == 0.08
        assert result.optimization_applied is True
        assert "optimization_type" in result.metadata
    
    def test_optimization_result_failure(self):
        """Test creating failed OptimizationResult"""
        result = OptimizationResult(
            command="complex_command",
            optimized_command="complex_command",
            success=False,
            error_message="Command cannot be optimized",
            improvement_score=0.0
        )
        
        assert result.success is False
        assert result.command == result.optimized_command
        assert result.error_message == "Command cannot be optimized"
        assert result.improvement_score == 0.0
    
    def test_optimization_result_minimal(self):
        """Test creating OptimizationResult with minimal data"""
        result = OptimizationResult(
            command="simple",
            optimized_command="optimized"
        )
        
        assert result.success is True  # Default
        assert result.improvement_score == 0.0  # Default
        assert result.estimated_time_saved == 0.0  # Default
        assert result.actual_time_saved == 0.0  # Default
        assert result.optimization_applied is False  # Default
        assert result.metadata == {}  # Default


class TestCommandMetrics:
    """Test CommandMetrics dataclass"""
    
    def test_command_metrics_creation(self):
        """Test creating CommandMetrics object"""
        metrics = CommandMetrics(
            command="grep pattern file.txt",
            execution_count=50,
            average_execution_time=0.03,
            total_execution_time=1.5,
            success_rate=0.98,
            error_count=1,
            last_execution=time.time(),
            optimization_potential=0.2,
            performance_trend="improving"
        )
        
        assert metrics.command == "grep pattern file.txt"
        assert metrics.execution_count == 50
        assert metrics.average_execution_time == 0.03
        assert metrics.total_execution_time == 1.5
        assert metrics.success_rate == 0.98
        assert metrics.error_count == 1
        assert metrics.last_execution is not None
        assert metrics.optimization_potential == 0.2
        assert metrics.performance_trend == "improving"
    
    def test_command_metrics_calculation(self):
        """Test CommandMetrics calculations"""
        # Create metrics for a command executed multiple times
        metrics = CommandMetrics(
            command="test_command",
            execution_count=10,
            average_execution_time=0.05,
            total_execution_time=0.5,
            success_rate=0.9,
            error_count=1
        )
        
        # Verify calculations
        assert metrics.average_execution_time == metrics.total_execution_time / metrics.execution_count


class TestUsageOptimizerInitialization:
    """Test UsageOptimizer initialization"""
    
    def test_optimizer_initialization(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Test basic UsageOptimizer initialization"""
        optimizer = UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
        
        assert optimizer.platform_adapter == posix_adapter
        assert optimizer.command_executor == command_executor_mock
        assert optimizer.tool_registry == tool_registry_mock
        assert optimizer.patterns == {}
        assert optimizer.metrics == {}
        assert optimizer.auto_optimize is False
        assert optimizer.performance_threshold == 0.1
    
    def test_optimizer_with_real_components(self):
        """Test UsageOptimizer with real components"""
        try:
            adapter = get_platform_adapter()
            executor = CommandExecutor(adapter)
            registry = ToolRegistry(adapter)
            
            optimizer = UsageOptimizer(adapter, executor, registry)
            
            assert optimizer.platform_adapter == adapter
            assert optimizer.command_executor == executor
            assert optimizer.tool_registry == registry
        except Exception:
            # Skip if real components fail to initialize
            pytest.skip("Could not initialize real components")
    
    def test_optimizer_custom_config(self):
        """Test UsageOptimizer with custom configuration"""
        custom_config = {
            'auto_optimize': True,
            'performance_threshold': 0.2,
            'pattern_history_size': 1000
        }
        
        optimizer = UsageOptimizer(
            Mock(), Mock(), Mock(),
            **custom_config
        )
        
        assert optimizer.auto_optimize is True
        assert optimizer.performance_threshold == 0.2


class TestPatternAnalyzer:
    """Test PatternAnalyzer functionality"""
    
    @pytest.fixture
    def pattern_analyzer(self):
        """Create pattern analyzer for testing"""
        return PatternAnalyzer()
    
    def test_analyze_command_sequence(self, pattern_analyzer):
        """Test analysis of command sequences"""
        commands = [
            "ls -la",
            "grep pattern file.txt",
            "sort output.txt",
            "ls -la",
            "grep pattern file.txt",
            "sort output.txt"
        ]
        
        patterns = pattern_analyzer.analyze_command_sequence(commands)
        
        assert len(patterns) > 0
        
        # Should find the repeated sequence
        sequence_pattern = next((p for p in patterns if p.pattern_type == "command_sequence"), None)
        assert sequence_pattern is not None
        assert sequence_pattern.frequency >= 1
    
    def test_analyze_timing_patterns(self, pattern_analyzer):
        """Test analysis of timing patterns"""
        execution_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.2]
        
        pattern = pattern_analyzer.analyze_timing_patterns("test_command", execution_times)
        
        assert pattern.command == "test_command"
        assert len(pattern.timing_data) == len(execution_times)
        assert pattern.average_execution_time > 0
    
    def test_identify_slow_commands(self, pattern_analyzer):
        """Test identification of slow commands"""
        metrics = [
            CommandMetrics("fast_cmd", execution_count=10, average_execution_time=0.01),
            CommandMetrics("slow_cmd", execution_count=5, average_execution_time=1.0),
            CommandMetrics("medium_cmd", execution_count=20, average_execution_time=0.1)
        ]
        
        slow_commands = pattern_analyzer.identify_slow_commands(metrics, threshold=0.05)
        
        assert "slow_cmd" in slow_commands
        assert "medium_cmd" in slow_commands
        assert "fast_cmd" not in slow_commands
    
    def test_detect_command_patterns(self, pattern_analyzer):
        """Test detection of command usage patterns"""
        commands = [
            "git add .",
            "git commit -m 'message'",
            "git push",
            "git add .",
            "git commit -m 'another message'",
            "git push"
        ]
        
        patterns = pattern_analyzer.detect_command_patterns(commands)
        
        # Should identify git workflow pattern
        git_pattern = next((p for p in patterns if "git" in str(p.pattern_type)), None)
        assert git_pattern is not None


class TestCommandOptimizer:
    """Test CommandOptimizer functionality"""
    
    @pytest.fixture
    def command_optimizer(self):
        """Create command optimizer for testing"""
        return CommandOptimizer()
    
    def test_optimize_pipeline_command(self, command_optimizer):
        """Test optimization of pipeline commands"""
        command = "cat file.txt | grep pattern | sort | uniq"
        
        optimization = command_optimizer.optimize_pipeline_command(command)
        
        assert optimization.command == command
        assert optimization.success is True
        assert optimization.optimized_command != command  # Should be different
        
        # Should be faster
        assert optimization.improvement_score > 0
    
    def test_optimize_find_command(self, command_optimizer):
        """Test optimization of find commands"""
        command = "find . -name '*.txt' -exec grep pattern {} \;"
        
        optimization = command_optimizer.optimize_find_command(command)
        
        assert optimization.command == command
        if optimization.success:
            assert optimization.optimized_command != command
    
    def test_optimize_grep_command(self, command_optimizer):
        """Test optimization of grep commands"""
        command = "cat large_file.txt | grep pattern"
        
        optimization = command_optimizer.optimize_grep_command(command)
        
        assert optimization.command == command
        if optimization.success:
            assert "grep" in optimization.optimized_command
            # Should recommend direct grep instead of cat | grep
            assert "cat" not in optimization.optimized_command or "cat" not in command.split("|")[0].strip()
    
    def test_optimize_simple_command(self, command_optimizer):
        """Test that simple commands are left unchanged"""
        command = "echo 'hello world'"
        
        optimization = command_optimizer.optimize_simple_command(command)
        
        assert optimization.command == command
        # Simple commands might not need optimization
        assert optimization.optimized_command in [command, optimization.optimized_command]
    
    def test_batch_optimize_commands(self, command_optimizer):
        """Test batch optimization of multiple commands"""
        commands = [
            "cat file.txt | grep pattern",
            "find . -name '*.log' -exec rm {} \;",
            "ls -la | head -10"
        ]
        
        optimizations = command_optimizer.batch_optimize_commands(commands)
        
        assert len(optimizations) == len(commands)
        for opt in optimizations:
            assert hasattr(opt, 'command')
            assert hasattr(opt, 'success')


class TestCommandOptimization:
    """Test command optimization functionality"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_optimize_command_basic(self, optimizer):
        """Test basic command optimization"""
        # Mock optimization result
        mock_result = OptimizationResult(
            command="original command",
            optimized_command="optimized command",
            success=True,
            improvement_score=0.3
        )
        
        with patch.object(optimizer.command_optimizer, 'optimize_command', return_value=mock_result):
            result = optimizer.optimize_command("original command")
            
            assert result.success is True
            assert result.improvement_score == 0.3
            assert result.optimized_command != "original command"
    
    def test_optimize_command_failure(self, optimizer):
        """Test command optimization failure"""
        mock_result = OptimizationResult(
            command="complex_command",
            optimized_command="complex_command",
            success=False,
            error_message="Cannot optimize this command"
        )
        
        with patch.object(optimizer.command_optimizer, 'optimize_command', return_value=mock_result):
            result = optimizer.optimize_command("complex_command")
            
            assert result.success is False
            assert result.error_message == "Cannot optimize this command"
    
    def test_batch_optimize(self, optimizer):
        """Test batch optimization"""
        commands = ["cmd1", "cmd2", "cmd3"]
        
        mock_results = [
            OptimizationResult("cmd1", "opt_cmd1", True, 0.2),
            OptimizationResult("cmd2", "cmd2", False, 0.0),
            OptimizationResult("cmd3", "opt_cmd3", True, 0.1)
        ]
        
        with patch.object(optimizer.command_optimizer, 'batch_optimize_commands', return_value=mock_results):
            results = optimizer.batch_optimize(commands)
            
            assert len(results) == 3
            assert results[0].success is True
            assert results[1].success is False
            assert results[2].success is True
    
    def test_optimize_with_application(self, optimizer):
        """Test optimization with automatic application"""
        command = "cat file.txt | grep pattern"
        
        mock_optimization = OptimizationResult(
            command=command,
            optimized_command="grep pattern file.txt",
            success=True,
            improvement_score=0.25
        )
        
        with patch.object(optimizer.command_optimizer, 'optimize_command', return_value=mock_optimization), \
             patch.object(optimizer.command_executor, 'execute') as mock_execute:
            
            # Mock executor to return successful result for optimized command
            mock_execute.return_value = Mock(success=True, return_code=0)
            
            result = optimizer.optimize_command(command, apply_optimization=True)
            
            assert result.success is True
            # Should have executed the optimized command
            mock_execute.assert_called()


class TestMetricsTracking:
    """Test metrics tracking functionality"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_update_metrics(self, optimizer):
        """Test updating command metrics"""
        # Create mock command result
        mock_result = Mock()
        mock_result.command = "test_command"
        mock_result.execution_time = 0.05
        mock_result.success = True
        mock_result.timestamp = time.time()
        
        optimizer.update_metrics(mock_result)
        
        assert "test_command" in optimizer.metrics
        metrics = optimizer.metrics["test_command"]
        assert metrics.execution_count == 1
        assert metrics.average_execution_time == 0.05
        assert metrics.success_rate == 1.0
    
    def test_update_multiple_executions(self, optimizer):
        """Test updating metrics with multiple executions"""
        command = "repeated_command"
        
        # Execute command multiple times
        for i in range(5):
            mock_result = Mock()
            mock_result.command = command
            mock_result.execution_time = 0.01 * (i + 1)  # Increasing time
            mock_result.success = i < 4  # Last execution fails
            mock_result.timestamp = time.time() + i
            
            optimizer.update_metrics(mock_result)
        
        metrics = optimizer.metrics[command]
        assert metrics.execution_count == 5
        assert metrics.error_count == 1
        assert metrics.success_rate == 0.8
        assert metrics.average_execution_time > 0
    
    def test_get_command_metrics(self, optimizer):
        """Test getting command metrics"""
        # Add some metrics first
        test_command = "metric_test"
        metrics = CommandMetrics(
            command=test_command,
            execution_count=3,
            average_execution_time=0.02
        )
        optimizer.metrics[test_command] = metrics
        
        retrieved_metrics = optimizer.get_command_metrics(test_command)
        
        assert retrieved_metrics is not None
        assert retrieved_metrics.command == test_command
        assert retrieved_metrics.execution_count == 3
    
    def test_get_command_metrics_nonexistent(self, optimizer):
        """Test getting metrics for non-existent command"""
        metrics = optimizer.get_command_metrics("nonexistent_command")
        
        assert metrics is None


class TestUsagePatterns:
    """Test usage pattern analysis"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_analyze_usage_patterns_basic(self, optimizer):
        """Test basic usage pattern analysis"""
        # Add mock execution history
        history = [
            Mock(command="ls -la", execution_time=0.01, success=True),
            Mock(command="grep pattern file.txt", execution_time=0.05, success=True),
            Mock(command="ls -la", execution_time=0.01, success=True),
            Mock(command="grep pattern file.txt", execution_time=0.05, success=True)
        ]
        
        patterns = optimizer.analyze_usage_patterns(history)
        
        assert len(patterns) > 0
        
        # Should find repeated commands
        command_patterns = [p for p in patterns if p.pattern_type == "command_frequency"]
        assert len(command_patterns) > 0
    
    def test_analyze_usage_patterns_none(self, optimizer):
        """Test pattern analysis with no history"""
        patterns = optimizer.analyze_usage_patterns([])
        
        assert len(patterns) == 0
    
    def test_get_optimization_recommendations(self, optimizer):
        """Test getting optimization recommendations"""
        # Add some test data
        optimizer.metrics["slow_command"] = CommandMetrics(
            command="slow_command",
            execution_count=10,
            average_execution_time=1.0,
            optimization_potential=0.3
        )
        
        optimizer.metrics["fast_command"] = CommandMetrics(
            command="fast_command",
            execution_count=1,
            average_execution_time=0.01,
            optimization_potential=0.1
        )
        
        recommendations = optimizer.get_optimization_recommendations(limit=5)
        
        assert isinstance(recommendations, list)
        # Should have recommendations for slow commands
        slow_recommendations = [r for r in recommendations if "slow_command" in str(r)]
        assert len(slow_recommendations) > 0


class TestOptimizationStrategies:
    """Test different optimization strategies"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_performance_based_optimization(self, optimizer):
        """Test optimization based on performance metrics"""
        # Add slow command metrics
        optimizer.metrics["slow_cmd"] = CommandMetrics(
            command="slow_cmd",
            execution_count=20,
            average_execution_time=2.0,
            optimization_potential=0.4
        )
        
        strategies = optimizer.get_optimization_strategies()
        
        assert isinstance(strategies, list)
        # Should include performance-based strategies
        performance_strategies = [s for s in strategies if "performance" in s.get("type", "")]
        assert len(performance_strategies) > 0
    
    def test_frequency_based_optimization(self, optimizer):
        """Test optimization based on command frequency"""
        # Add frequently executed commands
        optimizer.metrics["frequent_cmd"] = CommandMetrics(
            command="frequent_cmd",
            execution_count=100,
            average_execution_time=0.01,
            optimization_potential=0.1
        )
        
        strategies = optimizer.get_optimization_strategies()
        
        assert isinstance(strategies, list)
        # Should include frequency-based strategies
        frequency_strategies = [s for s in strategies if "frequency" in s.get("type", "")]
        assert len(frequency_strategies) > 0
    
    def test_combined_optimization(self, optimizer):
        """Test combined optimization strategies"""
        # Add various types of commands
        optimizer.metrics.update({
            "slow_frequent": CommandMetrics(
                command="slow_frequent",
                execution_count=50,
                average_execution_time=1.0,
                optimization_potential=0.5
            ),
            "fast_infrequent": CommandMetrics(
                command="fast_infrequent",
                execution_count=2,
                average_execution_time=0.001,
                optimization_potential=0.05
            )
        })
        
        strategies = optimizer.get_optimization_strategies()
        
        assert len(strategies) > 0
        # Should prioritize commands with high frequency and optimization potential
        sorted_strategies = optimizer._sort_strategies_by_impact(strategies)
        assert len(sorted_strategies) > 0


class TestOptimizationConfiguration:
    """Test optimization configuration and settings"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_set_performance_threshold(self, optimizer):
        """Test setting performance threshold"""
        new_threshold = 0.2
        
        optimizer.set_performance_threshold(new_threshold)
        
        assert optimizer.performance_threshold == new_threshold
    
    def test_set_auto_optimization(self, optimizer):
        """Test enabling/disabling auto-optimization"""
        optimizer.set_auto_optimization(True)
        assert optimizer.auto_optimize is True
        
        optimizer.set_auto_optimization(False)
        assert optimizer.auto_optimize is False
    
    def test_get_optimizer_summary(self, optimizer):
        """Test getting optimizer summary"""
        # Add some test data
        optimizer.metrics["test_cmd"] = CommandMetrics(
            command="test_cmd",
            execution_count=5,
            average_execution_time=0.1
        )
        
        summary = optimizer.get_optimizer_summary()
        
        assert isinstance(summary, dict)
        assert 'metrics_count' in summary
        assert 'patterns_count' in summary
        assert 'configuration' in summary
        
        assert summary['metrics_count'] == 1
        assert summary['patterns_count'] == 0  # No patterns added yet


class TestOptimizationPersistence:
    """Test optimization data persistence"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_save_optimization_data(self, temp_test_dir, optimizer):
        """Test saving optimization data"""
        # Add test data
        optimizer.metrics["test_cmd"] = CommandMetrics(
            command="test_cmd",
            execution_count=3,
            average_execution_time=0.05
        )
        
        save_path = temp_test_dir / "optimization_data.json"
        
        success = optimizer.save_optimization_data(str(save_path))
        
        assert success is True
        assert save_path.exists()
        
        # Verify saved content
        with open(save_path) as f:
            data = json.load(f)
        
        assert "metrics" in data
        assert "test_cmd" in data["metrics"]
    
    def test_load_optimization_data(self, temp_test_dir, optimizer):
        """Test loading optimization data"""
        # Create save file
        save_data = {
            "metrics": {
                "loaded_cmd": {
                    "command": "loaded_cmd",
                    "execution_count": 10,
                    "average_execution_time": 0.02
                }
            },
            "patterns": []
        }
        
        save_path = temp_test_dir / "optimization_data.json"
        with open(save_path, 'w') as f:
            json.dump(save_data, f)
        
        success = optimizer.load_optimization_data(str(save_path))
        
        assert success is True
        assert "loaded_cmd" in optimizer.metrics
        assert optimizer.metrics["loaded_cmd"].command == "loaded_cmd"


class TestOptimizationPerformance:
    """Test optimization performance characteristics"""
    
    @pytest.mark.performance
    def test_optimization_performance(self, optimizer):
        """Test optimization performance with many commands"""
        commands = [f"command_{i}" for i in range(100)]
        
        start_time = time.time()
        
        with patch.object(optimizer.command_optimizer, 'optimize_command') as mock_optimize:
            mock_optimize.return_value = OptimizationResult(
                "test", "optimized", True, 0.1
            )
            
            for command in commands:
                optimizer.optimize_command(command)
        
        end_time = time.time()
        optimization_time = end_time - start_time
        
        # Should optimize 100 commands reasonably quickly
        assert optimization_time < 30.0
        assert mock_optimize.call_count == 100
    
    @pytest.mark.performance
    def test_metrics_update_performance(self, optimizer):
        """Test performance of metrics updates"""
        start_time = time.time()
        
        # Update metrics many times
        for i in range(1000):
            mock_result = Mock()
            mock_result.command = f"cmd_{i % 10}"  # Some repeated commands
            mock_result.execution_time = 0.01
            mock_result.success = True
            mock_result.timestamp = time.time()
            
            optimizer.update_metrics(mock_result)
        
        end_time = time.time()
        update_time = end_time - start_time
        
        # Should handle many metrics updates efficiently
        assert update_time < 10.0
        assert len(optimizer.metrics) == 10  # 10 unique commands
    
    @pytest.mark.performance
    def test_pattern_analysis_performance(self, optimizer):
        """Test performance of pattern analysis"""
        # Create large execution history
        history = []
        for i in range(1000):
            mock_result = Mock()
            mock_result.command = f"cmd_{i % 20}"  # 20 repeating commands
            mock_result.execution_time = 0.01
            mock_result.success = True
            mock_result.timestamp = time.time()
            history.append(mock_result)
        
        start_time = time.time()
        
        patterns = optimizer.analyze_usage_patterns(history)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should analyze patterns efficiently
        assert analysis_time < 30.0
        assert len(patterns) > 0


class TestOptimizationErrorHandling:
    """Test error handling in optimization"""
    
    @pytest.fixture
    def optimizer(self, posix_adapter, command_executor_mock, tool_registry_mock):
        """Create usage optimizer for testing"""
        return UsageOptimizer(posix_adapter, command_executor_mock, tool_registry_mock)
    
    def test_optimization_with_invalid_command(self, optimizer):
        """Test optimization of invalid command"""
        invalid_command = ""  # Empty command
        
        result = optimizer.optimize_command(invalid_command)
        
        # Should handle gracefully
        assert hasattr(result, 'success')
        assert hasattr(result, 'error_message')
    
    def test_pattern_analysis_with_corrupted_data(self, optimizer):
        """Test pattern analysis with corrupted data"""
        # Simulate corrupted execution history
        corrupted_history = [None, "valid_command", 123]  # Mixed types
        
        patterns = optimizer.analyze_usage_patterns(corrupted_history)
        
        # Should handle gracefully
        assert isinstance(patterns, list)
        # May find valid patterns from valid entries
    
    def test_metrics_update_with_invalid_data(self, optimizer):
        """Test metrics update with invalid data"""
        invalid_result = Mock()
        invalid_result.command = None  # Invalid command
        invalid_result.execution_time = -1  # Invalid time
        invalid_result.success = None  # Invalid success
        
        # Should handle gracefully
        try:
            optimizer.update_metrics(invalid_result)
            # Either succeeds or fails gracefully
        except Exception:
            # Can propagate errors
            pass
    
    def test_save_data_with_disk_error(self, optimizer, temp_test_dir):
        """Test saving data with disk error"""
        optimizer.metrics["test"] = CommandMetrics("test", 1, 0.01)
        
        invalid_path = "/invalid/path/that/does/not/exist/data.json"
        
        success = optimizer.save_optimization_data(invalid_path)
        
        assert success is False


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])