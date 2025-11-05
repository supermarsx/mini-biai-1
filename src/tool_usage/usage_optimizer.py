"""
Usage Optimizer Module

Provides command pattern optimization and performance analysis
for the mini-biai-1 framework tool usage system.
"""

import os
import sys
import time
import statistics
import re
import json
from typing import Dict, List, Optional, Union, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib

from .platform_adapter import PlatformAdapter, get_platform_adapter
from .command_executor import CommandExecutor, CommandResult
from .tool_registry import ToolRegistry, ToolCategory


@dataclass
class UsagePattern:
    """Represents a usage pattern discovered from command history."""
    pattern_id: str
    command_template: str
    frequency: int
    avg_execution_time: float
    success_rate: float
    last_used: datetime
    platform_context: str
    shell_context: str
    parameters: Set[str]
    optimization_suggestions: List[str] = field(default_factory=list)
    performance_score: float = 0.0


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    original_command: str
    optimized_command: str
    performance_gain: float
    estimated_time_saved: float
    optimization_type: str
    confidence: float
    applied_changes: List[str]
    warnings: List[str] = field(default_factory=list)
    success: bool = True


@dataclass
class CommandMetrics:
    """Performance metrics for a command."""
    command: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    std_dev: float = 0.0
    last_execution: Optional[datetime] = None
    error_patterns: List[str] = field(default_factory=list)
    success_rate: float = 0.0
    efficiency_score: float = 0.0


class PatternAnalyzer:
    """Analyzes command usage patterns."""
    
    def __init__(self, min_frequency: int = 3, time_window_days: int = 30):
        """
        Initialize the pattern analyzer.
        
        Args:
            min_frequency: Minimum frequency to consider a pattern
            time_window_days: Days to look back for pattern analysis
        """
        self.min_frequency = min_frequency
        self.time_window_days = time_window_days
        self.pattern_cache: Dict[str, UsagePattern] = {}
    
    def analyze_execution_history(self, history: List[CommandResult]) -> List[UsagePattern]:
        """
        Analyze execution history to discover usage patterns.
        
        Args:
            history: List of command execution results
            
        Returns:
            List of discovered usage patterns
        """
        # Filter recent history
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.time_window_days)
        recent_history = [h for h in history if h.timestamp >= cutoff_date.timestamp()]
        
        if not recent_history:
            return []
        
        # Group commands by similarity
        command_groups = self._group_similar_commands(recent_history)
        
        patterns = []
        for group in command_groups:
            if len(group) >= self.min_frequency:
                pattern = self._create_pattern(group)
                if pattern:
                    patterns.append(pattern)
        
        # Cache patterns
        for pattern in patterns:
            self.pattern_cache[pattern.pattern_id] = pattern
        
        return patterns
    
    def _group_similar_commands(self, history: List[CommandResult]) -> List[List[CommandResult]]:
        """Group commands by similarity."""
        groups = defaultdict(list)
        
        for result in history:
            # Normalize command for grouping
            normalized = self._normalize_command(result.command)
            groups[normalized].append(result)
        
        return list(groups.values())
    
    def _normalize_command(self, command: str) -> str:
        """Normalize command for pattern detection."""
        # Remove variable parts (timestamps, IDs, etc.)
        normalized = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', command)
        normalized = re.sub(r'\d{2}:\d{2}:\d{2}', 'TIME', normalized)
        normalized = re.sub(r'\b[a-f0-9]{8,}\b', 'ID', normalized)  # Hashes/IDs
        normalized = re.sub(r'\b\d+\b', 'NUMBER', normalized)  # Numbers
        
        # Normalize paths
        normalized = re.sub(r'/[^/\s]+', '/PATH', normalized)
        normalized = re.sub(r'[A-Za-z]:\\[^\\\s]+', 'WINPATH', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _create_pattern(self, command_group: List[CommandResult]) -> Optional[UsagePattern]:
        """Create a usage pattern from a command group."""
        if not command_group:
            return None
        
        # Extract common template
        template = self._extract_common_template(command_group)
        
        # Calculate metrics
        execution_times = [r.execution_time for r in command_group if r.execution_time > 0]
        success_count = sum(1 for r in command_group if r.success)
        last_used = max(r.timestamp for r in command_group)
        
        avg_time = statistics.mean(execution_times) if execution_times else 0.0
        success_rate = success_count / len(command_group)
        
        # Extract parameters
        parameters = self._extract_parameters(command_group)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(command_group, template)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(
            len(command_group), avg_time, success_rate
        )
        
        return UsagePattern(
            pattern_id=self._generate_pattern_id(template),
            command_template=template,
            frequency=len(command_group),
            avg_execution_time=avg_time,
            success_rate=success_rate,
            last_used=datetime.fromtimestamp(last_used, timezone.utc),
            platform_context=self._get_platform_context(command_group),
            shell_context=self._get_shell_context(command_group),
            parameters=parameters,
            optimization_suggestions=suggestions,
            performance_score=performance_score
        )
    
    def _extract_common_template(self, command_group: List[CommandResult]) -> str:
        """Extract common template from similar commands."""
        # Use the most recent command as base template
        return max(command_group, key=lambda x: x.timestamp).command
    
    def _extract_parameters(self, command_group: List[CommandResult]) -> Set[str]:
        """Extract parameter patterns from commands."""
        parameters = set()
        
        for result in command_group:
            # Simple parameter extraction (could be enhanced)
            parts = result.command.split()
            for part in parts:
                if part.startswith('-') or part.startswith('--'):
                    parameters.add(part)
                elif '=' in part:
                    parameters.add(part.split('=')[0])
        
        return parameters
    
    def _generate_optimization_suggestions(self, command_group: List[CommandResult], 
                                         template: str) -> List[str]:
        """Generate optimization suggestions for a pattern."""
        suggestions = []
        
        # Check for inefficient patterns
        if 'find' in template and 'xargs' in template:
            suggestions.append("Consider using find's -exec instead of xargs for better performance")
        
        if template.count('grep') > 1:
            suggestions.append("Combine multiple grep commands using single grep with regex")
        
        if 'cat' in template and '|' in template and 'grep' in template:
            suggestions.append("Use grep directly on files instead of cat | grep")
        
        # Check for missing optimizations
        if 'sort' in template and 'uniq' in template:
            suggestions.append("Use sort -u instead of sort | unq for better performance")
        
        if 'tar' in template and 'gzip' in template:
            suggestions.append("Use tar -z for combined compression")
        
        # Platform-specific suggestions
        if self._is_windows_context(command_group):
            if 'dir' in template and '|' in template and 'findstr' in template:
                suggestions.append("Use dir /s /b for recursive search on Windows")
        else:
            if 'ls' in template and 'grep' in template:
                suggestions.append("Use ls with appropriate flags instead of ls | grep")
        
        return suggestions
    
    def _calculate_performance_score(self, frequency: int, avg_time: float, 
                                   success_rate: float) -> float:
        """Calculate performance score for a pattern."""
        # Higher frequency, lower time, higher success rate = better score
        frequency_score = min(frequency / 10.0, 1.0)  # Cap at 10+ uses
        time_score = max(0, 1.0 - (avg_time / 10.0))  # Normalize time
        success_score = success_rate
        
        return (frequency_score + time_score + success_score) / 3.0
    
    def _generate_pattern_id(self, template: str) -> str:
        """Generate unique ID for a pattern."""
        return hashlib.md5(template.encode()).hexdigest()[:16]
    
    def _get_platform_context(self, command_group: List[CommandResult]) -> str:
        """Get dominant platform context for a command group."""
        platforms = [r.working_directory for r in command_group if r.working_directory]
        return Counter(platforms).most_common(1)[0][0] if platforms else 'unknown'
    
    def _get_shell_context(self, command_group: List[CommandResult]) -> str:
        """Get dominant shell context for a command group."""
        shells = [r.shell_used for r in command_group if r.shell_used]
        return Counter(shells).most_common(1)[0][0] if shells else 'unknown'
    
    def _is_windows_context(self, command_group: List[CommandResult]) -> bool:
        """Check if command group is primarily Windows context."""
        windows_indicators = ['cmd.exe', 'powershell.exe', 'bat', 'com']
        for result in command_group:
            if result.shell_used and any(indicator in result.shell_used.lower() 
                                       for indicator in windows_indicators):
                return True
        return False


class CommandOptimizer:
    """Optimizes individual commands based on best practices."""
    
    def __init__(self, platform_adapter: PlatformAdapter):
        """
        Initialize the command optimizer.
        
        Args:
            platform_adapter: Platform adapter instance
        """
        self.platform_adapter = platform_adapter
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load optimization rules for different patterns."""
        return {
            'posix': {
                # File operations
                r'cat\s+([^\s]+)\s*\|\s*grep\s+([^\s]+)': r'grep \2 \1',
                r'cat\s+([^\s]+)\s*\|\s*less': r'less \1',
                r'cat\s+([^\s]+)\s*\|\s*more': r'more \1',
                r'ls\s+([^\s]+)\s*\|\s*grep\s+([^\s]+)': r'ls \1 | grep \2',
                
                # Combined commands
                r'sort\s+([^\s]+)\s*\|\s*uniq': r'sort -u \1',
                r'sort\s+([^\s]+)\s*\|\s*uniq\s*-c': r'sort \1 | uniq -c',
                
                # Find optimizations
                r'find\s+([^\s]+)\s*-name\s+"([^"]+)"\s*\|\s*xargs\s+rm': r'find \1 -name "\2" -delete',
                r'find\s+([^\s]+)\s*-name\s+"([^"]+)"\s*\|\s*xargs\s+cp\s+([^\s]+)': 
                r'find \1 -name "\2" -exec cp {} \3 \;',
                
                # Archive operations
                r'tar\s+-cf\s+([^\s]+)\s*([^\s]+)\s*&&\s*gzip\s+([^\s]+)': 
                r'tar -czf \1 \2',
                
                # Network operations
                r'curl\s+([^\s]+)\s*>\s*([^\s]+)': r'curl -o \2 \1',
                r'wget\s+([^\s]+)\s*-O\s+([^\s]+)': r'wget -O \2 \1',
            },
            'windows': {
                # File operations
                r'type\s+([^\s]+)\s*\|\s*findstr\s+([^\s]+)': r'findstr \2 \1',
                r'dir\s+([^\s]+)\s*\|\s*findstr\s+([^\s]+)': r'dir /s /b \1 | findstr \2',
                
                # Archive operations (PowerShell)
                r'powershell\s+-Command\s+"Compress-Archive\s*-Path\s+([^\s]+)\s*-DestinationPath\s+([^\s]+)"\s*&&\s*Compress-Archive\s*-Path\s+([^\s]+)\s*-DestinationPath\s+([^\s]+)':
                r'powershell -Command "Get-ChildItem \1, \3 | Compress-Archive -DestinationPath \2, \4"',
                
                # Environment variables
                r'echo\s+%([A-Z_]+)%': r'echo $env:\1',  # PowerShell style
            }
        }
    
    def optimize_command(self, command: str) -> OptimizationResult:
        """
        Optimize a single command.
        
        Args:
            command: Command to optimize
            
        Returns:
            OptimizationResult with optimization details
        """
        original_command = command
        optimized_command = command
        applied_changes = []
        warnings = []
        
        # Determine platform context
        rules_key = 'windows' if self.platform_adapter.platform_info.is_windows else 'posix'
        rules = self.optimization_rules.get(rules_key, {})
        
        # Apply optimization rules
        for pattern, replacement in rules.items():
            if re.search(pattern, optimized_command):
                new_command = re.sub(pattern, replacement, optimized_command)
                if new_command != optimized_command:
                    applied_changes.append(f"Applied rule: {pattern} -> {replacement}")
                    optimized_command = new_command
        
        # Apply general optimizations
        general_changes = self._apply_general_optimizations(optimized_command)
        applied_changes.extend(general_changes['changes'])
        optimized_command = general_changes['command']
        warnings.extend(general_changes['warnings'])
        
        # Calculate performance gain
        performance_gain = self._estimate_performance_gain(original_command, optimized_command)
        
        return OptimizationResult(
            original_command=original_command,
            optimized_command=optimized_command,
            performance_gain=performance_gain,
            estimated_time_saved=0.0,  # Would need benchmarking data
            optimization_type='pattern_matching',
            confidence=self._calculate_confidence(applied_changes),
            applied_changes=applied_changes,
            warnings=warnings,
            success=optimized_command != original_command or len(applied_changes) > 0
        )
    
    def _apply_general_optimizations(self, command: str) -> Dict[str, Any]:
        """Apply general command optimizations."""
        optimized = command
        changes = []
        warnings = []
        
        # Remove redundant whitespace
        if command != ' '.join(command.split()):
            optimized = ' '.join(command.split())
            changes.append("Normalized whitespace")
        
        # Optimize path separators
        if self.platform_adapter.platform_info.is_windows:
            # Convert forward slashes to backslashes
            if '/' in command and '\\' not in command:
                optimized = command.replace('/', '\\')
                changes.append("Converted path separators for Windows")
        else:
            # Ensure forward slashes for POSIX
            if '\\' in command:
                optimized = command.replace('\\', '/')
                changes.append("Converted path separators for POSIX")
        
        # Remove trailing spaces
        if command.endswith(' '):
            optimized = command.rstrip()
            changes.append("Removed trailing spaces")
        
        # Escape special characters
        if self._needs_escaping(optimized):
            optimized = self._escape_command(optimized)
            changes.append("Added proper escaping")
        
        return {
            'command': optimized,
            'changes': changes,
            'warnings': warnings
        }
    
    def _needs_escaping(self, command: str) -> bool:
        """Check if command needs proper escaping."""
        special_chars = [' ', '"', "'", '&', '|', ';', '>', '<', '`']
        return any(char in command for char in special_chars)
    
    def _escape_command(self, command: str) -> str:
        """Escape command for safe execution."""
        # Simple escaping - could be enhanced based on shell
        if self.platform_adapter.platform_info.is_windows:
            # Windows escaping (basic)
            return command.replace('"', '""')
        else:
            # POSIX escaping
            import shlex
            return shlex.quote(command)
    
    def _estimate_performance_gain(self, original: str, optimized: str) -> float:
        """Estimate performance gain from optimization."""
        # Simple heuristic based on command length and pattern complexity
        original_length = len(original)
        optimized_length = len(optimized)
        
        # Reduction in length suggests efficiency gain
        if original_length > 0:
            length_reduction = (original_length - optimized_length) / original_length
        else:
            length_reduction = 0
        
        # Pattern-based bonuses
        pattern_bonus = 0.0
        if 'cat |' in original:
            pattern_bonus += 0.1  # Cat pipeline elimination
        if 'sort | uniq' in original:
            pattern_bonus += 0.15  # Combined sort/uniq
        if 'find | xargs' in original:
            pattern_bonus += 0.2  # Find optimization
        
        return min(length_reduction + pattern_bonus, 0.5)  # Cap at 50%
    
    def _calculate_confidence(self, applied_changes: List[str]) -> float:
        """Calculate confidence score for optimizations."""
        if not applied_changes:
            return 0.0
        
        # More changes generally mean higher confidence
        confidence = min(len(applied_changes) / 5.0, 1.0)
        
        # Adjust based on change types
        high_confidence_changes = ['normalized whitespace', 'path separators', 'combined commands']
        for change in applied_changes:
            if any(hc in change.lower() for hc in high_confidence_changes):
                confidence += 0.1
        
        return min(confidence, 1.0)


class UsageOptimizer:
    """
    Comprehensive command usage optimization system.
    
    Provides pattern analysis, command optimization, and performance
    improvement recommendations for the tool usage system.
    """
    
    def __init__(self, platform_adapter: Optional[PlatformAdapter] = None,
                 command_executor: Optional[CommandExecutor] = None,
                 tool_registry: Optional[ToolRegistry] = None):
        """
        Initialize the usage optimizer.
        
        Args:
            platform_adapter: Platform adapter instance (uses default if None)
            command_executor: Command executor instance (creates default if None)
            tool_registry: Tool registry instance (creates default if None)
        """
        self.platform_adapter = platform_adapter or get_platform_adapter()
        self.command_executor = command_executor or CommandExecutor(self.platform_adapter)
        self.tool_registry = tool_registry or ToolRegistry(self.platform_adapter)
        
        # Initialize components
        self.pattern_analyzer = PatternAnalyzer()
        self.command_optimizer = CommandOptimizer(self.platform_adapter)
        
        # Performance tracking
        self.metrics_cache: Dict[str, CommandMetrics] = {}
        self.optimization_history: List[OptimizationResult] = []
        
        # Configuration
        self.auto_optimize = False
        self.performance_threshold = 0.1  # Minimum 10% improvement to suggest optimization
        self.max_optimization_history = 1000
    
    def analyze_usage_patterns(self, execution_history: Optional[List[CommandResult]] = None) -> List[UsagePattern]:
        """
        Analyze usage patterns from execution history.
        
        Args:
            execution_history: Command execution history (uses executor history if None)
            
        Returns:
            List of discovered usage patterns
        """
        if execution_history is None:
            execution_history = self.command_executor.get_execution_history()
        
        return self.pattern_analyzer.analyze_execution_history(execution_history)
    
    def optimize_command(self, command: str, apply_optimization: bool = False) -> OptimizationResult:
        """
        Optimize a single command.
        
        Args:
            command: Command to optimize
            apply_optimization: Whether to apply the optimization immediately
            
        Returns:
            OptimizationResult with optimization details
        """
        result = self.command_optimizer.optimize_command(command)
        
        # Add to optimization history
        self.optimization_history.append(result)
        
        # Trim history if needed
        if len(self.optimization_history) > self.max_optimization_history:
            self.optimization_history = self.optimization_history[-self.max_optimization_history:]
        
        # Apply optimization if requested
        if apply_optimization and result.success and result.performance_gain > self.performance_threshold:
            return self._apply_optimization(result)
        
        return result
    
    def _apply_optimization(self, optimization_result: OptimizationResult) -> OptimizationResult:
        """Apply an optimization and return updated result."""
        # In a real implementation, this would update command templates
        # or configuration files based on the optimization
        
        optimization_result.optimization_type = 'applied_' + optimization_result.optimization_type
        optimization_result.confidence = min(optimization_result.confidence + 0.1, 1.0)
        
        return optimization_result
    
    def batch_optimize(self, commands: List[str], min_improvement: float = 0.1) -> List[OptimizationResult]:
        """
        Optimize multiple commands in batch.
        
        Args:
            commands: List of commands to optimize
            min_improvement: Minimum improvement threshold
            
        Returns:
            List of optimization results
        """
        results = []
        
        for command in commands:
            result = self.optimize_command(command)
            
            # Filter by improvement threshold
            if result.performance_gain >= min_improvement:
                results.append(result)
            else:
                # Create a result indicating no significant improvement
                no_improvement = OptimizationResult(
                    original_command=command,
                    optimized_command=command,
                    performance_gain=0.0,
                    estimated_time_saved=0.0,
                    optimization_type='no_improvement',
                    confidence=1.0,
                    applied_changes=[],
                    warnings=[f'No significant improvement found (threshold: {min_improvement})'],
                    success=False
                )
                results.append(no_improvement)
        
        return results
    
    def get_performance_metrics(self, command: str) -> Optional[CommandMetrics]:
        """
        Get performance metrics for a specific command.
        
        Args:
            command: Command to get metrics for
            
        Returns:
            CommandMetrics object if available, None otherwise
        """
        return self.metrics_cache.get(command)
    
    def update_metrics(self, execution_result: CommandResult) -> None:
        """
        Update performance metrics based on execution result.
        
        Args:
            execution_result: Command execution result
        """
        command = execution_result.command
        
        if command not in self.metrics_cache:
            self.metrics_cache[command] = CommandMetrics(command=command)
        
        metrics = self.metrics_cache[command]
        
        # Update basic metrics
        metrics.total_executions += 1
        if execution_result.success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
        
        # Update timing metrics
        if execution_result.execution_time > 0:
            metrics.total_time += execution_result.execution_time
            metrics.min_time = min(metrics.min_time, execution_result.execution_time)
            metrics.max_time = max(metrics.max_time, execution_result.execution_time)
            
            # Recalculate average and standard deviation
            if metrics.total_executions > 0:
                metrics.avg_time = metrics.total_time / metrics.total_executions
                
                # Calculate standard deviation
                if metrics.total_executions > 1:
                    variance = sum((execution_result.execution_time - metrics.avg_time) ** 2 
                                 for result in self.command_executor.get_execution_history()
                                 if result.command == command and result.execution_time > 0) / (metrics.total_executions - 1)
                    metrics.std_dev = variance ** 0.5
        
        # Update other metrics
        metrics.last_execution = datetime.fromtimestamp(execution_result.timestamp, timezone.utc)
        metrics.success_rate = metrics.successful_executions / metrics.total_executions
        
        # Calculate efficiency score (could be enhanced)
        if metrics.avg_time > 0:
            metrics.efficiency_score = min(1.0, 1.0 / (1.0 + metrics.avg_time))
    
    def get_optimization_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations based on analysis.
        
        Args:
            limit: Maximum number of recommendations to return
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Analyze patterns for recommendations
        patterns = self.analyze_usage_patterns()
        for pattern in sorted(patterns, key=lambda p: p.performance_score, reverse=True):
            if len(recommendations) >= limit:
                break
            
            if pattern.performance_score < 0.5:  # Only recommend for low-performing patterns
                recommendations.append({
                    'type': 'pattern_optimization',
                    'pattern_id': pattern.pattern_id,
                    'template': pattern.command_template,
                    'frequency': pattern.frequency,
                    'avg_execution_time': pattern.avg_execution_time,
                    'success_rate': pattern.success_rate,
                    'suggestions': pattern.optimization_suggestions,
                    'potential_gain': (1.0 - pattern.performance_score) * 100
                })
        
        # Analyze individual commands
        for command, metrics in self.metrics_cache.items():
            if len(recommendations) >= limit:
                break
            
            if metrics.efficiency_score < 0.7:  # Low efficiency
                recommendations.append({
                    'type': 'command_optimization',
                    'command': command,
                    'total_executions': metrics.total_executions,
                    'avg_time': metrics.avg_time,
                    'success_rate': metrics.success_rate,
                    'efficiency_score': metrics.efficiency_score,
                    'potential_gain': (1.0 - metrics.efficiency_score) * 100
                })
        
        return recommendations
    
    def get_optimizer_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimizer state and capabilities.
        
        Returns:
            Dictionary containing optimizer summary
        """
        patterns = self.analyze_usage_patterns()
        
        return {
            'metrics': {
                'commands_tracked': len(self.metrics_cache),
                'total_executions': sum(m.total_executions for m in self.metrics_cache.values()),
                'average_efficiency': statistics.mean([m.efficiency_score for m in self.metrics_cache.values()]) if self.metrics_cache else 0.0
            },
            'patterns': {
                'total_patterns': len(patterns),
                'high_performance_patterns': len([p for p in patterns if p.performance_score > 0.7]),
                'optimization_candidates': len([p for p in patterns if p.performance_score < 0.5])
            },
            'optimizations': {
                'total_optimizations': len(self.optimization_history),
                'successful_optimizations': len([o for o in self.optimization_history if o.success]),
                'average_improvement': statistics.mean([o.performance_gain for o in self.optimization_history]) if self.optimization_history else 0.0
            },
            'configuration': {
                'auto_optimize': self.auto_optimize,
                'performance_threshold': self.performance_threshold,
                'max_optimization_history': self.max_optimization_history
            }
        }