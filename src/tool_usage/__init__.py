"""
Tool Usage Module

Core tool usage management system for the mini-biai-1 framework.
Provides comprehensive capabilities for tool detection, execution, 
optimization, and management across different platforms and environments.

Classes:
    PlatformAdapter: Base class for OS-specific adaptations
    PosixAdapter: Platform adapter for POSIX systems (Linux, macOS, Unix)
    WindowsAdapter: Platform adapter for Windows systems
    ShellDetector: Cross-platform shell identification and detection
    CommandExecutor: Secure command execution with security validation
    ToolRegistry: Comprehensive tool management and discovery system
    UsageOptimizer: Command pattern optimization and performance analysis
    UsageOptimizer: Main optimization system combining pattern analysis and command optimization

The tool_usage module provides:
- Cross-platform shell and tool detection
- Secure command execution with multiple security levels
- Comprehensive tool registry and management
- Pattern-based optimization and performance analysis
- Usage analytics and recommendation systems

Example usage:
    # Basic setup
    from src.tool_usage import ToolUsageManager
    
    manager = ToolUsageManager()
    
    # Execute a command
    result = manager.execute_command("ls -la", security_level="medium")
    print(f"Result: {result.stdout}")
    
    # Discover available tools
    tools = manager.discover_tools()
    print(f"Found {len(tools)} tools")
    
    # Optimize a command
    optimization = manager.optimize_command("cat file.txt | grep pattern")
    print(f"Optimized: {optimization.optimized_command}")
    
    # Get optimization recommendations
    recommendations = manager.get_optimization_recommendations()
    for rec in recommendations:
        print(f"Recommendation: {rec}")
"""

import sys
import os
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

# Import all components
from .platform_adapter import (
    PlatformAdapter, 
    PosixAdapter, 
    WindowsAdapter,
    get_platform_adapter,
    get_platform_info,
    PlatformInfo
)

from .shell_detector import (
    ShellDetector,
    ShellInfo
)

from .command_executor import (
    CommandExecutor,
    CommandResult,
    ExecutionConfig,
    ExecutionMode,
    SecurityLevel,
    SecurityValidator
)

from .tool_registry import (
    ToolRegistry,
    ToolMetadata,
    ToolCategory,
    ToolStatus,
    ToolDiscovery
)

from .usage_optimizer import (
    UsageOptimizer,
    UsagePattern,
    OptimizationResult,
    CommandMetrics,
    PatternAnalyzer,
    CommandOptimizer
)

# Import platform-specific command handlers
from .handlers import (
    # Base classes and exceptions
    BaseCommandHandler,
    CommandResult as HandlerCommandResult, 
    CommandConfig as HandlerCommandConfig,
    CommandExecutionError,
    SecurityError,
    TimeoutError,
    PlatformNotSupportedError,
    
    # Handler classes
    UnixShellHandler,
    PowerShellHandler, 
    WindowsCommandHandler,
    WSLHandler,
    SSHHandler,
    
    # SSH-specific exceptions
    SSHConnectionError,
    SSHAuthenticationError,
    SSHTimeoutError,
    
    # Utility functions
    get_handler_for_platform,
    get_handler_by_type,
    list_available_handlers,
    validate_platform_support,
    get_platform_info as get_handler_platform_info
)


class ToolUsageManager:
    """
    Main tool usage management interface.
    
    Provides a unified interface to all tool usage capabilities
    including detection, execution, optimization, and management.
    """
    
    def __init__(self, 
                 platform_adapter: Optional[PlatformAdapter] = None,
                 auto_discovery: bool = True,
                 auto_optimization: bool = False,
                 security_level: SecurityLevel = SecurityLevel.MEDIUM):
        """
        Initialize the tool usage manager.
        
        Args:
            platform_adapter: Platform adapter instance (uses default if None)
            auto_discovery: Whether to auto-discover tools on initialization
            auto_optimization: Whether to apply optimizations automatically
            security_level: Default security level for command execution
        """
        # Initialize platform adapter
        self.platform_adapter = platform_adapter or get_platform_adapter()
        
        # Initialize core components
        self.shell_detector = ShellDetector(self.platform_adapter)
        self.command_executor = CommandExecutor(self.platform_adapter)
        self.tool_registry = ToolRegistry(self.platform_adapter)
        self.usage_optimizer = UsageOptimizer(
            self.platform_adapter,
            self.command_executor,
            self.tool_registry
        )
        
        # Configure auto-features
        self.tool_registry.auto_discovery = auto_discovery
        self.usage_optimizer.auto_optimize = auto_optimization
        self.command_executor.default_config.security_level = security_level
        
        # Platform and system information
        self.platform_info = self.platform_adapter.platform_info
        self.shell_info = self.shell_detector.get_current_shell()
        
        # Initialize statistics
        self._initialization_time = self._get_current_timestamp()
        self._total_commands_executed = 0
        self._total_tools_discovered = 0
        self._total_optimizations_applied = 0
    
    def _get_current_timestamp(self) -> float:
        """Get current timestamp for statistics."""
        import time
        return time.time()
    
    # Shell Detection Methods
    
    def get_available_shells(self) -> List[ShellInfo]:
        """
        Get list of all available shells on the system.
        
        Returns:
            List of ShellInfo objects for available shells
        """
        return self.shell_detector.get_available_shells()
    
    def get_current_shell(self) -> Optional[ShellInfo]:
        """
        Get information about the currently active shell.
        
        Returns:
            ShellInfo object for the current shell
        """
        return self.shell_detector.get_current_shell()
    
    def get_shell_by_name(self, shell_name: str) -> Optional[ShellInfo]:
        """
        Get shell information by name.
        
        Args:
            shell_name: Name of the shell (e.g., 'bash', 'powershell')
            
        Returns:
            ShellInfo object if found, None otherwise
        """
        return self.shell_detector.get_shell_by_name(shell_name)
    
    def get_default_shell(self) -> Optional[ShellInfo]:
        """
        Get the default shell for the current platform.
        
        Returns:
            ShellInfo object for the default shell
        """
        return self.shell_detector.get_default_shell()
    
    # Command Execution Methods
    
    def execute_command(self, 
                       command: str, 
                       config: Optional[ExecutionConfig] = None,
                       **kwargs) -> CommandResult:
        """
        Execute a command with the specified configuration.
        
        Args:
            command: Command to execute
            config: Execution configuration (uses default if None)
            **kwargs: Additional configuration parameters
            
        Returns:
            CommandResult object with execution details
        """
        # Create config from kwargs if provided
        if config is None and kwargs:
            config = ExecutionConfig(**kwargs)
        
        # Update metrics after execution
        result = self.command_executor.execute(command, config)
        self.usage_optimizer.update_metrics(result)
        self._total_commands_executed += 1
        
        return result
    
    def execute_interactive(self, command: str, stdin: Optional[str] = None,
                          timeout: float = 30.0) -> CommandResult:
        """
        Execute a command in interactive mode.
        
        Args:
            command: Command to execute
            stdin: Input to send to the command
            timeout: Execution timeout in seconds
            
        Returns:
            CommandResult object with execution details
        """
        config = ExecutionConfig(
            mode=ExecutionMode.INTERACTIVE,
            stdin=stdin,
            timeout=timeout
        )
        
        return self.execute_command(command, config)
    
    def execute_background(self, command: str) -> CommandResult:
        """
        Execute a command in background mode.
        
        Args:
            command: Command to execute
            
        Returns:
            CommandResult object with execution details
        """
        config = ExecutionConfig(mode=ExecutionMode.BACKGROUND)
        return self.execute_command(command, config)
    
    def terminate_process(self, pid: int) -> bool:
        """
        Terminate a running process.
        
        Args:
            pid: Process ID to terminate
            
        Returns:
            True if process was terminated successfully
        """
        return self.command_executor.terminate_process(pid)
    
    def get_process_status(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get status information for a running process.
        
        Args:
            pid: Process ID to check
            
        Returns:
            Dictionary with process status information
        """
        return self.command_executor.get_process_status(pid)
    
    # Tool Registry Methods
    
    def register_tool(self, tool_metadata: ToolMetadata) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            tool_metadata: Tool metadata to register
            
        Returns:
            True if registration successful, False otherwise
        """
        success = self.tool_registry.register_tool(tool_metadata)
        if success:
            self._total_tools_discovered += 1
        return success
    
    def discover_tools(self, categories: Optional[List[ToolCategory]] = None) -> List[ToolMetadata]:
        """
        Discover and register all available tools.
        
        Args:
            categories: Optional list of categories to limit discovery to
            
        Returns:
            List of ToolMetadata objects for discovered tools
        """
        count = self.tool_registry.discover_and_register_tools(categories)
        self._total_tools_discovered += count
        return self.tool_registry.get_available_tools() if hasattr(self.tool_registry, 'get_available_tools') else []
    
    def get_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Get tool metadata by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolMetadata object if found, None otherwise
        """
        return self.tool_registry.get_tool(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolMetadata]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of ToolMetadata objects in the category
        """
        return self.tool_registry.get_tools_by_category(category)
    
    def search_tools(self, query: str) -> List[ToolMetadata]:
        """
        Search for tools by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of ToolMetadata objects matching the query
        """
        return self.tool_registry.search_tools(query)
    
    def update_tool_status(self, tool_name: str) -> bool:
        """
        Update the status of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if update successful, False otherwise
        """
        return self.tool_registry.update_tool_status(tool_name)
    
    # Optimization Methods
    
    def optimize_command(self, command: str, apply_optimization: bool = False) -> OptimizationResult:
        """
        Optimize a single command.
        
        Args:
            command: Command to optimize
            apply_optimization: Whether to apply the optimization immediately
            
        Returns:
            OptimizationResult with optimization details
        """
        result = self.usage_optimizer.optimize_command(command, apply_optimization)
        if result.success:
            self._total_optimizations_applied += 1
        return result
    
    def batch_optimize(self, commands: List[str], min_improvement: float = 0.1) -> List[OptimizationResult]:
        """
        Optimize multiple commands in batch.
        
        Args:
            commands: List of commands to optimize
            min_improvement: Minimum improvement threshold
            
        Returns:
            List of optimization results
        """
        return self.usage_optimizer.batch_optimize(commands, min_improvement)
    
    def analyze_usage_patterns(self, execution_history: Optional[List[CommandResult]] = None) -> List[UsagePattern]:
        """
        Analyze usage patterns from execution history.
        
        Args:
            execution_history: Command execution history (uses manager history if None)
            
        Returns:
            List of discovered usage patterns
        """
        return self.usage_optimizer.analyze_usage_patterns(execution_history)
    
    def get_optimization_recommendations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations based on analysis.
        
        Args:
            limit: Maximum number of recommendations to return
            
        Returns:
            List of optimization recommendations
        """
        return self.usage_optimizer.get_optimization_recommendations(limit)
    
    def get_performance_metrics(self, command: str) -> Optional[CommandMetrics]:
        """
        Get performance metrics for a specific command.
        
        Args:
            command: Command to get metrics for
            
        Returns:
            CommandMetrics object if available, None otherwise
        """
        return self.usage_optimizer.get_performance_metrics(command)
    
    # Analytics and Reporting Methods
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[CommandResult]:
        """
        Get command execution history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of CommandResult objects
        """
        return self.command_executor.get_execution_history(limit)
    
    def get_tool_registry_summary(self) -> Dict[str, Any]:
        """
        Get summary of the tool registry.
        
        Returns:
            Dictionary containing registry summary
        """
        return self.tool_registry.get_registry_summary()
    
    def get_shell_detection_summary(self) -> Dict[str, Any]:
        """
        Get summary of shell detection results.
        
        Returns:
            Dictionary containing shell detection summary
        """
        return self.shell_detector.get_shell_summary()
    
    def get_optimizer_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimizer state and capabilities.
        
        Returns:
            Dictionary containing optimizer summary
        """
        return self.usage_optimizer.get_optimizer_summary()
    
    def get_platform_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive platform and system summary.
        
        Returns:
            Dictionary containing platform and system information
        """
        return {
            'platform': {
                'name': self.platform_info.name,
                'version': self.platform_info.version,
                'machine': self.platform_info.machine,
                'is_windows': self.platform_info.is_windows,
                'is_posix': self.platform_info.is_posix,
                'is_linux': self.platform_info.is_linux,
                'is_macos': self.platform_info.is_macos
            },
            'current_shell': {
                'name': self.shell_info.name if self.shell_info else None,
                'path': self.shell_info.path if self.shell_info else None,
                'version': self.shell_info.version if self.shell_info else None
            } if self.shell_info else None,
            'statistics': {
                'initialization_time': self._initialization_time,
                'uptime_seconds': self._get_current_timestamp() - self._initialization_time,
                'total_commands_executed': self._total_commands_executed,
                'total_tools_discovered': self._total_tools_discovered,
                'total_optimizations_applied': self._total_optimizations_applied
            }
        }
    
    def export_configuration(self, filepath: Union[str, Path]) -> bool:
        """
        Export tool usage configuration and registry.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            export_data = {
                'metadata': {
                    'exported_at': self._get_current_timestamp(),
                    'platform': self.platform_info.name,
                    'manager_version': '1.0'
                },
                'platform_info': {
                    'name': self.platform_info.name,
                    'version': self.platform_info.version,
                    'machine': self.platform_info.machine,
                    'is_windows': self.platform_info.is_windows,
                    'is_posix': self.platform_info.is_posix
                },
                'shell_info': {
                    'current_shell': {
                        'name': self.shell_info.name if self.shell_info else None,
                        'path': self.shell_info.path if self.shell_info else None,
                        'version': self.shell_info.version if self.shell_info else None
                    },
                    'available_shells': [
                        {
                            'name': shell.name,
                            'path': shell.path,
                            'version': shell.version,
                            'capabilities': list(shell.capabilities) if shell.capabilities else []
                        }
                        for shell in self.shell_detector.get_available_shells()
                    ]
                },
                'configuration': {
                    'auto_discovery': self.tool_registry.auto_discovery,
                    'auto_optimization': self.usage_optimizer.auto_optimize,
                    'default_security_level': self.command_executor.default_config.security_level.value,
                    'performance_threshold': self.usage_optimizer.performance_threshold
                },
                'statistics': {
                    'initialization_time': self._initialization_time,
                    'total_commands_executed': self._total_commands_executed,
                    'total_tools_discovered': self._total_tools_discovered,
                    'total_optimizations_applied': self._total_optimizations_applied
                }
            }
            
            # Export tool registry
            registry_filepath = str(filepath).replace('.json', '_registry.json')
            self.tool_registry.export_registry(registry_filepath, 'json')
            
            # Export main configuration
            import json
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
            
        except Exception:
            return False
    
    def clear_history(self) -> None:
        """Clear execution history and reset counters."""
        self.command_executor.clear_history()
        self._total_commands_executed = 0
        self._initialization_time = self._get_current_timestamp()


# Convenience functions for quick access

def create_tool_usage_manager(**kwargs) -> ToolUsageManager:
    """
    Create a ToolUsageManager instance with the provided configuration.
    
    Args:
        **kwargs: Configuration parameters for the manager
        
    Returns:
        ToolUsageManager instance
    """
    return ToolUsageManager(**kwargs)


def get_quick_summary() -> Dict[str, Any]:
    """
    Get a quick summary of the current system and tools.
    
    Returns:
        Dictionary with quick system summary
    """
    manager = ToolUsageManager()
    return {
        'platform': manager.get_platform_summary(),
        'shells': manager.get_shell_detection_summary(),
        'registry': manager.get_tool_registry_summary(),
        'optimizer': manager.get_optimizer_summary()
    }


def quick_execute(command: str, **kwargs) -> CommandResult:
    """
    Quick command execution with default settings.
    
    Args:
        command: Command to execute
        **kwargs: Additional execution parameters
        
    Returns:
        CommandResult object
    """
    manager = ToolUsageManager()
    return manager.execute_command(command, **kwargs)


# Platform-specific handler convenience functions

def get_platform_handler() -> BaseCommandHandler:
    """
    Get the appropriate command handler for the current platform.
    
    This function automatically detects the current platform and returns
    the most suitable command handler with default configuration.
    
    Returns:
        BaseCommandHandler: Platform-specific command handler instance
        
    Raises:
        PlatformNotSupportedError: If no suitable handler is found for the platform
    """
    return get_handler_for_platform()


def execute_with_handler(command: str, handler_type: str = None, **kwargs) -> HandlerCommandResult:
    """
    Execute a command using a specific or automatically detected handler.
    
    Args:
        command: Command to execute
        handler_type: Type of handler to use ('unix', 'powershell', 'cmd', 'wsl', 'ssh', or None for auto-detect)
        **kwargs: Additional handler initialization parameters
        
    Returns:
        HandlerCommandResult object with execution details
    """
    if handler_type:
        handler = get_handler_by_type(handler_type, **kwargs)
    else:
        handler = get_platform_handler()
    
    return handler.execute_command(command)


def test_platform_handlers() -> Dict[str, bool]:
    """
    Test all available handlers on the current platform.
    
    Returns:
        Dictionary mapping handler types to their availability status
    """
    results = {}
    
    handler_tests = [
        ('unix', lambda: UnixShellHandler().is_platform_supported()),
        ('powershell', lambda: PowerShellHandler().is_platform_supported()),
        ('cmd', lambda: WindowsCommandHandler().is_platform_supported()),
        ('wsl', lambda: WSLHandler().is_platform_supported()),
        ('ssh', lambda: SSHHandler(hostname="test", username="test").is_platform_supported())
    ]
    
    for handler_type, test_func in handler_tests:
        try:
            results[handler_type] = test_func()
        except Exception:
            results[handler_type] = False
    
    return results


def get_comprehensive_platform_info() -> Dict[str, Any]:
    """
    Get comprehensive platform information including handler capabilities.
    
    Returns:
        Dictionary containing comprehensive platform and handler information
    """
    base_info = get_platform_info()
    handler_info = get_handler_platform_info()
    handler_tests = test_platform_handlers()
    
    return {
        'platform_info': base_info,
        'handler_info': handler_info,
        'handler_availability': handler_tests,
        'recommended_handler': None
    }


# Enhanced ToolUsageManager with handler integration
class EnhancedToolUsageManager(ToolUsageManager):
    """
    Enhanced ToolUsageManager with platform-specific handler integration.
    
    Extends the base ToolUsageManager with additional capabilities for
    platform-specific command execution using the new handlers.
    """
    
    def __init__(self, **kwargs):
        """Initialize the enhanced tool usage manager."""
        super().__init__(**kwargs)
        self.platform_handlers = {}
        self._initialize_platform_handlers()
    
    def _initialize_platform_handlers(self):
        """Initialize available platform handlers."""
        try:
            self.platform_handlers['auto'] = get_platform_handler()
        except PlatformNotSupportedError:
            pass
        
        # Initialize specific handlers if available
        handler_types = ['unix', 'powershell', 'cmd', 'wsl', 'ssh']
        for handler_type in handler_types:
            try:
                if validate_platform_support(handler_type):
                    self.platform_handlers[handler_type] = get_handler_by_type(handler_type)
            except Exception:
                pass
    
    def execute_with_platform_handler(self, command: str, handler_type: str = 'auto') -> HandlerCommandResult:
        """
        Execute a command using a platform-specific handler.
        
        Args:
            command: Command to execute
            handler_type: Type of handler to use ('auto' or specific type)
            
        Returns:
            HandlerCommandResult object with execution details
        """
        if handler_type == 'auto':
            handler = self.platform_handlers.get('auto')
            if not handler:
                raise PlatformNotSupportedError("No platform handler available")
        else:
            handler = self.platform_handlers.get(handler_type)
            if not handler:
                raise PlatformNotSupportedError(f"Handler '{handler_type}' not available")
        
        return handler.execute_command(command)
    
    def get_platform_handler_info(self) -> Dict[str, Any]:
        """
        Get information about available platform handlers.
        
        Returns:
            Dictionary containing handler information
        """
        info = {
            'available_handlers': list(self.platform_handlers.keys()),
            'default_handler': 'auto' if 'auto' in self.platform_handlers else None,
            'handler_details': {}
        }
        
        for handler_name, handler in self.platform_handlers.items():
            try:
                handler_info = {
                    'type': handler.__class__.__name__,
                    'supported': handler.is_platform_supported(),
                    'platform': handler.environment_info.get('platform', 'unknown')
                }
                
                # Get handler-specific info
                if hasattr(handler, 'get_unix_info'):
                    handler_info.update(handler.get_unix_info())
                elif hasattr(handler, 'get_powershell_info'):
                    handler_info.update(handler.get_powershell_info())
                elif hasattr(handler, 'get_windows_info'):
                    handler_info.update(handler.get_windows_info())
                elif hasattr(handler, 'get_wsl_info'):
                    handler_info.update(handler.get_wsl_info())
                elif hasattr(handler, 'get_ssh_info'):
                    handler_info.update(handler.get_ssh_info())
                
                info['handler_details'][handler_name] = handler_info
                
            except Exception as e:
                info['handler_details'][handler_name] = {'error': str(e)}
        
        return info


# Version information
__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"
__email__ = "team@mini-biai-1.org"

# Module-level exports
__all__ = [
    # Core classes
    'ToolUsageManager',
    'PlatformAdapter',
    'PosixAdapter', 
    'WindowsAdapter',
    'ShellDetector',
    'CommandExecutor',
    'ToolRegistry',
    'UsageOptimizer',
    
    # Data classes
    'ShellInfo',
    'ToolMetadata',
    'CommandResult',
    'ExecutionConfig',
    'UsagePattern',
    'OptimizationResult',
    'CommandMetrics',
    'PlatformInfo',
    
    # Enums
    'ExecutionMode',
    'SecurityLevel',
    'ToolCategory',
    'ToolStatus',
    
    # Platform-specific command handlers
    'BaseCommandHandler',
    'HandlerCommandResult', 
    'HandlerCommandConfig',
    'CommandExecutionError',
    'SecurityError',
    'TimeoutError',
    'PlatformNotSupportedError',
    
    # Handler classes
    'UnixShellHandler',
    'PowerShellHandler', 
    'WindowsCommandHandler',
    'WSLHandler',
    'SSHHandler',
    
    # SSH-specific exceptions
    'SSHConnectionError',
    'SSHAuthenticationError',
    'SSHTimeoutError',
    
    # Convenience functions
    'create_tool_usage_manager',
    'get_quick_summary',
    'quick_execute',
    'get_platform_adapter',
    'get_platform_info',
    'get_handler_for_platform',
    'get_handler_by_type',
    'list_available_handlers',
    'validate_platform_support',
    'get_handler_platform_info',
    'get_platform_handler',
    'execute_with_handler',
    'test_platform_handlers',
    'get_comprehensive_platform_info',
    'EnhancedToolUsageManager',
]