"""
Unit Tests for Command Executor Component
==========================================

Comprehensive unit tests for the CommandExecutor class, testing command execution,
security validation, process management, and cross-platform capabilities.
"""

import pytest
import os
import sys
import subprocess
import time
import signal
import threading
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.command_executor import (
        CommandExecutor, CommandResult, ExecutionConfig, 
        ExecutionMode, SecurityLevel, SecurityValidator
    )
    from tool_usage.platform_adapter import PosixAdapter, WindowsAdapter
except ImportError as e:
    pytest.skip(f"Could not import CommandExecutor: {e}", allow_module_level=True)


class TestCommandResult:
    """Test CommandResult dataclass"""
    
    def test_command_result_creation(self):
        """Test creating CommandResult object"""
        result = CommandResult(
            command="echo 'test'",
            stdout="test\n",
            stderr="",
            return_code=0,
            execution_time=0.01,
            timestamp=time.time(),
            pid=1234,
            success=True
        )
        
        assert result.command == "echo 'test'"
        assert result.stdout == "test\n"
        assert result.stderr == ""
        assert result.return_code == 0
        assert result.execution_time == 0.01
        assert result.success is True
        assert result.pid == 1234
        assert result.timestamp is not None
    
    def test_command_result_failure(self):
        """Test creating failed CommandResult"""
        result = CommandResult(
            command="false",
            stdout="",
            stderr="Command failed",
            return_code=1,
            execution_time=0.01,
            timestamp=time.time(),
            success=False
        )
        
        assert result.success is False
        assert result.return_code != 0
        assert "Command failed" in result.stderr
    
    def test_command_result_minimal(self):
        """Test creating CommandResult with minimal data"""
        result = CommandResult(
            command="test",
            stdout="",
            stderr="",
            return_code=0,
            execution_time=0.0,
            timestamp=time.time()
        )
        
        assert result.command == "test"
        assert result.pid is None  # Default
        assert result.success is True  # Default


class TestExecutionConfig:
    """Test ExecutionConfig dataclass"""
    
    def test_execution_config_creation(self):
        """Test creating ExecutionConfig object"""
        config = ExecutionConfig(
            mode=ExecutionMode.SYNCHRONOUS,
            timeout=30.0,
            security_level=SecurityLevel.MEDIUM,
            working_directory="/tmp",
            environment_vars={"TEST": "value"}
        )
        
        assert config.mode == ExecutionMode.SYNCHRONOUS
        assert config.timeout == 30.0
        assert config.security_level == SecurityLevel.MEDIUM
        assert config.working_directory == "/tmp"
        assert config.environment_vars["TEST"] == "value"
    
    def test_execution_config_defaults(self):
        """Test ExecutionConfig with default values"""
        config = ExecutionConfig()
        
        assert config.mode == ExecutionMode.SYNCHRONOUS
        assert config.timeout is None
        assert config.security_level == SecurityLevel.MEDIUM
        assert config.working_directory is None
        assert config.environment_vars is None
        assert config.stdin is None
        assert config.stdout is None
        assert config.stderr is None
    
    @pytest.mark.parametrize("mode", [ExecutionMode.SYNCHRONOUS, ExecutionMode.ASYNC, ExecutionMode.INTERACTIVE, ExecutionMode.BACKGROUND])
    def test_execution_modes(self, mode):
        """Test different execution modes"""
        config = ExecutionConfig(mode=mode)
        assert config.mode == mode
    
    @pytest.mark.parametrize("level", [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.PARANOID])
    def test_security_levels(self, level):
        """Test different security levels"""
        config = ExecutionConfig(security_level=level)
        assert config.security_level == level


class TestSecurityValidator:
    """Test SecurityValidator functionality"""
    
    def test_validate_safe_command(self):
        """Test validation of safe commands"""
        validator = SecurityValidator()
        
        safe_commands = [
            "echo 'hello'",
            "ls -la /tmp",
            "pwd",
            "whoami",
            "cat /etc/hosts"
        ]
        
        for command in safe_commands:
            assert validator.validate_command(command), f"Command should be safe: {command}"
    
    def test_validate_dangerous_command(self):
        """Test validation of dangerous commands"""
        validator = SecurityValidator()
        
        dangerous_commands = [
            "rm -rf /",
            "format C:",
            "sudo rm -rf /",
            "> /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda"
        ]
        
        for command in dangerous_commands:
            assert not validator.validate_command(command), f"Command should be dangerous: {command}"
    
    def test_validate_path_traversal(self):
        """Test validation of path traversal attempts"""
        validator = SecurityValidator()
        
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/../../../etc/shadow",
            "....//....//....//etc/passwd",
            "/tmp/../../../etc/passwd"
        ]
        
        for path in malicious_paths:
            assert not validator.validate_path(path), f"Path should be malicious: {path}"
    
    def test_validate_environment_variables(self):
        """Test validation of environment variables"""
        validator = SecurityValidator()
        
        # Safe environment
        safe_env = {"PATH": "/usr/bin", "HOME": "/home/user", "USER": "test"}
        assert validator.validate_environment(safe_env)
        
        # Unsafe environment
        unsafe_env = {"PATH": "/malicious/bin", "LD_PRELOAD": "/evil.so"}
        assert not validator.validate_environment(unsafe_env)
    
    def test_security_level_enforcement(self):
        """Test enforcement of different security levels"""
        validator = SecurityValidator()
        
        # Test different security levels
        for level in [SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH, SecurityLevel.PARANOID]:
            config = ExecutionConfig(security_level=level)
            
            # Should accept safe commands regardless of level
            assert validator.validate_command("echo 'test'", config)
            
            # Should reject dangerous commands at all levels
            assert not validator.validate_command("rm -rf /", config)


class TestCommandExecutorInitialization:
    """Test CommandExecutor initialization"""
    
    def test_executor_initialization(self, posix_adapter):
        """Test basic CommandExecutor initialization"""
        executor = CommandExecutor(posix_adapter)
        
        assert executor.platform_adapter == posix_adapter
        assert executor.default_config is not None
        assert executor.execution_history == []
        assert executor._active_processes == {}
    
    def test_executor_with_default_adapter(self):
        """Test CommandExecutor with default adapter"""
        executor = CommandExecutor()
        assert executor.platform_adapter is not None
    
    def test_executor_custom_default_config(self):
        """Test CommandExecutor with custom default config"""
        custom_config = ExecutionConfig(
            timeout=60.0,
            security_level=SecurityLevel.HIGH
        )
        
        executor = CommandExecutor(default_config=custom_config)
        
        assert executor.default_config.timeout == 60.0
        assert executor.default_config.security_level == SecurityLevel.HIGH


class TestCommandExecution:
    """Test command execution functionality"""
    
    @pytest.fixture
    def executor_mock(self, posix_adapter):
        """Create executor with mocked subprocess"""
        executor = CommandExecutor(posix_adapter)
        
        # Mock subprocess.Popen
        mock_process = Mock()
        mock_process.communicate.return_value = ("test output", "")
        mock_process.returncode = 0
        mock_process.pid = 1234
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_popen.return_value = mock_process
            
            yield executor, mock_popen
    
    def test_execute_simple_command(self, executor_mock):
        """Test execution of a simple command"""
        executor, mock_popen = executor_mock
        
        mock_process = Mock()
        mock_process.communicate.return_value = ("hello world\n", "")
        mock_process.returncode = 0
        mock_process.pid = 1234
        mock_popen.return_value = mock_process
        
        result = executor.execute("echo 'hello world'")
        
        assert result.command == "echo 'hello world'"
        assert result.stdout == "hello world\n"
        assert result.return_code == 0
        assert result.success is True
        assert result.pid == 1234
        assert result.execution_time >= 0
    
    def test_execute_command_with_timeout(self, executor_mock):
        """Test command execution with timeout"""
        executor, mock_popen = executor_mock
        
        # Mock timeout scenario
        mock_process = Mock()
        mock_process.communicate.side_effect = subprocess.TimeoutExpired("test", 5.0)
        mock_process.pid = 1234
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_popen.return_value = mock_process
            
            result = executor.execute("sleep 10", timeout=1.0)
            
            assert result.success is False
            assert "timeout" in result.stderr.lower() or result.return_code == -1
    
    def test_execute_command_with_error(self, executor_mock):
        """Test command execution with error"""
        executor, mock_popen = executor_mock
        
        mock_process = Mock()
        mock_process.communicate.return_value = ("", "error: command not found")
        mock_process.returncode = 127
        mock_process.pid = 1234
        mock_popen.return_value = mock_process
        
        result = executor.execute("nonexistent_command")
        
        assert result.success is False
        assert result.return_code == 127
        assert "command not found" in result.stderr
    
    def test_execute_with_custom_working_directory(self, executor_mock):
        """Test command execution with custom working directory"""
        executor, mock_popen = executor_mock
        
        mock_process = Mock()
        mock_process.communicate.return_value = ("output", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        custom_dir = "/tmp/test"
        result = executor.execute("pwd", working_directory=custom_dir)
        
        # Verify that cwd was passed to subprocess
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert 'cwd' in call_args.kwargs or 'cwd' in call_args.args[1]
    
    def test_execute_with_environment_variables(self, executor_mock):
        """Test command execution with custom environment"""
        executor, mock_popen = executor_mock
        
        mock_process = Mock()
        mock_process.communicate.return_value = ("output", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        custom_env = {"TEST_VAR": "test_value", "PATH": "/custom/path"}
        result = executor.execute("echo $TEST_VAR", environment_vars=custom_env)
        
        # Verify that env was passed to subprocess
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        assert 'env' in call_args.kwargs or 'env' in call_args.args[1]


class TestExecutionModes:
    """Test different execution modes"""
    
    @pytest.fixture
    def executor(self, posix_adapter):
        """Create executor for testing"""
        return CommandExecutor(posix_adapter)
    
    def test_synchronous_execution(self, executor):
        """Test synchronous execution mode"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("sync output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            config = ExecutionConfig(mode=ExecutionMode.SYNCHRONOUS)
            result = executor.execute("echo 'sync'", config)
            
            assert result.success is True
            # Synchronous execution should complete before returning
            mock_process.communicate.assert_called_once()
    
    def test_interactive_execution(self, executor):
        """Test interactive execution mode"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("interactive output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            config = ExecutionConfig(
                mode=ExecutionMode.INTERACTIVE,
                stdin="test input"
            )
            result = executor.execute_interactive("cat", stdin="test input")
            
            assert result.success is True
            # Interactive execution should pass stdin
            call_args = mock_popen.call_args
            assert 'stdin' in call_args.kwargs or 'stdin' in call_args.args[1]
    
    def test_background_execution(self, executor):
        """Test background execution mode"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("", "")
            mock_process.returncode = 0
            mock_process.pid = 9999
            mock_popen.return_value = mock_process
            
            config = ExecutionConfig(mode=ExecutionMode.BACKGROUND)
            result = executor.execute_background("sleep 10")
            
            assert result.success is True
            # Background execution should return immediately
            mock_process.communicate.assert_not_called()  # No immediate communication
    
    def test_async_execution(self, executor):
        """Test asynchronous execution mode"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("async output", "")
            mock_process.returncode = 0
            mock_process.pid = 8888
            mock_popen.return_value = mock_process
            
            config = ExecutionConfig(mode=ExecutionMode.ASYNC)
            result = executor.execute("long_running_command", config)
            
            assert result.success is True
            # Async execution should return quickly with PID
            assert result.pid == 8888


class TestProcessManagement:
    """Test process management functionality"""
    
    @pytest.fixture
    def executor(self, posix_adapter):
        """Create executor for testing"""
        return CommandExecutor(posix_adapter)
    
    def test_terminate_process(self, executor):
        """Test process termination"""
        with patch('tool_usage.command_executor.subprocess.run') as mock_run:
            # Mock successful termination
            mock_run.return_value = Mock(returncode=0)
            
            success = executor.terminate_process(1234)
            assert success is True
            
            # Verify that kill command was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert 'kill' in call_args or 'taskkill' in call_args
    
    def test_terminate_nonexistent_process(self, executor):
        """Test termination of nonexistent process"""
        with patch('tool_usage.command_executor.subprocess.run') as mock_run:
            # Mock failure to terminate
            mock_run.side_effect = subprocess.CalledProcessError(1, 'kill')
            
            success = executor.terminate_process(9999)
            assert success is False
    
    def test_get_process_status(self, executor):
        """Test getting process status"""
        # Mock process status retrieval
        with patch('tool_usage.command_executor.psutil.Process') as mock_process_class:
            mock_process = Mock()
            mock_process.name.return_value = "python"
            mock_process.status.return_value = "running"
            mock_process.cpu_percent.return_value = 10.5
            mock_process.memory_info.return_value = Mock(rss=1024000)
            mock_process_class.return_value = mock_process
            
            status = executor.get_process_status(1234)
            
            assert status is not None
            assert status['name'] == "python"
            assert status['status'] == "running"
            assert status['cpu_percent'] == 10.5
            assert status['memory_mb'] == 1.024  # 1024000 bytes / 1024 / 1024
    
    def test_get_process_status_nonexistent(self, executor):
        """Test getting status of nonexistent process"""
        with patch('tool_usage.command_executor.psutil.Process') as mock_process_class:
            mock_process_class.side_effect = Exception("Process not found")
            
            status = executor.get_process_status(9999)
            assert status is None


class TestExecutionHistory:
    """Test execution history tracking"""
    
    @pytest.fixture
    def executor(self, posix_adapter):
        """Create executor for testing"""
        return CommandExecutor(posix_adapter)
    
    def test_history_tracking(self, executor):
        """Test that execution history is tracked"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("test", "")
            mock_process.returncode = 0
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            # Execute multiple commands
            executor.execute("echo 'test1'")
            executor.execute("echo 'test2'")
            executor.execute("echo 'test3'")
            
            # Check history
            history = executor.get_execution_history()
            assert len(history) == 3
            assert history[0].command == "echo 'test1'"
            assert history[1].command == "echo 'test2'"
            assert history[2].command == "echo 'test3'"
    
    def test_history_limit(self, executor):
        """Test execution history size limit"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("test", "")
            mock_process.returncode = 0
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            # Execute more commands than the limit
            for i in range(100):
                executor.execute(f"echo 'test{i}'")
            
            # History should be limited
            history = executor.get_execution_history()
            assert len(history) <= executor.max_history_size
    
    def test_clear_history(self, executor):
        """Test clearing execution history"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("test", "")
            mock_process.returncode = 0
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            # Execute commands
            executor.execute("echo 'test1'")
            executor.execute("echo 'test2'")
            
            assert len(executor.execution_history) == 2
            
            # Clear history
            executor.clear_history()
            
            assert len(executor.execution_history) == 0


class TestSecurityValidation:
    """Test security validation during execution"""
    
    @pytest.fixture
    def executor(self, posix_adapter):
        """Create executor with security validator"""
        return CommandExecutor(posix_adapter)
    
    def test_security_validation_enabled(self, executor):
        """Test that security validation is enabled by default"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("safe", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            # Execute safe command
            result = executor.execute("echo 'safe command'")
            
            assert result.success is True
            mock_popen.assert_called_once()
    
    def test_security_validation_dangerous_command(self, executor):
        """Test that dangerous commands are blocked"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            # Execute dangerous command
            result = executor.execute("rm -rf /")
            
            # Should fail security validation
            assert result.success is False
            assert "security" in result.stderr.lower() or "blocked" in result.stderr.lower()
            # subprocess.Popen should not be called
            mock_popen.assert_not_called()
    
    def test_security_level_low(self, executor):
        """Test execution with low security level"""
        config = ExecutionConfig(security_level=SecurityLevel.LOW)
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            result = executor.execute("echo 'test'", config)
            
            assert result.success is True
            mock_popen.assert_called_once()
    
    def test_security_level_paranoid(self, executor):
        """Test execution with paranoid security level"""
        config = ExecutionConfig(security_level=SecurityLevel.PARANOID)
        
        # Even safe commands might be blocked at paranoid level
        with patch.object(executor.security_validator, 'validate_command', return_value=False):
            result = executor.execute("echo 'test'", config)
            
            assert result.success is False
            assert "security" in result.stderr.lower()


class TestCrossPlatformCompatibility:
    """Test cross-platform execution compatibility"""
    
    @pytest.mark.compatibility
    def test_posix_command_execution(self, mock_platform_linux, posix_adapter):
        """Test command execution on POSIX systems"""
        executor = CommandExecutor(posix_adapter)
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("posix output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            result = executor.execute("ls -la")
            
            assert result.success is True
            assert "posix output" in result.stdout
    
    @pytest.mark.compatibility
    def test_windows_command_execution(self, mock_platform_windows, windows_adapter):
        """Test command execution on Windows systems"""
        executor = CommandExecutor(windows_adapter)
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("windows output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            result = executor.execute("dir")
            
            assert result.success is True
            assert "windows output" in result.stdout
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("command,expected_platform", [
        ("ls -la", "posix"),
        ("dir", "windows"),
        ("echo 'test'", "both"),
        ("pwd", "posix"),
        ("cd", "both")
    ])
    def test_platform_specific_commands(self, command, expected_platform):
        """Test platform-specific command handling"""
        current_platform = "posix" if os.name == "posix" else "windows"
        
        # Skip test if platform doesn't match expected
        if expected_platform != "both" and expected_platform != current_platform:
            pytest.skip(f"Command {command} not supported on {current_platform}")
        
        # Test execution
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("output", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            executor = CommandExecutor()
            result = executor.execute(command)
            
            assert result.success is True


class TestPerformanceAndOptimization:
    """Test performance characteristics and optimization"""
    
    @pytest.mark.performance
    def test_execution_performance(self):
        """Test command execution performance"""
        executor = CommandExecutor()
        
        # Time simple command execution
        start_time = time.time()
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("fast", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            result = executor.execute("echo 'performance test'")
            end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Should execute quickly (mocked execution should be very fast)
        assert execution_time < 1.0
        assert result.success is True
    
    @pytest.mark.performance
    def test_history_performance(self):
        """Test performance impact of history tracking"""
        executor = CommandExecutor()
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("test", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            # Time multiple executions with history
            start_time = time.time()
            for i in range(100):
                executor.execute(f"echo 'test{i}'")
            end_time = time.time()
            
            execution_time = end_time - start_time
            
            # Should handle many executions efficiently
            assert execution_time < 10.0
            assert len(executor.execution_history) == 100
    
    @pytest.mark.performance
    def test_concurrent_execution_simulation(self):
        """Test simulated concurrent execution"""
        executor = CommandExecutor()
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("concurrent", "")
            mock_process.returncode = 0
            mock_popen.return_value = mock_process
            
            # Simulate concurrent execution
            threads = []
            for i in range(10):
                thread = threading.Thread(target=lambda: executor.execute(f"echo 'thread{i}'"))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # All executions should complete
            assert len(executor.execution_history) == 10


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience"""
    
    @pytest.fixture
    def executor(self, posix_adapter):
        """Create executor for testing"""
        return CommandExecutor(posix_adapter)
    
    def test_subprocess_error_handling(self, executor):
        """Test handling of subprocess errors"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_popen.side_effect = subprocess.CalledProcessError(1, "test_command")
            
            result = executor.execute("test_command")
            
            assert result.success is False
            assert result.return_code == 1
            assert "CalledProcessError" in result.stderr
    
    def test_os_error_handling(self, executor):
        """Test handling of OS errors"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_popen.side_effect = OSError("File not found")
            
            result = executor.execute("nonexistent_command")
            
            assert result.success is False
            assert "OSError" in result.stderr or "File not found" in result.stderr
    
    def test_memory_error_handling(self, executor):
        """Test handling of memory errors during execution"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.side_effect = MemoryError("Out of memory")
            mock_popen.return_value = mock_process
            
            result = executor.execute("memory_intensive_command")
            
            assert result.success is False
            assert "MemoryError" in result.stderr or "memory" in result.stderr.lower()
    
    def test_graceful_degradation(self, executor):
        """Test graceful degradation under adverse conditions"""
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            # Simulate various failures but still return a result
            mock_popen.side_effect = [
                OSError("Temporary failure"),
                Exception("Unknown error"),
                subprocess.TimeoutExpired("command", 5.0)
            ]
            
            # Should handle errors gracefully
            for i in range(3):
                result = executor.execute(f"command{i}")
                # Each should return a result, even if failed
                assert hasattr(result, 'success')
                assert hasattr(result, 'return_code')
                assert hasattr(result, 'stderr')


class TestCommandExecutorSummary:
    """Test summary and statistics functionality"""
    
    @pytest.fixture
    def executor(self, posix_adapter):
        """Create executor with some history"""
        executor = CommandExecutor(posix_adapter)
        
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("output", "")
            mock_process.returncode = 0
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            # Add some execution history
            executor.execute("echo 'test1'")
            executor.execute("echo 'test2'")
            executor.execute("echo 'test3'")
        
        return executor
    
    def test_execution_statistics(self, executor):
        """Test execution statistics generation"""
        # This would require implementing get_execution_statistics method
        # For now, we'll test the concept
        history = executor.get_execution_history()
        
        assert len(history) == 3
        assert all(hasattr(result, 'execution_time') for result in history)
    
    def test_performance_metrics(self, executor):
        """Test performance metrics collection"""
        history = executor.get_execution_history()
        
        # Calculate basic metrics
        total_executions = len(history)
        total_time = sum(result.execution_time for result in history if hasattr(result, 'execution_time'))
        success_count = sum(1 for result in history if result.success)
        
        assert total_executions > 0
        assert success_count <= total_executions
        assert total_time >= 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])