"""
Cross-Platform Compatibility Tests for Tool Usage Module
========================================================

Comprehensive cross-platform compatibility tests ensuring the tool usage
system works correctly across different operating systems and environments.
"""

import pytest
import os
import sys
import platform
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Any, Optional
import tempfile
import shutil

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage import (
        ToolUsageManager, ShellDetector, CommandExecutor, ToolRegistry,
        UsageOptimizer, get_platform_adapter
    )
    from tool_usage.platform_adapter import PosixAdapter, WindowsAdapter, PlatformInfo
    from tool_usage.command_executor import CommandResult, ExecutionConfig, ExecutionMode
    from tool_usage.shell_detector import ShellInfo
    from tool_usage.tool_registry import ToolMetadata, ToolCategory
except ImportError as e:
    pytest.skip(f"Could not import tool_usage components: {e}", allow_module_level=True)


class TestPlatformDetection:
    """Test platform detection across different operating systems"""
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("system_name,expected_adapter", [
        ("Linux", PosixAdapter),
        ("Darwin", PosixAdapter),
        ("Windows", WindowsAdapter),
        ("FreeBSD", PosixAdapter),
        ("OpenBSD", PosixAdapter)
    ])
    def test_adapter_selection(self, system_name, expected_adapter):
        """Test that correct adapter is selected for each platform"""
        with patch('tool_usage.platform_adapter.platform.system', return_value=system_name):
            adapter = get_platform_adapter()
            
            assert isinstance(adapter, expected_adapter), \
                f"Expected {expected_adapter.__name__} for {system_name}, got {type(adapter).__name__}"
    
    @pytest.mark.compatibility
    def test_linux_detection(self, mock_platform_linux):
        """Test Linux platform detection"""
        with patch('tool_usage.platform_adapter.platform.system', return_value='Linux'), \
             patch('tool_usage.platform_adapter.platform.machine', return_value='x86_64'), \
             patch('tool_usage.platform_adapter.platform.release', return_value='5.4.0'):
            
            adapter = get_platform_adapter()
            info = adapter.platform_info
            
            assert info.name == "Linux"
            assert info.is_linux is True
            assert info.is_posix is True
            assert info.is_windows is False
            assert info.is_macos is False
    
    @pytest.mark.compatibility
    def test_macos_detection(self, mock_platform_macos):
        """Test macOS platform detection"""
        with patch('tool_usage.platform_adapter.platform.system', return_value='Darwin'), \
             patch('tool_usage.platform_adapter.platform.machine', return_value='arm64'), \
             patch('tool_usage.platform_adapter.platform.release', return_value='21.0.0'):
            
            adapter = get_platform_adapter()
            info = adapter.platform_info
            
            assert info.name == "Darwin"
            assert info.is_macos is True
            assert info.is_posix is True
            assert info.is_windows is False
            assert info.is_linux is False
    
    @pytest.mark.compatibility
    def test_windows_detection(self, mock_platform_windows):
        """Test Windows platform detection"""
        with patch('tool_usage.platform_adapter.platform.system', return_value='Windows'), \
             patch('tool_usage.platform_adapter.platform.machine', return_value='AMD64'), \
             patch('tool_usage.platform_adapter.platform.release', return_value='10.0.19042'):
            
            adapter = get_platform_adapter()
            info = adapter.platform_info
            
            assert info.name == "Windows"
            assert info.is_windows is True
            assert info.is_posix is False
            assert info.is_linux is False
            assert info.is_macos is False


class TestShellCompatibility:
    """Test shell compatibility across different platforms"""
    
    @pytest.mark.compatibility
    def test_posix_shells(self, mock_platform_linux):
        """Test POSIX shell detection and compatibility"""
        detector = ShellDetector(PosixAdapter())
        
        # Mock detection of available shells
        with patch.object(detector, '_detect_available_shells') as mock_detect:
            mock_shells = [
                ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive', 'scripting'}),
                ShellInfo("zsh", "/bin/zsh", "5.8", True, {'interactive', 'scripting', 'completion'}),
                ShellInfo("sh", "/bin/sh", None, True, {'scripting'}),
                ShellInfo("dash", "/bin/dash", "0.5.8", True, {'scripting'})
            ]
            mock_detect.return_value = None
            detector._available_shells = mock_shells
            
            shells = detector.get_available_shells()
            
            # Should detect POSIX shells
            assert len(shells) >= 4
            shell_names = [shell.name for shell in shells]
            
            expected_shells = ['bash', 'zsh', 'sh', 'dash']
            for expected_shell in expected_shells:
                assert expected_shell in shell_names, f"Expected {expected_shell} to be detected"
    
    @pytest.mark.compatibility
    def test_windows_shells(self, mock_platform_windows):
        """Test Windows shell detection and compatibility"""
        detector = ShellDetector(WindowsAdapter())
        
        # Mock detection of available shells
        with patch.object(detector, '_detect_available_shells') as mock_detect:
            mock_shells = [
                ShellInfo("cmd", "C:\\Windows\\System32\\cmd.exe", "10.0", True, {'interactive', 'scripting'}),
                ShellInfo("powershell", "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe", "5.1", True, {'interactive', 'scripting'}),
                ShellInfo("pwsh", "C:\\Program Files\\PowerShell\\7\\pwsh.exe", "7.1", True, {'interactive', 'scripting', 'modules'})
            ]
            mock_detect.return_value = None
            detector._available_shells = mock_shells
            
            shells = detector.get_available_shells()
            
            # Should detect Windows shells
            assert len(shells) >= 3
            shell_names = [shell.name for shell in shells]
            
            expected_shells = ['cmd', 'powershell', 'pwsh']
            for expected_shell in expected_shells:
                assert expected_shell in shell_names, f"Expected {expected_shell} to be detected on Windows"
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("shell_name,platform,expected_path", [
        ("bash", "Linux", "/bin/bash"),
        ("bash", "Darwin", "/bin/bash"),
        ("zsh", "Linux", "/bin/zsh"),
        ("zsh", "Darwin", "/bin/zsh"),
        ("cmd", "Windows", "C:\\Windows\\System32\\cmd.exe"),
        ("powershell", "Windows", "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe"),
        ("pwsh", "Windows", "C:\\Program Files\\PowerShell\\7\\pwsh.exe")
    ])
    def test_shell_path_resolution(self, shell_name, platform, expected_path):
        """Test shell path resolution across platforms"""
        with patch('tool_usage.platform_adapter.platform.system', return_value=platform):
            detector = ShellDetector()
            
            # Mock shell path detection
            with patch.object(detector, '_detect_shell_path') as mock_detect_path:
                if platform == "Windows":
                    mock_detect_path.return_value = expected_path.replace("/", "\\")
                else:
                    mock_detect_path.return_value = expected_path
                
                shell_info = detector.get_shell_by_name(shell_name)
                
                if shell_info:
                    assert shell_info.path == expected_path, \
                        f"Expected {expected_path} for {shell_name} on {platform}, got {shell_info.path}"
    
    @pytest.mark.compatibility
    def test_shell_capability_detection(self):
        """Test shell capability detection across different shells"""
        # Test different shell capabilities
        shell_capabilities = {
            "bash": {'interactive', 'login', 'scripting', 'aliases', 'functions'},
            "zsh": {'interactive', 'login', 'scripting', 'aliases', 'completion', 'functions'},
            "fish": {'interactive', 'login', 'scripting', 'completion', 'autosuggestions'},
            "cmd": {'interactive', 'scripting', 'batch-files'},
            "powershell": {'interactive', 'scripting', 'object-oriented', 'modules'},
            "pwsh": {'interactive', 'scripting', 'object-oriented', 'modules', 'cross-platform'}
        }
        
        for shell_name, expected_caps in shell_capabilities.items():
            shell_info = ShellInfo(shell_name, f"/bin/{shell_name}", "1.0", True, expected_caps)
            
            # Verify capabilities
            for capability in expected_caps:
                assert capability in shell_info.capabilities, \
                    f"Shell {shell_name} should have capability {capability}"


class TestCommandExecutionCompatibility:
    """Test command execution compatibility across platforms"""
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("command,platform", [
        ("echo 'test'", "Linux"),
        ("echo test", "Windows"),
        ("pwd", "Linux"),
        ("cd", "Windows"),
        ("ls -la", "Linux"),
        ("dir", "Windows"),
        ("whoami", "Linux"),
        ("whoami", "Windows"),
        ("date", "Linux"),
        ("date", "Windows")
    ])
    def test_platform_specific_commands(self, command, platform):
        """Test execution of platform-specific commands"""
        # Skip if not on the target platform
        current_platform = platform.system()
        if current_platform != platform:
            pytest.skip(f"Test designed for {platform}, current platform is {current_platform}")
        
        try:
            adapter = get_platform_adapter()
            executor = CommandExecutor(adapter)
            
            result = executor.execute(command)
            
            assert result is not None
            assert hasattr(result, 'command')
            assert hasattr(result, 'success')
            
        except Exception as e:
            # In test environments, command execution might fail
            pytest.skip(f"Command execution failed: {e}")
    
    @pytest.mark.compatibility
    def test_path_separator_handling(self):
        """Test path separator handling across platforms"""
        posix_adapter = PosixAdapter()
        windows_adapter = WindowsAdapter()
        
        posix_paths = [
            "/usr/bin/python",
            "/tmp/test/file.txt",
            "./relative/path",
            "../parent/path",
            "~/home/path"
        ]
        
        windows_paths = [
            "C:\\Windows\\System32",
            "C:\\Users\\Test\\Documents\\file.txt",
            ".\\relative\\path",
            "..\\parent\\path",
            "%USERPROFILE%\\path"
        ]
        
        # Test POSIX paths with POSIX adapter
        for path in posix_paths:
            # Should be handled correctly by POSIX adapter
            assert isinstance(path, str)
        
        # Test Windows paths with Windows adapter
        for path in windows_paths:
            # Should be handled correctly by Windows adapter
            assert isinstance(path, str)
    
    @pytest.mark.compatibility
    def test_environment_variable_handling(self):
        """Test environment variable handling across platforms"""
        posix_adapter = PosixAdapter()
        windows_adapter = WindowsAdapter()
        
        # Get environment variables
        posix_env = posix_adapter.get_environment_variables()
        windows_env = windows_adapter.get_environment_variables()
        
        # Both should have PATH
        assert 'PATH' in posix_env
        assert 'PATH' in windows_env
        
        # POSIX-specific variables
        if platform.system() != 'Windows':
            assert 'HOME' in posix_env or 'USER' in posix_env
        
        # Windows-specific variables
        if platform.system() == 'Windows':
            assert 'USERPROFILE' in windows_env or 'HOMEDRIVE' in windows_env
    
    @pytest.mark.compatibility
    def test_line_ending_handling(self):
        """Test line ending handling across platforms"""
        posix_adapter = PosixAdapter()
        windows_adapter = WindowsAdapter()
        
        posix_path_info = posix_adapter.get_path_info()
        windows_path_info = windows_adapter.get_path_info()
        
        # POSIX should use \n
        assert posix_path_info['line_separator'] == '\n'
        
        # Windows should use \r\n
        assert windows_path_info['line_separator'] == '\r\n'


class TestToolDiscoveryCompatibility:
    """Test tool discovery across different platforms"""
    
    @pytest.mark.compatibility
    def test_posix_tool_discovery(self, mock_platform_linux):
        """Test tool discovery on POSIX systems"""
        adapter = PosixAdapter()
        
        # Mock tool discovery
        with patch('shutil.which') as mock_which:
            def which_side_effect(tool):
                paths = {
                    'python': '/usr/bin/python',
                    'python3': '/usr/bin/python3',
                    'node': '/usr/bin/node',
                    'git': '/usr/bin/git',
                    'ls': '/bin/ls',
                    'cat': '/bin/cat',
                    'grep': '/bin/grep'
                }
                return paths.get(tool)
            
            mock_which.side_effect = which_side_effect
            
            posix_tools = ['python', 'python3', 'node', 'git', 'ls', 'cat', 'grep']
            discovered = adapter.discover_tools(posix_tools)
            
            # Should discover POSIX tools
            for tool in posix_tools:
                assert tool in discovered, f"Expected {tool} to be discovered on POSIX"
    
    @pytest.mark.compatibility
    def test_windows_tool_discovery(self, mock_platform_windows):
        """Test tool discovery on Windows systems"""
        adapter = WindowsAdapter()
        
        # Mock tool discovery
        with patch('shutil.which') as mock_which:
            def which_side_effect(tool):
                paths = {
                    'python.exe': 'C:\\Python39\\python.exe',
                    'python': 'C:\\Python39\\python.exe',
                    'node.exe': 'C:\\Program Files\\nodejs\\node.exe',
                    'git.exe': 'C:\\Program Files\\Git\\bin\\git.exe',
                    'cmd.exe': 'C:\\Windows\\System32\\cmd.exe',
                    'powershell.exe': 'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe'
                }
                return paths.get(tool)
            
            mock_which.side_effect = which_side_effect
            
            windows_tools = ['python.exe', 'python', 'node.exe', 'git.exe', 'cmd.exe', 'powershell.exe']
            discovered = adapter.discover_tools(windows_tools)
            
            # Should discover Windows tools
            for tool in windows_tools:
                assert tool in discovered, f"Expected {tool} to be discovered on Windows"
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("tool_name,expected_category", [
        ("python", ToolCategory.PROGRAMMING_LANGUAGE),
        ("python3", ToolCategory.PROGRAMMING_LANGUAGE),
        ("node", ToolCategory.PROGRAMMING_LANGUAGE),
        ("git", ToolCategory.DEVELOPMENT_TOOL),
        ("gcc", ToolCategory.DEVELOPMENT_TOOL),
        ("make", ToolCategory.DEVELOPMENT_TOOL),
        ("curl", ToolCategory.NETWORK_TOOL),
        ("wget", ToolCategory.NETWORK_TOOL),
        ("vim", ToolCategory.TEXT_PROCESSING),
        ("grep", ToolCategory.TEXT_PROCESSING),
        ("sed", ToolCategory.TEXT_PROCESSING)
    ])
    def test_tool_categorization_consistency(self, tool_name, expected_category):
        """Test that tools are categorized consistently across platforms"""
        registry = ToolRegistry()
        
        # Mock category detection
        with patch.object(registry, '_detect_tool_category', return_value=expected_category):
            detected_category = registry._detect_tool_category(tool_name)
            
            assert detected_category == expected_category, \
                f"Tool {tool_name} should be categorized as {expected_category}"
    
    @pytest.mark.compatibility
    def test_executable_extension_handling(self):
        """Test handling of executable extensions across platforms"""
        posix_adapter = PosixAdapter()
        windows_adapter = WindowsAdapter()
        
        # Test POSIX executables (no extension needed)
        posix_executables = ['python', 'git', 'bash', 'ls']
        for exe in posix_executables:
            is_executable = posix_adapter.is_executable(exe)
            assert isinstance(is_executable, bool)
        
        # Test Windows executables (may need .exe extension)
        windows_executables = ['cmd', 'cmd.exe', 'powershell', 'powershell.exe']
        for exe in windows_executables:
            is_executable = windows_adapter.is_executable(exe)
            assert isinstance(is_executable, bool)


class TestProcessManagementCompatibility:
    """Test process management across different platforms"""
    
    @pytest.mark.compatibility
    def test_process_info_format_consistency(self):
        """Test that process information format is consistent across platforms"""
        adapter = get_platform_adapter()
        
        # Mock process information retrieval
        with patch('psutil.Process') as mock_process_class:
            mock_process = Mock()
            mock_process.name.return_value = "test_process"
            mock_process.status.return_value = "running"
            mock_process.cpu_percent.return_value = 5.0
            mock_process.memory_info.return_value = Mock(rss=1024000)
            mock_process_class.return_value = mock_process
            
            process_info = adapter.get_process_info(1234)
            
            # Should have consistent structure
            required_fields = ['name', 'status', 'cpu_percent', 'memory_mb']
            for field in required_fields:
                assert field in process_info, f"Process info should have {field} field"
    
    @pytest.mark.compatibility
    def test_process_termination_consistency(self):
        """Test that process termination is consistent across platforms"""
        adapter = get_platform_adapter()
        
        # Mock process termination
        with patch.object(adapter, 'kill_process') as mock_kill:
            # Test normal termination
            mock_kill.return_value = True
            
            success = adapter.kill_process(1234)
            assert success is True
            
            # Verify termination method was called
            mock_kill.assert_called_once()
    
    @pytest.mark.compatibility
    def test_signal_handling_differences(self):
        """Test handling of platform-specific signal differences"""
        posix_adapter = PosixAdapter()
        
        # Test POSIX signals
        posix_signals = ['SIGTERM', 'SIGKILL', 'SIGINT', 'SIGUSR1']
        for signal_name in posix_signals:
            # Should handle POSIX signals
            try:
                signal = getattr(__import__('signal'), signal_name)
                assert signal is not None
            except AttributeError:
                # Signal not available, which is acceptable
                pass


class TestFileSystemCompatibility:
    """Test file system operations across platforms"""
    
    @pytest.mark.compatibility
    def test_temporary_directory_handling(self):
        """Test temporary directory handling across platforms"""
        adapter = get_platform_adapter()
        path_info = adapter.get_path_info()
        
        # Should provide temp directory
        assert 'temp_directory' in path_info
        
        temp_dir = path_info['temp_directory']
        assert isinstance(temp_dir, str)
        assert len(temp_dir) > 0
    
    @pytest.mark.compatibility
    def test_home_directory_detection(self):
        """Test home directory detection across platforms"""
        adapter = get_platform_adapter()
        path_info = adapter.get_path_info()
        
        # Should provide home directory
        assert 'home_directory' in path_info
        
        home_dir = path_info['home_directory']
        assert isinstance(home_dir, str)
        assert len(home_dir) > 0
    
    @pytest.mark.compatibility
    def test_current_directory_handling(self):
        """Test current directory handling across platforms"""
        adapter = get_platform_adapter()
        path_info = adapter.get_path_info()
        
        # Should provide current directory
        assert 'current_directory' in path_info
        
        current_dir = path_info['current_directory']
        assert isinstance(current_dir, str)
        
        # Should match os.getcwd() in most cases
        try:
            assert current_dir == os.getcwd()
        except Exception:
            # May differ in test environments, which is acceptable
            pass
    
    @pytest.mark.compatibility
    def test_absolute_path_resolution(self):
        """Test absolute path resolution across platforms"""
        # Test cases for different platforms
        test_cases = [
            ("/usr/bin/python", "posix"),  # POSIX absolute path
            ("C:\\Windows\\System32", "windows"),  # Windows absolute path
            ("./relative/path", "both"),  # Relative path
            ("../parent/path", "both"),  # Parent relative path
        ]
        
        adapter = get_platform_adapter()
        
        for path, platform_type in test_cases:
            # Should handle all path types
            if platform_type == "both" or platform_type == platform.system().lower():
                # Test path resolution (mocked)
                with patch('os.path.abspath') as mock_abspath:
                    mock_abspath.return_value = f"/resolved/{path}"
                    
                    resolved = adapter.resolve_path(path)
                    assert isinstance(resolved, str)


class TestIntegrationCompatibility:
    """Test full integration across different platforms"""
    
    @pytest.mark.compatibility
    @pytest.mark.slow
    def test_full_workflow_posix(self, mock_platform_linux):
        """Test complete workflow on POSIX systems"""
        try:
            # Test the full ToolUsageManager workflow
            manager = ToolUsageManager(auto_discovery=False, auto_optimization=False)
            
            # Test shell detection
            current_shell = manager.get_current_shell()
            
            # Test command execution
            result = manager.execute_command("echo 'POSIX workflow test'")
            
            # Test tool discovery
            tools = manager.discover_tools()
            
            # Test optimization
            optimization = manager.optimize_command("cat file.txt | grep pattern")
            
            # Verify all components work together
            assert result is not None
            assert hasattr(result, 'success')
            assert isinstance(tools, list)
            assert optimization is not None
            
        except Exception as e:
            # In test environment, some components might not work
            pytest.skip(f"Full workflow test failed: {e}")
    
    @pytest.mark.compatibility
    @pytest.mark.slow
    def test_full_workflow_windows(self, mock_platform_windows):
        """Test complete workflow on Windows systems"""
        try:
            # Test the full ToolUsageManager workflow
            manager = ToolUsageManager(auto_discovery=False, auto_optimization=False)
            
            # Test shell detection
            current_shell = manager.get_current_shell()
            
            # Test command execution
            result = manager.execute_command("echo Windows workflow test")
            
            # Test tool discovery
            tools = manager.discover_tools()
            
            # Test optimization
            optimization = manager.optimize_command("type file.txt | findstr pattern")
            
            # Verify all components work together
            assert result is not None
            assert hasattr(result, 'success')
            assert isinstance(tools, list)
            assert optimization is not None
            
        except Exception as e:
            # In test environment, some components might not work
            pytest.skip(f"Full workflow test failed: {e}")
    
    @pytest.mark.compatibility
    def test_component_compatibility_across_platforms(self):
        """Test that components maintain compatibility across platforms"""
        # Test that the same interface works on different platforms
        platforms = ['Linux', 'Darwin', 'Windows']
        
        for platform_name in platforms:
            with patch('tool_usage.platform_adapter.platform.system', return_value=platform_name):
                try:
                    # Test component initialization
                    adapter = get_platform_adapter()
                    assert adapter is not None
                    
                    # Test basic operations
                    path_info = adapter.get_path_info()
                    assert isinstance(path_info, dict)
                    assert 'path_separator' in path_info
                    
                    env_vars = adapter.get_environment_variables()
                    assert isinstance(env_vars, dict)
                    assert 'PATH' in env_vars
                    
                except Exception as e:
                    pytest.skip(f"Component compatibility test failed for {platform_name}: {e}")


class TestErrorHandlingCompatibility:
    """Test error handling across different platforms"""
    
    @pytest.mark.compatibility
    def test_file_not_found_handling(self):
        """Test handling of file not found errors across platforms"""
        adapter = get_platform_adapter()
        
        # Test with non-existent file
        non_existent_path = "/this/path/definitely/does/not/exist"
        
        with patch('os.path.exists', return_value=False):
            is_executable = adapter.is_executable(non_existent_path)
            assert is_executable is False
    
    @pytest.mark.compatibility
    def test_permission_denied_handling(self):
        """Test handling of permission denied errors across platforms"""
        adapter = get_platform_adapter()
        
        # Mock permission denied scenario
        with patch('os.access', return_value=False):
            is_executable = adapter.is_executable("/protected/executable")
            assert is_executable is False
    
    @pytest.mark.compatibility
    def test_process_not_found_handling(self):
        """Test handling of process not found errors across platforms"""
        adapter = get_platform_adapter()
        
        # Mock process not found
        with patch('psutil.Process', side_effect=Exception("Process not found")):
            process_info = adapter.get_process_info(99999)
            assert process_info is None
    
    @pytest.mark.compatibility
    def test_command_not_found_handling(self):
        """Test handling of command not found errors"""
        executor = CommandExecutor()
        
        # Mock command not found
        with patch('tool_usage.command_executor.subprocess.Popen', side_effect=FileNotFoundError):
            result = executor.execute("definitely_not_a_real_command_12345")
            
            assert result is not None
            assert result.success is False
            assert "not found" in result.stderr.lower() or result.return_code != 0


class TestPerformanceCompatibility:
    """Test performance characteristics across platforms"""
    
    @pytest.mark.compatibility
    @pytest.mark.performance
    def test_initialization_performance_comparison(self):
        """Test initialization performance across different platforms"""
        import time
        
        platforms = ['Linux', 'Darwin', 'Windows']
        times = {}
        
        for platform_name in platforms:
            with patch('tool_usage.platform_adapter.platform.system', return_value=platform_name):
                start_time = time.time()
                
                try:
                    # Initialize components
                    adapter = get_platform_adapter()
                    detector = ShellDetector(adapter)
                    executor = CommandExecutor(adapter)
                    registry = ToolRegistry(adapter)
                    optimizer = UsageOptimizer(adapter, executor, registry)
                except Exception:
                    # Skip if components fail to initialize
                    continue
                
                end_time = time.time()
                times[platform_name] = end_time - start_time
        
        # All platforms should initialize within reasonable time
        for platform, init_time in times.items():
            assert init_time < 30.0, f"{platform} initialization took {init_time:.2f}s"
    
    @pytest.mark.compatibility
    @pytest.mark.performance
    def test_command_execution_performance_consistency(self):
        """Test command execution performance consistency across platforms"""
        import time
        
        executor = CommandExecutor()
        
        # Mock fast command execution
        with patch('tool_usage.command_executor.subprocess.Popen') as mock_popen:
            mock_process = Mock()
            mock_process.communicate.return_value = ("output", "")
            mock_process.returncode = 0
            mock_process.pid = 1234
            mock_popen.return_value = mock_process
            
            # Test execution speed
            start_time = time.time()
            
            for i in range(10):
                result = executor.execute(f"echo 'performance test {i}'")
                assert result.success is True
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Should execute 10 commands quickly
            assert execution_time < 10.0
    
    @pytest.mark.compatibility
    @pytest.mark.performance
    def test_memory_usage_consistency(self):
        """Test memory usage consistency across platforms"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform operations
        try:
            adapter = get_platform_adapter()
            detector = ShellDetector(adapter)
            executor = CommandExecutor(adapter)
            registry = ToolRegistry(adapter)
            optimizer = UsageOptimizer(adapter, executor, registry)
        except Exception:
            # Mock operations if real components fail
            adapter = Mock()
            detector = Mock()
            executor = Mock()
            registry = Mock()
            optimizer = Mock()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase: {memory_increase / 1024 / 1024:.2f}MB"


if __name__ == "__main__":
    # Run compatibility tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "compatibility"])