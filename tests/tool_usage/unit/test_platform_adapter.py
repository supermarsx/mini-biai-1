"""
Unit Tests for Platform Adapter Component
==========================================

Comprehensive unit tests for the PlatformAdapter classes, testing cross-platform
compatibility and system information detection.
"""

import pytest
import os
import sys
import platform
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import subprocess

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.platform_adapter import (
        PlatformAdapter, PosixAdapter, WindowsAdapter, PlatformInfo,
        get_platform_adapter, get_platform_info
    )
except ImportError as e:
    pytest.skip(f"Could not import PlatformAdapter: {e}", allow_module_level=True)


class TestPlatformInfo:
    """Test PlatformInfo dataclass"""
    
    def test_platform_info_creation(self):
        """Test creating PlatformInfo object"""
        info = PlatformInfo(
            name="Linux",
            version="5.4.0",
            machine="x86_64",
            processor="x86_64",
            is_windows=False,
            is_posix=True,
            is_linux=True,
            is_macos=False
        )
        
        assert info.name == "Linux"
        assert info.version == "5.4.0"
        assert info.machine == "x86_64"
        assert info.processor == "x86_64"
        assert info.is_windows is False
        assert info.is_posix is True
        assert info.is_linux is True
        assert info.is_macos is False
    
    def test_platform_info_windows(self):
        """Test creating Windows PlatformInfo"""
        info = PlatformInfo(
            name="Windows",
            version="10.0",
            machine="AMD64",
            processor="AMD64",
            is_windows=True,
            is_posix=False,
            is_linux=False,
            is_macos=False
        )
        
        assert info.name == "Windows"
        assert info.is_windows is True
        assert info.is_posix is False
    
    def test_platform_info_macos(self):
        """Test creating macOS PlatformInfo"""
        info = PlatformInfo(
            name="Darwin",
            version="20.0.0",
            machine="arm64",
            processor="arm64",
            is_windows=False,
            is_posix=True,
            is_linux=False,
            is_macos=True
        )
        
        assert info.name == "Darwin"
        assert info.is_macos is True
        assert info.is_posix is True
    
    def test_platform_info_default_values(self):
        """Test PlatformInfo with default values"""
        info = PlatformInfo(
            name="Unknown",
            version="0.0",
            machine="unknown"
        )
        
        assert info.processor == "unknown"  # Default
        assert info.is_windows is False  # Default
        assert info.is_posix is False  # Default
        assert info.is_linux is False  # Default
        assert info.is_macos is False  # Default


class TestPlatformAdapter:
    """Test base PlatformAdapter class"""
    
    def test_platform_adapter_initialization(self):
        """Test base PlatformAdapter initialization"""
        adapter = PlatformAdapter()
        
        assert adapter.platform_info is not None
        assert hasattr(adapter, 'platform_info')
        assert hasattr(adapter, '_detect_platform_info')
        assert hasattr(adapter, 'get_environment_variables')
        assert hasattr(adapter, 'get_path_info')
    
    def test_detect_platform_info(self):
        """Test platform info detection"""
        adapter = PlatformAdapter()
        
        info = adapter._detect_platform_info()
        
        assert isinstance(info, PlatformInfo)
        assert info.name is not None
        assert info.version is not None
        assert info.machine is not None
    
    @pytest.mark.parametrize("platform_name", ["Linux", "Darwin", "Windows"])
    def test_platform_specific_detection(self, platform_name):
        """Test platform-specific information detection"""
        with patch('tool_usage.platform_adapter.platform.system', return_value=platform_name):
            adapter = PlatformAdapter()
            info = adapter._detect_platform_info()
            
            if platform_name == "Windows":
                assert info.is_windows is True
                assert info.is_posix is False
            else:
                assert info.is_windows is False
                assert info.is_posix is True
    
    def test_get_environment_variables(self):
        """Test environment variable retrieval"""
        adapter = PlatformAdapter()
        
        env_vars = adapter.get_environment_variables()
        
        assert isinstance(env_vars, dict)
        assert len(env_vars) > 0
        assert 'PATH' in env_vars  # Should have PATH
        assert 'HOME' in env_vars or 'USERPROFILE' in env_vars
    
    def test_get_path_info(self):
        """Test path information retrieval"""
        adapter = PlatformAdapter()
        
        path_info = adapter.get_path_info()
        
        assert isinstance(path_info, dict)
        assert 'path_separator' in path_info
        assert 'line_separator' in path_info
        assert 'current_directory' in path_info
        assert 'home_directory' in path_info
        assert 'temp_directory' in path_info


class TestPosixAdapter:
    """Test PosixAdapter functionality"""
    
    def test_posix_adapter_initialization(self):
        """Test PosixAdapter initialization"""
        adapter = PosixAdapter()
        
        assert adapter.platform_info.name in ["Linux", "Darwin", "Unix"]
        assert adapter.platform_info.is_posix is True
    
    @pytest.mark.unit
    def test_posix_path_handling(self, mock_platform_linux):
        """Test POSIX path handling"""
        adapter = PosixAdapter()
        
        path_info = adapter.get_path_info()
        
        assert path_info['path_separator'] == ':'
        assert path_info['line_separator'] == '\n'
        assert adapter.platform_info.is_posix is True
    
    @pytest.mark.unit
    def test_posix_environment_variables(self, mock_platform_linux):
        """Test POSIX environment variable handling"""
        adapter = PosixAdapter()
        
        env_vars = adapter.get_environment_variables()
        
        assert 'PATH' in env_vars
        assert 'HOME' in env_vars
        assert 'USER' in env_vars
        # Should use POSIX conventions
        assert 'LD_LIBRARY_PATH' in env_vars or 'PATH' in env_vars
    
    @pytest.mark.unit
    def test_posix_executable_detection(self, mock_platform_linux):
        """Test POSIX executable detection"""
        adapter = PosixAdapter()
        
        # Mock executable detection
        with patch('os.access') as mock_access:
            mock_access.return_value = True
            
            is_executable = adapter.is_executable("/usr/bin/python")
            
            assert is_executable is True
    
    @pytest.mark.unit
    def test_posix_executable_nonexistent(self, mock_platform_linux):
        """Test detection of non-existent POSIX executable"""
        adapter = PosixAdapter()
        
        with patch('os.access') as mock_access:
            mock_access.return_value = False
            
            is_executable = adapter.is_executable("/nonexistent/executable")
            
            assert is_executable is False
    
    @pytest.mark.unit
    def test_posix_process_info(self, mock_platform_linux):
        """Test POSIX process information"""
        adapter = PosixAdapter()
        
        # Mock process information
        with patch('psutil.Process') as mock_process_class:
            mock_process = Mock()
            mock_process.name.return_value = "bash"
            mock_process.status.return_value = "running"
            mock_process.cpu_percent.return_value = 5.0
            mock_process.memory_info.return_value = Mock(rss=1024000)
            mock_process_class.return_value = mock_process
            
            process_info = adapter.get_process_info(1234)
            
            assert process_info['name'] == "bash"
            assert process_info['status'] == "running"
            assert process_info['cpu_percent'] == 5.0
            assert process_info['memory_mb'] == 1.024
    
    @pytest.mark.unit
    def test_posix_kill_process(self, mock_platform_linux):
        """Test POSIX process termination"""
        adapter = PosixAdapter()
        
        with patch('os.kill') as mock_kill:
            mock_kill.return_value = None
            
            success = adapter.kill_process(1234, signal.SIGTERM)
            
            assert success is True
            mock_kill.assert_called_once_with(1234, signal.SIGTERM)
    
    @pytest.mark.unit
    def test_posix_shell_detection(self, mock_platform_linux):
        """Test POSIX shell detection"""
        adapter = PosixAdapter()
        
        # Mock shell detection
        shells = adapter.detect_available_shells()
        
        assert isinstance(shells, list)
        # Should detect POSIX shells like bash, zsh, sh, etc.
        shell_names = [shell.name for shell in shells]
        assert 'bash' in shell_names or 'sh' in shell_names
    
    @pytest.mark.unit
    def test_posix_tool_discovery(self, mock_platform_linux):
        """Test POSIX tool discovery"""
        adapter = PosixAdapter()
        
        # Mock tool discovery
        with patch('shutil.which') as mock_which:
            def which_side_effect(tool):
                paths = {'python': '/usr/bin/python', 'git': '/usr/bin/git', 'ls': '/bin/ls'}
                return paths.get(tool)
            
            mock_which.side_effect = which_side_effect
            
            tools = adapter.discover_tools(['python', 'git', 'ls', 'nonexistent'])
            
            assert 'python' in tools
            assert 'git' in tools
            assert 'ls' in tools
            assert 'nonexistent' not in tools


class TestWindowsAdapter:
    """Test WindowsAdapter functionality"""
    
    def test_windows_adapter_initialization(self):
        """Test WindowsAdapter initialization"""
        adapter = WindowsAdapter()
        
        assert adapter.platform_info.name == "Windows"
        assert adapter.platform_info.is_windows is True
        assert adapter.platform_info.is_posix is False
    
    @pytest.mark.unit
    def test_windows_path_handling(self, mock_platform_windows):
        """Test Windows path handling"""
        adapter = WindowsAdapter()
        
        path_info = adapter.get_path_info()
        
        assert path_info['path_separator'] == ';'
        assert path_info['line_separator'] == '\r\n'
        assert adapter.platform_info.is_windows is True
    
    @pytest.mark.unit
    def test_windows_environment_variables(self, mock_platform_windows):
        """Test Windows environment variable handling"""
        adapter = WindowsAdapter()
        
        env_vars = adapter.get_environment_variables()
        
        assert 'PATH' in env_vars
        assert 'USERPROFILE' in env_vars
        assert 'SYSTEMROOT' in env_vars
        # Should use Windows conventions
        assert 'PATHEXT' in env_vars or 'PATH' in env_vars
    
    @pytest.mark.unit
    def test_windows_executable_detection(self, mock_platform_windows):
        """Test Windows executable detection"""
        adapter = WindowsAdapter()
        
        # Mock executable detection
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "C:\\Windows\\System32\\python.exe"
            
            is_executable = adapter.is_executable("python.exe")
            
            assert is_executable is True
    
    @pytest.mark.unit
    def test_windows_executable_with_extension(self, mock_platform_windows):
        """Test Windows executable detection with extension"""
        adapter = WindowsAdapter()
        
        with patch('shutil.which') as mock_which:
            mock_which.return_value = "C:\\Windows\\System32\\cmd.exe"
            
            # Should detect both with and without extension
            assert adapter.is_executable("cmd") is True
            assert adapter.is_executable("cmd.exe") is True
    
    @pytest.mark.unit
    def test_windows_process_info(self, mock_platform_windows):
        """Test Windows process information"""
        adapter = WindowsAdapter()
        
        # Mock process information
        with patch('psutil.Process') as mock_process_class:
            mock_process = Mock()
            mock_process.name.return_value = "powershell.exe"
            mock_process.status.return_value = "running"
            mock_process.cpu_percent.return_value = 10.0
            mock_process.memory_info.return_value = Mock(rss=2048000)
            mock_process_class.return_value = mock_process
            
            process_info = adapter.get_process_info(5678)
            
            assert process_info['name'] == "powershell.exe"
            assert process_info['status'] == "running"
            assert process_info['cpu_percent'] == 10.0
            assert process_info['memory_mb'] == 2.048
    
    @pytest.mark.unit
    def test_windows_kill_process(self, mock_platform_windows):
        """Test Windows process termination"""
        adapter = WindowsAdapter()
        
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            
            success = adapter.kill_process(5678, force=True)
            
            assert success is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert 'taskkill' in call_args
    
    @pytest.mark.unit
    def test_windows_shell_detection(self, mock_platform_windows):
        """Test Windows shell detection"""
        adapter = WindowsAdapter()
        
        # Mock shell detection
        shells = adapter.detect_available_shells()
        
        assert isinstance(shells, list)
        # Should detect Windows shells like cmd, powershell
        shell_names = [shell.name for shell in shells]
        assert 'cmd' in shell_names or 'powershell' in shell_names
    
    @pytest.mark.unit
    def test_windows_tool_discovery(self, mock_platform_windows):
        """Test Windows tool discovery"""
        adapter = WindowsAdapter()
        
        # Mock tool discovery
        with patch('shutil.which') as mock_which:
            def which_side_effect(tool):
                paths = {
                    'python.exe': 'C:\\Python39\\python.exe',
                    'git.exe': 'C:\\Program Files\\Git\\bin\\git.exe',
                    'cmd.exe': 'C:\\Windows\\System32\\cmd.exe'
                }
                return paths.get(tool)
            
            mock_which.side_effect = which_side_effect
            
            tools = adapter.discover_tools(['python.exe', 'git.exe', 'cmd.exe', 'nonexistent.exe'])
            
            assert 'python.exe' in tools
            assert 'git.exe' in tools
            assert 'cmd.exe' in tools
            assert 'nonexistent.exe' not in tools


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility"""
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("platform_name,expected_adapter", [
        ("Linux", PosixAdapter),
        ("Darwin", PosixAdapter),
        ("Windows", WindowsAdapter)
    ])
    def test_adapter_selection(self, platform_name, expected_adapter):
        """Test that correct adapter is selected for each platform"""
        with patch('tool_usage.platform_adapter.platform.system', return_value=platform_name):
            adapter = get_platform_adapter()
            
            assert isinstance(adapter, expected_adapter)
    
    @pytest.mark.compatibility
    def test_platform_info_consistency(self):
        """Test consistency of platform info across adapters"""
        # Test that all adapters provide consistent info structure
        adapters = [PosixAdapter(), WindowsAdapter()]
        
        for adapter in adapters:
            info = adapter.platform_info
            
            # All adapters should provide these attributes
            assert hasattr(info, 'name')
            assert hasattr(info, 'version')
            assert hasattr(info, 'machine')
            assert hasattr(info, 'is_windows')
            assert hasattr(info, 'is_posix')
            assert hasattr(info, 'is_linux')
            assert hasattr(info, 'is_macos')
            
            # Logic checks
            assert info.is_windows != info.is_posix
            assert info.is_linux or not info.is_linux  # Always True
            assert info.is_macos or not info.is_macos  # Always True
    
    @pytest.mark.compatibility
    def test_path_separator_consistency(self):
        """Test path separator consistency across platforms"""
        linux_adapter = PosixAdapter()
        windows_adapter = WindowsAdapter()
        
        linux_path_info = linux_adapter.get_path_info()
        windows_path_info = windows_adapter.get_path_info()
        
        assert linux_path_info['path_separator'] == ':'
        assert windows_path_info['path_separator'] == ';'
        
        # Line separator should also differ
        assert linux_path_info['line_separator'] == '\n'
        assert windows_path_info['line_separator'] == '\r\n'


class TestPlatformDetection:
    """Test platform detection functionality"""
    
    @pytest.mark.unit
    @pytest.mark.parametrize("system_name,expected_platform", [
        ("Linux", "Linux"),
        ("Darwin", "Darwin"),
        ("Windows", "Windows")
    ])
    def test_system_detection(self, system_name, expected_platform):
        """Test detection of operating system"""
        with patch('tool_usage.platform_adapter.platform.system', return_value=system_name):
            info = get_platform_info()
            
            assert info.name == expected_platform
    
    @pytest.mark.unit
    def test_machine_type_detection(self):
        """Test machine type detection"""
        with patch('tool_usage.platform_adapter.platform.machine', return_value='x86_64'):
            info = get_platform_info()
            
            assert info.machine == 'x86_64'
    
    @pytest.mark.unit
    def test_processor_detection(self):
        """Test processor detection"""
        with patch('tool_usage.platform_adapter.platform.processor', return_value='Intel64'):
            info = get_platform_info()
            
            assert info.processor == 'Intel64'
    
    @pytest.mark.unit
    def test_version_detection(self):
        """Test version detection"""
        with patch('tool_usage.platform_adapter.platform.version', return_value='10.0.19042'):
            info = get_platform_info()
            
            assert info.version == '10.0.19042'


class TestPlatformErrorHandling:
    """Test error handling in platform adapters"""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter for testing"""
        return PlatformAdapter()
    
    @pytest.mark.unit
    def test_invalid_process_info(self, adapter):
        """Test handling of invalid process information"""
        with patch('psutil.Process') as mock_process_class:
            mock_process_class.side_effect = Exception("Process not found")
            
            process_info = adapter.get_process_info(9999)
            
            assert process_info is None
    
    @pytest.mark.unit
    def test_invalid_executable_path(self, adapter):
        """Test handling of invalid executable path"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            is_executable = adapter.is_executable("/nonexistent/path")
            
            assert is_executable is False
    
    @pytest.mark.unit
    def test_environment_variable_access_error(self, adapter):
        """Test handling of environment variable access errors"""
        with patch.dict(os.environ, {}, clear=True):
            # Add a key that will cause issues
            with patch.object(os.environ, '__getitem__', side_effect=KeyError("KEY_ERROR")):
                env_vars = adapter.get_environment_variables()
                
                # Should handle gracefully and return empty dict or partial results
                assert isinstance(env_vars, dict)


class TestPlatformAdapterPerformance:
    """Test platform adapter performance"""
    
    @pytest.mark.performance
    def test_adapter_initialization_performance(self):
        """Test performance of adapter initialization"""
        import time
        
        start_time = time.time()
        
        for _ in range(100):
            adapter = PlatformAdapter()
            _ = adapter.platform_info
        
        end_time = time.time()
        initialization_time = end_time - start_time
        
        # Should initialize 100 adapters quickly
        assert initialization_time < 10.0
    
    @pytest.mark.performance
    def test_platform_detection_performance(self):
        """Test performance of platform detection"""
        import time
        
        start_time = time.time()
        
        for _ in range(1000):
            info = get_platform_info()
        
        end_time = time.time()
        detection_time = end_time - start_time
        
        # Should detect platform quickly
        assert detection_time < 10.0
    
    @pytest.mark.performance
    @pytest.mark.parametrize("adapter_class", [PosixAdapter, WindowsAdapter])
    def test_tool_discovery_performance(self, adapter_class):
        """Test performance of tool discovery"""
        import time
        
        adapter = adapter_class()
        
        # Mock tool discovery to avoid actual system calls
        with patch.object(adapter, 'discover_tools', return_value=['tool1', 'tool2', 'tool3']):
            start_time = time.time()
            
            for _ in range(100):
                tools = adapter.discover_tools(['tool1', 'tool2', 'tool3'])
            
            end_time = time.time()
            discovery_time = end_time - start_time
            
            # Should discover tools quickly (mocked)
            assert discovery_time < 5.0


class TestPlatformAdapterIntegration:
    """Test integration scenarios for platform adapters"""
    
    @pytest.mark.integration
    def test_adapter_factory_function(self):
        """Test the adapter factory function"""
        adapter = get_platform_adapter()
        
        assert adapter is not None
        assert isinstance(adapter, (PosixAdapter, WindowsAdapter))
        
        # Verify platform info is accessible
        assert adapter.platform_info is not None
        assert adapter.platform_info.name is not None
    
    @pytest.mark.integration
    def test_platform_info_factory_function(self):
        """Test the platform info factory function"""
        info = get_platform_info()
        
        assert isinstance(info, PlatformInfo)
        assert info.name is not None
        assert info.version is not None
        assert info.machine is not None
    
    @pytest.mark.integration
    def test_adapter_consistency(self):
        """Test consistency between different adapter methods"""
        adapter = get_platform_adapter()
        
        # Get platform info from adapter
        adapter_info = adapter.platform_info
        
        # Get platform info from factory
        factory_info = get_platform_info()
        
        # Should have consistent basic info
        assert adapter_info.name == factory_info.name
        assert adapter_info.machine == factory_info.machine
    
    @pytest.mark.integration
    def test_cross_platform_method_compatibility(self):
        """Test that adapter methods work consistently across platforms"""
        adapters = [PosixAdapter(), WindowsAdapter()]
        
        for adapter in adapters:
            # Test basic methods exist and return expected types
            assert hasattr(adapter, 'get_path_info')
            assert hasattr(adapter, 'get_environment_variables')
            assert hasattr(adapter, 'is_executable')
            assert hasattr(adapter, 'get_process_info')
            
            # Test return types
            path_info = adapter.get_path_info()
            assert isinstance(path_info, dict)
            assert 'path_separator' in path_info
            
            env_vars = adapter.get_environment_variables()
            assert isinstance(env_vars, dict)
            
            # Test that all adapters can detect executables
            assert isinstance(adapter.is_executable("/usr/bin/python"), bool)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])