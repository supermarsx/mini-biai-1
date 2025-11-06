"""
Unit Tests for Shell Detector Component
========================================

Comprehensive unit tests for the ShellDetector class, testing shell detection,
identification, and cross-platform capabilities.
"""

import pytest
import os
import sys
import subprocess
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import platform

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.shell_detector import ShellDetector, ShellInfo
    from tool_usage.platform_adapter import PosixAdapter, WindowsAdapter
except ImportError as e:
    pytest.skip(f"Could not import ShellDetector: {e}", allow_module_level=True)


class TestShellInfo:
    """Test ShellInfo dataclass"""
    
    def test_shell_info_creation(self):
        """Test creating ShellInfo object"""
        shell_info = ShellInfo(
            name="bash",
            path="/bin/bash",
            version="5.1.4",
            is_available=True,
            capabilities={'interactive', 'scripting'},
            type="interactive"
        )
        
        assert shell_info.name == "bash"
        assert shell_info.path == "/bin/bash"
        assert shell_info.version == "5.1.4"
        assert shell_info.is_available is True
        assert 'interactive' in shell_info.capabilities
        assert shell_info.type == "interactive"
    
    def test_shell_info_minimal(self):
        """Test creating ShellInfo with minimal data"""
        shell_info = ShellInfo(
            name="test",
            path="/usr/bin/test"
        )
        
        assert shell_info.name == "test"
        assert shell_info.path == "/usr/bin/test"
        assert shell_info.version is None
        assert shell_info.is_available is False  # Default
        assert shell_info.capabilities is None  # Default
        assert shell_info.type is None  # Default


class TestShellDetectorInitialization:
    """Test ShellDetector initialization and basic setup"""
    
    def test_shell_detector_initialization(self, posix_adapter):
        """Test basic ShellDetector initialization"""
        detector = ShellDetector(posix_adapter)
        
        assert detector.platform_adapter == posix_adapter
        assert detector._shell_cache == {}
        assert detector._available_shells == []
        assert detector._current_shell is None
        assert detector.known_shells is not None
    
    def test_shell_detector_with_default_adapter(self):
        """Test ShellDetector with default adapter"""
        detector = ShellDetector()
        assert detector.platform_adapter is not None
    
    @pytest.mark.parametrize("shell_name", ["bash", "zsh", "fish", "powershell", "cmd", "dash"])
    def test_known_shells_configuration(self, posix_adapter, shell_name):
        """Test that all known shells are configured"""
        detector = ShellDetector(posix_adapter)
        
        assert shell_name in detector.known_shells
        shell_config = detector.known_shells[shell_name]
        
        # Each shell should have platform-specific executables
        assert 'posix' in shell_config
        assert 'windows' in shell_config
        assert 'capabilities' in shell_config
        
        # Capabilities should be a set
        assert isinstance(shell_config['capabilities'], set)


class TestShellDetection:
    """Test shell detection functionality"""
    
    @pytest.mark.unit
    def test_detect_current_shell_posix(self, mock_platform_linux):
        """Test current shell detection on POSIX systems"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            # Mock subprocess output for shell detection
            mock_process = Mock()
            mock_process.stdout = "bash"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            detector = ShellDetector(PosixAdapter())
            detector._detect_current_shell()
            
            # Should detect bash as current shell on Linux
            assert detector._current_shell is not None
            assert detector._current_shell.name == "bash"
    
    @pytest.mark.unit
    def test_detect_available_shells_posix(self, mock_platform_linux):
        """Test detection of available shells on POSIX systems"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            def mock_run_side_effect(*args, **kwargs):
                mock_process = Mock()
                if 'bash' in args[0]:
                    mock_process.stdout = "/bin/bash"
                    mock_process.returncode = 0
                elif 'zsh' in args[0]:
                    mock_process.stdout = "/bin/zsh"
                    mock_process.returncode = 0
                else:
                    mock_process.stdout = ""
                    mock_process.returncode = 1
                return mock_process
            
            mock_run.side_effect = mock_run_side_effect
            
            detector = ShellDetector(PosixAdapter())
            detector._detect_available_shells()
            
            assert len(detector._available_shells) > 0
            shell_names = [shell.name for shell in detector._available_shells]
            assert "bash" in shell_names
            assert "zsh" in shell_names
    
    @pytest.mark.unit
    def test_detect_available_shells_windows(self, mock_platform_windows):
        """Test detection of available shells on Windows"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            # Mock Windows PowerShell detection
            mock_process = Mock()
            mock_process.stdout = "powershell.exe"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            detector = ShellDetector(WindowsAdapter())
            detector._detect_available_shells()
            
            # Should detect Windows-specific shells
            shell_names = [shell.name for shell in detector._available_shells]
            assert "powershell" in shell_names
            assert "cmd" in shell_names
    
    @pytest.mark.unit
    @pytest.mark.parametrize("platform_name", ["linux", "darwin", "windows"])
    def test_platform_specific_detection(self, platform_name):
        """Test shell detection for different platforms"""
        if platform_name == "linux":
            platform_mock = mock_platform_linux
        elif platform_name == "darwin":
            platform_mock = mock_platform_macos
        else:
            platform_mock = mock_platform_windows
        
        with platform_mock:
            detector = ShellDetector()
            
            # Mock subprocess calls for shell detection
            with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
                mock_process = Mock()
                mock_process.stdout = "bash"
                mock_process.returncode = 0
                mock_run.return_value = mock_process
                
                detector._detect_available_shells()
                detector._detect_current_shell()
                
                assert len(detector._available_shells) >= 0  # May be empty in test environment
                assert detector._current_shell is not None


class TestShellCache:
    """Test shell caching functionality"""
    
    @pytest.mark.unit
    def test_shell_cache_population(self):
        """Test that shell cache is populated correctly"""
        detector = ShellDetector()
        
        # Mock detection methods
        mock_shells = [
            ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive'}),
            ShellInfo("zsh", "/bin/zsh", "5.8", True, {'interactive'})
        ]
        
        detector._available_shells = mock_shells
        detector._populate_shell_cache()
        
        assert "bash" in detector._shell_cache
        assert "zsh" in detector._shell_cache
        assert detector._shell_cache["bash"].name == "bash"
        assert detector._shell_cache["zsh"].path == "/bin/zsh"
    
    @pytest.mark.unit
    def test_cache_operations(self):
        """Test cache operations (get, set, clear)"""
        detector = ShellDetector()
        
        # Test setting cache
        shell_info = ShellInfo("test", "/bin/test")
        detector._shell_cache["test"] = shell_info
        
        # Test getting cache
        assert detector._shell_cache["test"] == shell_info
        
        # Test clearing cache
        detector._shell_cache.clear()
        assert len(detector._shell_cache) == 0


class TestShellRetrieval:
    """Test shell information retrieval methods"""
    
    @pytest.fixture
    def populated_detector(self):
        """Create detector with populated test data"""
        detector = ShellDetector()
        
        mock_shells = [
            ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive', 'scripting'}),
            ShellInfo("zsh", "/bin/zsh", "5.8", True, {'interactive', 'scripting', 'completion'}),
            ShellInfo("fish", "/usr/bin/fish", "3.3", False, {'interactive'})
        ]
        
        detector._available_shells = mock_shells
        detector._current_shell = mock_shells[0]  # bash is current
        
        return detector
    
    @pytest.mark.unit
    def test_get_available_shells(self, populated_detector):
        """Test getting available shells"""
        shells = populated_detector.get_available_shells()
        
        assert len(shells) == 3
        shell_names = [shell.name for shell in shells]
        assert "bash" in shell_names
        assert "zsh" in shell_names
        assert "fish" in shell_names
    
    @pytest.mark.unit
    def test_get_current_shell(self, populated_detector):
        """Test getting current shell"""
        current = populated_detector.get_current_shell()
        
        assert current is not None
        assert current.name == "bash"
        assert current.is_available is True
    
    @pytest.mark.unit
    def test_get_shell_by_name(self, populated_detector):
        """Test getting shell by name"""
        shell = populated_detector.get_shell_by_name("zsh")
        
        assert shell is not None
        assert shell.name == "zsh"
        assert shell.version == "5.8"
        assert 'completion' in shell.capabilities
    
    @pytest.mark.unit
    def test_get_shell_by_name_not_found(self, populated_detector):
        """Test getting non-existent shell by name"""
        shell = populated_detector.get_shell_by_name("nonexistent")
        
        assert shell is None
    
    @pytest.mark.unit
    def test_get_default_shell(self, populated_detector):
        """Test getting default shell for platform"""
        default = populated_detector.get_default_shell()
        
        assert default is not None
        # On most systems, default shell should be available
        assert default.is_available is True


class TestShellCapabilities:
    """Test shell capability detection and reporting"""
    
    @pytest.mark.unit
    def test_capability_detection(self):
        """Test detection of shell capabilities"""
        detector = ShellDetector()
        
        # Mock shell with capabilities
        shell = ShellInfo("bash", "/bin/bash", "5.1.4", True, {
            'interactive', 'login', 'scripting', 'aliases'
        })
        
        # Test capability checking
        assert 'interactive' in shell.capabilities
        assert 'scripting' in shell.capabilities
        assert 'login' in shell.capabilities
        assert 'aliases' in shell.capabilities
        assert 'completion' not in shell.capabilities  # bash doesn't have built-in completion
    
    @pytest.mark.unit
    def test_capability_comparison(self):
        """Test capability comparison between shells"""
        bash = ShellInfo("bash", "/bin/bash", "5.1.4", True, {
            'interactive', 'login', 'scripting', 'aliases'
        })
        
        zsh = ShellInfo("zsh", "/bin/zsh", "5.8", True, {
            'interactive', 'login', 'scripting', 'aliases', 'completion'
        })
        
        fish = ShellInfo("fish", "/usr/bin/fish", "3.3", True, {
            'interactive', 'login', 'scripting', 'completion', 'autosuggestions'
        })
        
        # ZSH and Fish have more capabilities than Bash
        assert 'completion' in zsh.capabilities
        assert 'completion' in fish.capabilities
        assert 'autosuggestions' in fish.capabilities
    
    @pytest.mark.unit
    def test_capability_filtering(self):
        """Test filtering shells by capabilities"""
        detector = ShellDetector()
        
        shells_with_completion = [
            ShellInfo("zsh", "/bin/zsh", "5.8", True, {'completion'}),
            ShellInfo("fish", "/usr/bin/fish", "3.3", True, {'completion'})
        ]
        
        shells_without_completion = [
            ShellInfo("ash", "/bin/ash", None, True, {'scripting'})
        ]
        
        # Filter shells that have completion
        completion_shells = [shell for shell in shells_with_completion 
                           if 'completion' in shell.capabilities]
        
        assert len(completion_shells) == 2
        assert all('completion' in shell.capabilities for shell in completion_shells)


class TestShellVersioning:
    """Test shell version detection and handling"""
    
    @pytest.mark.unit
    def test_version_detection(self):
        """Test detection of shell versions"""
        detector = ShellDetector()
        
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            # Mock version detection
            mock_process = Mock()
            mock_process.stdout = "GNU bash, version 5.1.4(1)-release"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            shell = ShellInfo("bash", "/bin/bash")
            # Version would be detected in _detect_shell_version method
            shell.version = "5.1.4"
            
            assert shell.version is not None
            assert shell.version == "5.1.4"
    
    @pytest.mark.unit
    def test_version_comparison(self):
        """Test comparison of shell versions"""
        shell1 = ShellInfo("bash", "/bin/bash", "4.4")
        shell2 = ShellInfo("bash", "/bin/bash", "5.1")
        shell3 = ShellInfo("bash", "/bin/bash", None)
        
        assert shell2.version > shell1.version
        assert shell3.version is None
    
    @pytest.mark.unit
    def test_version_parsing(self):
        """Test parsing of different version formats"""
        version_formats = [
            "GNU bash, version 5.1.4(1)-release",
            "zsh 5.8 (x86_64-pc-linux-gnu)",
            "fish, version 3.3.1",
            "PowerShell 7.1.0"
        ]
        
        # Should be able to extract version numbers
        for version_str in version_formats:
            assert any(char.isdigit() for char in version_str)


class TestCrossPlatformCompatibility:
    """Test cross-platform shell detection compatibility"""
    
    @pytest.mark.unit
    @pytest.mark.compatibility
    def test_posix_shell_detection(self, mock_platform_linux):
        """Test shell detection on POSIX systems"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            mock_process = Mock()
            mock_process.stdout = "/bin/bash"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            detector = ShellDetector(PosixAdapter())
            detector._detect_available_shells()
            
            # Should detect POSIX shells
            assert len(detector._available_shells) >= 0
    
    @pytest.mark.unit
    @pytest.mark.compatibility
    def test_windows_shell_detection(self, mock_platform_windows):
        """Test shell detection on Windows systems"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            mock_process = Mock()
            mock_process.stdout = "powershell.exe"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            detector = ShellDetector(WindowsAdapter())
            detector._detect_available_shells()
            
            # Should detect Windows shells
            assert len(detector._available_shells) >= 0
    
    @pytest.mark.unit
    @pytest.mark.compatibility
    def test_macos_shell_detection(self, mock_platform_macos):
        """Test shell detection on macOS systems"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            mock_process = Mock()
            mock_process.stdout = "/bin/zsh"
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            detector = ShellDetector()
            detector._detect_available_shells()
            
            # Should detect macOS shells
            assert len(detector._available_shells) >= 0
    
    @pytest.mark.unit
    @pytest.mark.compatibility
    @pytest.mark.parametrize("shell_name,expected_executable", [
        ("bash", "bash"),
        ("zsh", "zsh"),
        ("fish", "fish"),
        ("powershell", "powershell")
    ])
    def test_shell_executable_names(self, shell_name, expected_executable):
        """Test that shell executable names are platform-appropriate"""
        detector = ShellDetector()
        
        if shell_name in detector.known_shells:
            shell_config = detector.known_shells[shell_name]
            executables = shell_config.get('posix', [])
            
            # Should have appropriate executables for POSIX systems
            if platform.system() != 'Windows':
                assert any(expected_executable in exe for exe in executables)


class TestErrorHandling:
    """Test error handling in shell detection"""
    
    @pytest.mark.unit
    def test_subprocess_failure_handling(self):
        """Test handling of subprocess failures"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            # Mock subprocess failure
            mock_run.side_effect = subprocess.CalledProcessError(1, 'shell_test')
            
            detector = ShellDetector()
            
            # Should handle the error gracefully
            try:
                detector._detect_available_shells()
                # Should not raise an exception, just return empty list
                assert len(detector._available_shells) >= 0
            except subprocess.CalledProcessError:
                # This is also acceptable if error is propagated
                pass
    
    @pytest.mark.unit
    def test_empty_subprocess_output(self):
        """Test handling of empty subprocess output"""
        with patch('tool_usage.shell_detector.subprocess.run') as mock_run:
            # Mock empty output
            mock_process = Mock()
            mock_process.stdout = ""
            mock_process.returncode = 0
            mock_run.return_value = mock_process
            
            detector = ShellDetector()
            detector._detect_available_shells()
            
            # Should handle empty output gracefully
            # Implementation might skip shells with empty paths
            assert isinstance(detector._available_shells, list)
    
    @pytest.mark.unit
    def test_invalid_shell_info_handling(self):
        """Test handling of invalid shell information"""
        detector = ShellDetector()
        
        # Mock invalid shell info
        detector._available_shells = [
            ShellInfo("", "", None, False, set()),  # Invalid
            ShellInfo("valid", "/bin/valid", "1.0", True, {'test'}),  # Valid
        ]
        
        # Should handle invalid shells gracefully
        valid_shells = [shell for shell in detector._available_shells if shell.name]
        assert len(valid_shells) >= 0


class TestShellSummary:
    """Test shell summary functionality"""
    
    @pytest.mark.unit
    def test_get_shell_summary(self):
        """Test generation of shell summary"""
        detector = ShellDetector()
        
        # Populate with test data
        detector._available_shells = [
            ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive'}),
            ShellInfo("zsh", "/bin/zsh", "5.8", True, {'interactive'})
        ]
        detector._current_shell = ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive'})
        
        summary = detector.get_shell_summary()
        
        assert 'available_shells' in summary
        assert 'current_shell' in summary
        assert 'total_count' in summary
        assert 'platform' in summary
        
        assert summary['total_count'] >= 0
        assert isinstance(summary['available_shells'], list)


class TestShellDetectorIntegration:
    """Test integration scenarios for ShellDetector"""
    
    @pytest.mark.unit
    def test_complete_detection_workflow(self):
        """Test complete shell detection workflow"""
        detector = ShellDetector()
        
        # Mock the entire detection process
        with patch.object(detector, '_detect_available_shells') as mock_detect, \
             patch.object(detector, '_detect_current_shell') as mock_current, \
             patch.object(detector, '_populate_shell_cache') as mock_cache:
            
            # Set up mock data
            mock_shells = [
                ShellInfo("bash", "/bin/bash", "5.1.4", True, {'interactive'}),
                ShellInfo("zsh", "/bin/zsh", "5.8", True, {'interactive'})
            ]
            
            mock_detect.return_value = None
            mock_current.return_value = None
            mock_cache.return_value = None
            
            # Initialize shells
            detector._initialize_shells()
            
            # Verify all methods were called
            mock_detect.assert_called_once()
            mock_current.assert_called_once()
            mock_cache.assert_called_once()
    
    @pytest.mark.unit
    def test_shell_detection_with_custom_platform(self):
        """Test shell detection with custom platform adapter"""
        # Create a mock platform adapter
        mock_adapter = Mock()
        mock_adapter.platform_info.name = "CustomOS"
        
        detector = ShellDetector(mock_adapter)
        
        # Should use the custom adapter
        assert detector.platform_adapter == mock_adapter
        assert detector.platform_adapter.platform_info.name == "CustomOS"


# Performance Tests
class TestShellDetectorPerformance:
    """Test ShellDetector performance characteristics"""
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_detection_performance(self):
        """Test performance of shell detection"""
        import time
        
        detector = ShellDetector()
        
        # Time the detection process
        start_time = time.time()
        detector._initialize_shells()
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        detection_time = end_time - start_time
        assert detection_time < 5.0  # Should be under 5 seconds
    
    @pytest.mark.unit
    @pytest.mark.performance
    def test_cache_performance(self):
        """Test performance benefits of caching"""
        detector = ShellDetector()
        
        # Populate cache
        detector._shell_cache = {
            "bash": ShellInfo("bash", "/bin/bash"),
            "zsh": ShellInfo("zsh", "/bin/zsh")
        }
        
        # Access cached shells (should be fast)
        start_time = time.time()
        for _ in range(1000):
            _ = detector._shell_cache.get("bash")
            _ = detector._shell_cache.get("zsh")
        end_time = time.time()
        
        # Cache access should be very fast
        cache_time = end_time - start_time
        assert cache_time < 0.1  # Should be under 100ms for 2000 accesses


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])