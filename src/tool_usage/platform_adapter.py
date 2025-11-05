"""
Platform Adapter Base Class

Provides a base class for OS-specific adaptations in the mini-biai-1 framework.
Handles platform-specific behavior for different operating systems.
"""

import os
import platform
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlatformInfo:
    """Information about the current platform."""
    name: str
    version: str
    machine: str
    processor: str
    is_posix: bool
    is_windows: bool
    is_linux: bool
    is_macos: bool


class PlatformAdapter(ABC):
    """
    Base class for OS-specific adaptations.
    
    This abstract base class defines the interface for platform-specific
    implementations across different operating systems.
    """
    
    def __init__(self, platform_info: PlatformInfo):
        """
        Initialize the platform adapter.
        
        Args:
            platform_info: Platform information container
        """
        self.platform_info = platform_info
        self._initialize_platform_specifics()
    
    @abstractmethod
    def _initialize_platform_specifics(self) -> None:
        """Initialize platform-specific configurations and paths."""
        pass
    
    @abstractmethod
    def get_shell_path(self) -> str:
        """Get the path to the default shell."""
        pass
    
    @abstractmethod
    def get_executable_extension(self) -> str:
        """Get the executable extension for the platform."""
        pass
    
    @abstractmethod
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges."""
        pass
    
    @abstractmethod
    def get_environment_variables(self) -> Dict[str, str]:
        """Get platform-specific environment variables."""
        pass
    
    @abstractmethod
    def get_path_separator(self) -> str:
        """Get the path separator for the platform."""
        pass
    
    @abstractmethod
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for the current platform."""
        pass


class PosixAdapter(PlatformAdapter):
    """Platform adapter for POSIX systems (Linux, macOS, Unix)."""
    
    def _initialize_platform_specifics(self) -> None:
        """Initialize POSIX-specific configurations."""
        self.default_shells = ['/bin/bash', '/bin/zsh', '/bin/sh']
        self.executable_extension = ''
        self.path_separator = ':'
        self.line_ending = '\n'
    
    def get_shell_path(self) -> str:
        """Get the path to the default shell."""
        shell = os.environ.get('SHELL', '')
        if shell and os.path.exists(shell):
            return shell
        
        # Fallback to common shells
        for potential_shell in self.default_shells:
            if os.path.exists(potential_shell):
                return potential_shell
        
        return '/bin/bash'  # Ultimate fallback
    
    def get_executable_extension(self) -> str:
        """Get the executable extension (empty for POSIX)."""
        return self.executable_extension
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges."""
        try:
            return os.geteuid() == 0 if hasattr(os, 'geteuid') else False
        except (AttributeError, OSError):
            return False
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get POSIX-specific environment variables."""
        return {
            'HOME': os.environ.get('HOME', ''),
            'PATH': os.environ.get('PATH', ''),
            'USER': os.environ.get('USER', ''),
            'SHELL': os.environ.get('SHELL', ''),
            'TMPDIR': os.environ.get('TMPDIR', '/tmp'),
        }
    
    def get_path_separator(self) -> str:
        """Get the path separator (colon for POSIX)."""
        return self.path_separator
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for POSIX systems."""
        path_obj = Path(path)
        # Expand user home directory if needed
        if str(path_obj).startswith('~'):
            return path_obj.expanduser()
        return path_obj.resolve()


class WindowsAdapter(PlatformAdapter):
    """Platform adapter for Windows systems."""
    
    def _initialize_platform_specifics(self) -> None:
        """Initialize Windows-specific configurations."""
        self.executable_extension = '.exe'
        self.path_separator = ';'
        self.line_ending = '\r\n'
    
    def get_shell_path(self) -> str:
        """Get the path to the Windows Command Prompt or PowerShell."""
        # Prefer PowerShell if available
        powershell_path = self._find_powershell()
        if powershell_path:
            return powershell_path
        
        # Fallback to Command Prompt
        cmd_path = self._find_cmd()
        if cmd_path:
            return cmd_path
        
        return 'cmd.exe'
    
    def _find_powershell(self) -> Optional[str]:
        """Find PowerShell executable path."""
        powershell_paths = [
            r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe',
            r'C:\Windows\SysWOW64\WindowsPowerShell\v1.0\powershell.exe',
            r'C:\Windows\System32\WindowsPowerShell\v1.0\pwsh.exe',
        ]
        
        for path in powershell_paths:
            if os.path.exists(path):
                return path
        
        # Check PATH environment variable
        try:
            result = subprocess.run(
                ['where', 'powershell'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def _find_cmd(self) -> Optional[str]:
        """Find Command Prompt executable path."""
        cmd_path = r'C:\Windows\System32\cmd.exe'
        if os.path.exists(cmd_path):
            return cmd_path
        
        try:
            result = subprocess.run(
                ['where', 'cmd'], 
                capture_output=True, 
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def get_executable_extension(self) -> str:
        """Get the executable extension (.exe for Windows)."""
        return self.executable_extension
    
    def is_admin(self) -> bool:
        """Check if the current process has administrative privileges."""
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        except (AttributeError, ImportError):
            return False
    
    def get_environment_variables(self) -> Dict[str, str]:
        """Get Windows-specific environment variables."""
        return {
            'USERPROFILE': os.environ.get('USERPROFILE', ''),
            'PATH': os.environ.get('PATH', ''),
            'USERNAME': os.environ.get('USERNAME', ''),
            'COMPUTERNAME': os.environ.get('COMPUTERNAME', ''),
            'TEMP': os.environ.get('TEMP', ''),
            'TMP': os.environ.get('TMP', ''),
        }
    
    def get_path_separator(self) -> str:
        """Get the path separator (semicolon for Windows)."""
        return self.path_separator
    
    def normalize_path(self, path: Union[str, Path]) -> Path:
        """Normalize path for Windows systems."""
        path_obj = Path(path)
        # Handle Windows drive letters and UNC paths
        return path_obj.resolve()


def get_platform_adapter() -> PlatformAdapter:
    """
    Factory function to create the appropriate platform adapter.
    
    Returns:
        PlatformAdapter: The appropriate adapter for the current platform
    """
    system_info = PlatformInfo(
        name=platform.system(),
        version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
        is_posix=platform.system() in ['Linux', 'Darwin'],
        is_windows=platform.system() == 'Windows',
        is_linux=platform.system() == 'Linux',
        is_macos=platform.system() == 'Darwin'
    )
    
    if system_info.is_windows:
        return WindowsAdapter(system_info)
    else:
        return PosixAdapter(system_info)


def get_platform_info() -> PlatformInfo:
    """
    Get information about the current platform.
    
    Returns:
        PlatformInfo: Container with platform details
    """
    return get_platform_adapter().platform_info