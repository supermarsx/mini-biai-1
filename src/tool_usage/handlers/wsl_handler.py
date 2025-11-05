"""
Windows Subsystem for Linux (WSL) Handler for mini-biai-1

This module provides specialized command execution capabilities for Windows
Subsystem for Linux (WSL) environments. It enables seamless Linux command
execution within Windows while providing cross-platform file system access
and path translation capabilities.

The WSL handler implements:
- WSL distribution detection and management
- Linux command execution within Windows
- Cross-platform file system access and path translation
- WSL-specific utilities and system integration
- Linux package management and process handling
- Seamless Windows/Linux interoperability

┌─────────────────────────────────────────────────────────────────────┐
│                      WSL Handler Architecture                       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │    WSL      │ │  Cross-     │ │   Linux     │
    │ Execution   │ │  Platform   │ │ Utilities   │
    │             │ │  File       │ │             │
    │ • wsl.exe   │ │  Access     │ │ • apt       │
    │ • linux     │ │ • Path      │ │ • systemctl │
    │ • distros   │ │   Trans.    │ │ • systemd   │
    └─────────────┘ └─────────────┘ └─────────────┘

Key Features:
- WSL distribution detection and management
- Linux command execution through WSL
- Cross-platform file system access (/mnt/c/ for Windows drives)
- Automatic path translation between Windows and Linux formats
- Linux package management integration
- WSL-specific utilities and system integration
- Seamless Windows/Linux interoperability
- WSL service and process management

Architecture Benefits:
- Seamless Linux command execution within Windows
- Cross-platform file system and path translation
- Linux utility integration through WSL
- WSL-specific optimization and management
- Windows/Linux interoperability bridge

Dependencies:
    Core: subprocess, os, sys (process management)
    Windows: winreg (WSL configuration), psutil (process management)
    WSL: wsl.exe (WSL integration), pathlib (path handling)
    Utils: typing (type annotations), re (pattern matching)

Error Handling:
    The WSL handler implements comprehensive error handling:
        - WSL distribution availability and initialization checks
        - Cross-platform path validation and translation
        - Linux command execution error handling
        - WSL-specific error propagation and translation
        - Cross-platform compatibility management
        - Graceful fallback for missing WSL distributions

Usage Examples:

Basic WSL Command Execution:
    >>> from src.tool_usage.handlers.wsl_handler import WSLHandler
    >>> 
    >>> handler = WSLHandler()
    >>> result = handler.execute_command("ls -la /mnt/c/Windows")
    >>> print(f"Windows directory via WSL:\n{result.stdout}")

WSL Distribution Management:
    >>> # List available WSL distributions
    >>> result = handler.list_wsl_distributions()
    >>> print(f"Available distributions: {result.stdout}")
    >>> 
    >>> # Execute command in specific distribution
    >>> result = handler.execute_command("cat /etc/os-release", distro="Ubuntu-20.04")
    >>> print(f"Distribution info:\n{result.stdout}")

Cross-Platform File Access:
    >>> # Access Windows files from Linux
    >>> result = handler.execute_command("find /mnt/c/Users -name '*.py' -type f")
    >>> print(f"Python files in Windows:\n{result.stdout}")
    >>> 
    >>> # Access Linux files from Windows
    >>> result = handler.execute_command("ls -la /home/$USER")
    >>> print(f"Linux home directory:\n{result.stdout}")

Path Translation:
    >>> # Convert Windows path to WSL path
    >>> windows_path = "C:\\Users\\John\\Documents"
    >>> wsl_path = handler.windows_to_wsl_path(windows_path)
    >>> print(f"WSL path: {wsl_path}")
    >>> 
    >>> # Execute command on Windows path via WSL
    >>> result = handler.execute_command(f"ls -la {wsl_path}")
    >>> print(f"Windows directory via WSL:\n{result.stdout}")

Linux Package Management:
    >>> # Install packages using WSL
    >>> result = handler.execute_command("sudo apt update && sudo apt install -y python3-pip")
    >>> print(f"Package installation:\n{result.stdout}")
    >>> 
    >>> # Check installed packages
    >>> result = handler.execute_command("dpkg -l | grep python3")
    >>> print(f"Python packages:\n{result.stdout}")

WSL Service Management:
    >>> # Manage WSL services
    >>> result = handler.execute_command("sudo systemctl status ssh")
    >>> print(f"SSH service status:\n{result.stdout}")
    >>> 
    >>> # Start WSL service
    >>> result = handler.execute_command("sudo systemctl start ssh")
    >>> print(f"Service start result: {result.success}")

Linux Process Management:
    >>> # Monitor Linux processes from WSL
    >>> result = handler.execute_command("ps aux | grep python")
    >>> print(f"Python processes:\n{result.stdout}")
    >>> 
    >>> # Kill process by name
    >>> result = handler.execute_command("pkill -f python")
    >>> print(f"Process kill result: {result.success}")

Cross-Platform Development:
    >>> # Development workflow across platforms
    >>> result = handler.execute_command("cd /mnt/c/Development && git status")
    >>> print(f"Git status on Windows project:\n{result.stdout}")
    >>> 
    >>> # Run Python script from Windows directory
    >>> result = handler.execute_command("python3 /mnt/c/Development/my_script.py")
    >>> print(f"Script execution:\n{result.stdout}")

WSL Environment Variables:
    >>> # Set and use WSL environment variables
    >>> result = handler.execute_command("export MY_VAR=wsl_value && echo $MY_VAR")
    >>> print(f"WSL environment variable:\n{result.stdout}")

WSL Network Operations:
    >>> # Network operations through WSL
    >>> result = handler.execute_command("curl -s https://api.github.com/users/octocat")
    >>> print(f"GitHub API via WSL:\n{result.stdout}")

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

import subprocess
import os
import sys
import time
import pathlib
import re
import shutil
from typing import Dict, List, Optional, Union, Any, Tuple

from .base_handler import BaseCommandHandler, CommandResult, CommandConfig


class WSLHandler(BaseCommandHandler):
    """
    Specialized command handler for Windows Subsystem for Linux (WSL) environments.
    
    This handler provides optimized command execution for WSL, enabling Linux
    commands to run seamlessly within Windows. It implements advanced features
    like cross-platform file system access, path translation, WSL distribution
    management, and Linux utility integration.
    
    The handler automatically detects available WSL distributions and provides:
    - WSL distribution detection and management
    - Linux command execution through WSL
    - Cross-platform file system access (/mnt/c/ for Windows drives)
    - Automatic path translation between Windows and Linux formats
    - Linux package management and service integration
    - WSL-specific utilities and system integration
    
    Attributes:
        wsl_path (str): Path to the WSL executable (wsl.exe)
        available_distributions (Dict[str, str]): Available WSL distributions
        default_distribution (str): Default WSL distribution name
        supports_wsl2 (bool): Whether WSL2 features are supported
        windows_drives (List[str]): Available Windows drives for WSL access
    
    Example Usage:
        >>> handler = WSLHandler()
        >>> 
        >>> # Basic WSL command
        >>> result = handler.execute_command("uname -a")
        >>> print(f"WSL kernel info: {result.stdout}")
        >>> 
        >>> # Cross-platform file access
        >>> result = handler.execute_command("ls -la /mnt/c/Users")
        >>> print(f"Windows users directory: {result.stdout}")
        
        >>> # Execute in specific distribution
        >>> result = handler.execute_command("cat /etc/lsb-release", distro="Ubuntu-20.04")
        >>> print(f"Ubuntu version: {result.stdout}")
    """
    
    def __init__(self, config: Optional[CommandConfig] = None):
        """
        Initialize the WSL handler.
        
        Args:
            config: Optional configuration for command execution.
                   Uses WSL-specific optimizations if not provided.
        """
        super().__init__(config)
        self.wsl_path = self._detect_wsl()
        self.available_distributions = self._detect_wsl_distributions()
        self.default_distribution = self._get_default_distribution()
        self.supports_wsl2 = self._check_wsl2_support()
        self.windows_drives = self._get_windows_drives()
        
        if self.config.enable_logging:
            self.logger.info(f"WSL Handler initialized:")
            self.logger.info(f"  WSL Path: {self.wsl_path}")
            self.logger.info(f"  Distributions: {list(self.available_distributions.keys())}")
            self.logger.info(f"  Default Distribution: {self.default_distribution}")
            self.logger.info(f"  WSL2 Support: {'✓' if self.supports_wsl2 else '✗'}")
            self.logger.info(f"  Windows Drives: {self.windows_drives}")
    
    def _detect_wsl(self) -> str:
        """
        Detect the WSL executable path.
        
        Returns:
            str: Path to wsl.exe
        """
        import shutil
        
        # Check for wsl.exe in common locations
        wsl_paths = [
            r'C:\Windows\System32\wsl.exe',
            r'C:\Windows\SysNative\wsl.exe',
            'wsl.exe'
        ]
        
        for wsl_path in wsl_paths:
            if shutil.which(wsl_path) or os.path.exists(wsl_path):
                return wsl_path
        
        # Fallback to 'wsl' command
        return shutil.which('wsl') or 'wsl'
    
    def _detect_wsl_distributions(self) -> Dict[str, str]:
        """
        Detect available WSL distributions.
        
        Returns:
            Dict[str, str]: Dictionary mapping distribution names to their status
        """
        try:
            list_command = f'"{self.wsl_path}" --list --verbose'
            result = subprocess.run(
                list_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            distributions = {}
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines[2:]:  # Skip header lines
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            name = parts[0]
                            version = parts[1]
                            state = parts[2] if len(parts) > 2 else "Unknown"
                            distributions[name] = {
                                'version': version,
                                'state': state
                            }
            
            return distributions
            
        except Exception:
            return {}
    
    def _get_default_distribution(self) -> str:
        """
        Get the default WSL distribution.
        
        Returns:
            str: Name of the default distribution, or empty string if none
        """
        try:
            default_command = f'"{self.wsl_path}" --list --quiet'
            result = subprocess.run(
                default_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                distributions = result.stdout.strip().split('\n')
                return distributions[0] if distributions else ""
            else:
                return ""
                
        except Exception:
            return ""
    
    def _check_wsl2_support(self) -> bool:
        """
        Check if WSL2 features are supported.
        
        Returns:
            bool: True if WSL2 is supported
        """
        try:
            # Check if wsl command supports version 2
            version_command = f'"{self.wsl_path}" --help'
            result = subprocess.run(
                version_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0 and '--version' in result.stdout
            
        except Exception:
            return False
    
    def _get_windows_drives(self) -> List[str]:
        """
        Get list of Windows drives available in WSL.
        
        Returns:
            List[str]: Letters of available Windows drives
        """
        drives = []
        for drive_letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            drive_path = f"/mnt/{drive_letter.lower()}/"
            drives.append(drive_letter + ":")
        return drives
    
    def _is_platform_supported(self) -> bool:
        """
        Check if the current platform is supported by this handler.
        
        Supports Windows platforms with WSL available.
        
        Returns:
            bool: True if Windows with WSL is available
        """
        return (sys.platform.startswith('win') and 
                self.wsl_path is not None and 
                shutil.which(self.wsl_path) is not None)
    
    def execute_command(self, command: str, config: Optional[CommandConfig] = None,
                       distro: str = None) -> CommandResult:
        """
        Execute a command in the WSL environment.
        
        This method provides WSL-specific command execution with support for:
        - WSL distribution selection and management
        - Cross-platform file system access
        - Linux command execution through WSL
        - Path translation between Windows and Linux formats
        
        Args:
            command: The Linux command string to execute
            config: Optional override configuration for this execution
            distro: Optional WSL distribution name to use
            
        Returns:
            CommandResult: Standardized result with WSL-specific enhancements
            
        Raises:
            CommandExecutionError: If command execution fails
            SecurityError: If command fails security validation
            TimeoutError: If command execution exceeds timeout
        """
        if not self.is_platform_supported():
            raise PlatformNotSupportedError(f"Platform {sys.platform} not supported by WSL handler")
        
        # Use provided config or default
        exec_config = config or self.config
        
        # Validate command for WSL-specific security
        self._validate_wsl_command(command)
        
        # Use specified distribution or default
        target_distro = distro or self.default_distribution
        
        if self.config.enable_logging:
            self.logger.debug(f"Executing WSL command: {command}")