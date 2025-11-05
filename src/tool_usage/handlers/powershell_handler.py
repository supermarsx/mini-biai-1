"""
Windows PowerShell Handler for mini-biai-1

This module provides specialized command execution capabilities for Windows
PowerShell environments. It optimizes for PowerShell Core and Windows PowerShell
while providing advanced features like script execution, PowerShell-specific
cmdlets, and Windows integration.

The PowerShell handler implements:
- Cross-platform PowerShell execution (PowerShell Core, Windows PowerShell)
- Windows-specific path handling and environment variable management
- PowerShell script execution and cmdlet support
- Windows integration and system utilities
- PowerShell pipeline and object processing
- Windows registry and service management

┌─────────────────────────────────────────────────────────────────────┐
│                   PowerShell Handler Architecture                   │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  PowerShell │ │   Windows   │ │   Script    │
    │ Execution   │ │  Integration│ │ Management  │
    │             │ │             │ │             │
    │ • pwsh      │ │ • Registry  │ │ • ScriptBlock│
    │ • PowerShell│ │ • Services  │ │ • Module    │
    │ • ISE       │ │ • WMI       │ │ • Pipeline  │
    └─────────────┘ └─────────────┘ └─────────────┘

Key Features:
- Multi-version PowerShell support (Core, Windows PowerShell, ISE)
- PowerShell script execution and cmdlet integration
- Windows-specific path and environment handling
- PowerShell pipeline and object processing
- Windows registry and service management
- PowerShell remoting and remote execution
- Advanced error handling with PowerShell-specific error records
- Windows integration (WMI, Active Directory, etc.)

Architecture Benefits:
- Optimized for Windows PowerShell ecosystem
- Advanced Windows integration capabilities
- PowerShell-specific object and pipeline processing
- Comprehensive Windows system management
- Script execution with PowerShell security features

Dependencies:
    Core: subprocess, os, sys (Windows process management)
    Windows: winreg (registry access), psutil (process management)
    PowerShell: json (PowerShell object serialization)
    Utils: pathlib (path handling), typing (type annotations)

Error Handling:
    The PowerShell handler implements comprehensive error handling:
        - PowerShell-specific error record propagation
        - Windows exception handling and translation
        - Script execution timeout and termination
        - PowerShell security policy enforcement
        - Cross-version PowerShell compatibility
        - Graceful fallback for missing PowerShell features

Usage Examples:

Basic PowerShell Execution:
    >>> from src.tool_usage.handlers.powershell_handler import PowerShellHandler
    >>> 
    >>> handler = PowerShellHandler()
    >>> result = handler.execute_command("Get-Process | Select-Object -First 5")
    >>> print(f"Processes:\n{result.stdout}")

PowerShell Script Execution:
    >>> # Execute PowerShell script from string
    >>> script = """
    ... $processes = Get-Process | Where-Object {$_.CPU -gt 1.0}
    ... $processes | Format-Table Name, CPU, WorkingSet
    ... """
    >>> result = handler.execute_command(script)
    >>> print(f"High CPU processes:\n{result.stdout}")

Windows Registry Operations:
    >>> # Read from Windows registry
    >>> command = 'Get-ItemProperty "HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion"'
    >>> result = handler.execute_command(command)
    >>> print(f"Windows version info:\n{result.stdout}")

Windows Service Management:
    >>> # List Windows services
    >>> command = "Get-Service | Where-Object {$_.Status -eq 'Running'} | Select-Object -First 10"
    >>> result = handler.execute_command(command)
    >>> print(f"Running services:\n{result.stdout}")

PowerShell Pipeline Processing:
    >>> # Use PowerShell pipeline for complex operations
    >>> command = "Get-Process | Sort-Object CPU -Descending | Select-Object -First 3"
    >>> result = handler.execute_command(command)
    >>> print(f"Top processes by CPU:\n{result.stdout}")

PowerShell Object Processing:
    >>> # Work with PowerShell objects
    >>> command = '''
    ... $diskInfo = Get-WmiObject -Class Win32_LogicalDisk | 
    ...     Select-Object DeviceID, Size, FreeSpace |
    ...     ForEach-Object {
    ...         [PSCustomObject]@{
    ...             Drive = $_.DeviceID
    ...             TotalGB = [math]::Round($_.Size / 1GB, 2)
    ...             FreeGB = [math]::Round($_.FreeSpace / 1GB, 2)
    ...             FreePercent = [math]::Round(($_.FreeSpace / $_.Size) * 100, 2)
    ...         }
    ...     }
    ... $diskInfo | Format-Table
    ... '''
    >>> result = handler.execute_command(command)
    >>> print(f"Disk usage:\n{result.stdout}")

PowerShell Module Usage:
    >>> # Use PowerShell modules
    >>> command = "Import-Module ActiveDirectory; Get-ADUser -Filter * | Select-Object -First 5"
    >>> result = handler.execute_command(command)
    >>> print(f"Active Directory users:\n{result.stdout}")

Windows PowerShell Remoting:
    >>> # Execute commands on remote systems
    >>> command = '''
    ... $session = New-PSSession -ComputerName "Server01" -Credential $cred
    ... Invoke-Command -Session $session -ScriptBlock { Get-Process }
    ... Remove-PSSession $session
    ... '''
    >>> result = handler.execute_command(command)
    >>> print(f"Remote processes:\n{result.stdout}")

Environment Variable Management:
    >>> # Set and use PowerShell environment variables
    >>> command = '''
    ... $env:Path += ";C:\\CustomPath"
    ... $env:CUSTOM_VAR = "test_value"
    ... Write-Host "PATH: $env:Path"
    ... Write-Host "CUSTOM_VAR: $env:CUSTOM_VAR"
    ... '''
    >>> result = handler.execute_command(command)
    >>> print(f"Environment variables:\n{result.stdout}")

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

import subprocess
import os
import sys
import time
import json
import pathlib
import re
from typing import Dict, List, Optional, Union, Any, Tuple

from .base_handler import BaseCommandHandler, CommandResult, CommandConfig


class PowerShellHandler(BaseCommandHandler):
    """
    Specialized command handler for Windows PowerShell environments.
    
    This handler provides optimized command execution for PowerShell Core,
    Windows PowerShell, and PowerShell ISE. It implements advanced features
    like PowerShell script execution, cmdlet integration, Windows system
    management, and PowerShell-specific object processing.
    
    The handler automatically detects the available PowerShell version and
    provides platform-specific optimizations for:
    - PowerShell script and cmdlet execution
    - Windows system integration and management
    - PowerShell object pipeline processing
    - Windows registry and service access
    - PowerShell remoting capabilities
    
    Attributes:
        powershell_path (str): Path to the detected PowerShell executable
        powershell_version (str): Detected PowerShell version
        is_powershell_core (bool): True if using PowerShell Core (pwsh)
        supports_remoting (bool): Whether PowerShell remoting is available
        available_modules (List[str]): Available PowerShell modules
    
    Example Usage:
        >>> handler = PowerShellHandler()
        >>> 
        >>> # Basic PowerShell command
        >>> result = handler.execute_command("Get-Date")
        >>> print(f"Current date: {result.stdout}")
        >>> 
        >>> # PowerShell pipeline
        >>> result = handler.execute_command("Get-Process | Sort-Object CPU -Descending | Select-Object -First 5")
        >>> print(f"Top processes: {result.stdout}")
        
        >>> # Windows system information
        >>> result = handler.execute_command("Get-ComputerInfo | Select-Object WindowsProductName, TotalPhysicalMemory")
        >>> print(f"System info: {result.stdout}")
    """
    
    def __init__(self, config: Optional[CommandConfig] = None):
        """
        Initialize the PowerShell handler.
        
        Args:
            config: Optional configuration for command execution.
                   Uses PowerShell-specific optimizations if not provided.
        """
        super().__init__(config)
        self.powershell_path = self._detect_powershell()
        self.powershell_version = self._get_powershell_version()
        self.is_powershell_core = self._is_powershell_core()
        self.supports_remoting = self._check_remoting_support()
        self.available_modules = self._get_available_modules()
        
        if self.config.enable_logging:
            self.logger.info(f"PowerShell Handler initialized:")
            self.logger.info(f"  PowerShell: {self.powershell_path}")
            self.logger.info(f"  Version: {self.powershell_version}")
            self.logger.info(f"  Core: {'✓' if self.is_powershell_core else '✗'}")
            self.logger.info(f"  Remoting: {'✓' if self.supports_remoting else '✗'}")
            self.logger.info(f"  Modules: {len(self.available_modules)} available")
    
    def _detect_powershell(self) -> str:
        """
        Detect the best available PowerShell executable.
        
        Checks for PowerShell executables in order of preference:
        PowerShell Core (pwsh), Windows PowerShell, PowerShell ISE.
        
        Returns:
            str: Path to the detected PowerShell executable
        """
        import shutil
        
        # Preferred PowerShell executables
        preferred_powershell = ['pwsh', 'powershell', 'powershell_ise']
        
        # Check each PowerShell executable
        for ps_name in preferred_powershell:
            ps_path = shutil.which(ps_name)
            if ps_path:
                return ps_path
        
        # Windows-specific paths
        windows_paths = [
            r'C:\Program Files\PowerShell\*\pwsh.exe',
            r'C:\Windows\System32\WindowsPowerShell\v*\powershell.exe',
            r'C:\Windows\SysWOW64\WindowsPowerShell\v*\powershell.exe'
        ]
        
        import glob
        for pattern in windows_paths:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]  # Return the first match
        
        # Fallback to 'powershell' command
        return 'powershell'
    
    def _get_powershell_version(self) -> str:
        """
        Get the PowerShell version string.
        
        Returns:
            str: PowerShell version information
        """
        try:
            version_command = f'"{self.powershell_path}" -Command "$PSVersionTable.PSVersion.ToString()"'
            result = subprocess.run(
                version_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "Unknown"
                
        except Exception:
            return "Unknown"
    
    def _is_powershell_core(self) -> bool:
        """
        Check if this is PowerShell Core (cross-platform).
        
        Returns:
            bool: True if using PowerShell Core
        """
        try:
            version_command = f'"{self.powershell_path}" -Command "if ($PSVersionTable.PSEdition -eq ''Core'') {{ ''Core'' }} else {{ ''Desktop'' }}"'
            result = subprocess.run(
                version_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0 and 'Core' in result.stdout
            
        except Exception:
            return False
    
    def _check_remoting_support(self) -> bool:
        """
        Check if PowerShell remoting is available.
        
        Returns:
            bool: True if remoting capabilities are available
        """
        try:
            # Check if remoting modules are available
            remoting_command = f'"{self.powershell_path}" -Command "Get-Command Enable-PSRemoting -ErrorAction SilentlyContinue"'
            result = subprocess.run(
                remoting_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _get_available_modules(self) -> List[str]:
        """
        Get list of available PowerShell modules.
        
        Returns:
            List[str]: Names of available PowerShell modules
        """
        try:
            modules_command = f'"{self.powershell_path}" -Command "Get-Module -ListAvailable | Select-Object -ExpandProperty Name"'
            result = subprocess.run(
                modules_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                modules = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                return modules
            else:
                return []
                
        except Exception:
            return []
    
    def _is_platform_supported(self) -> bool:
        """
        Check if the current platform is supported by this handler.
        
        Supports Windows platforms with PowerShell available.
        
        Returns:
            bool: True if Windows with PowerShell is available
        """
        return sys.platform.startswith('win') and self.powershell_path != 'powershell' or shutil.which('powershell') is not None
    
    def execute_command(self, command: str, config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a command in the PowerShell environment.
        
        This method provides PowerShell-specific command execution with support for:
        - PowerShell scripts and cmdlets
        - Windows system integration
        - PowerShell object pipeline processing
        - Windows registry and service management
        
        Args:
            command: The command or script string to execute
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Standardized result with PowerShell-specific enhancements
            
        Raises:
            CommandExecutionError: If command execution fails
            SecurityError: If command fails security validation
            TimeoutError: If command execution exceeds timeout
        """
        if not self.is_platform_supported():
            raise PlatformNotSupportedError(f"Platform {sys.platform} not supported by PowerShell handler")
        
        # Use provided config or default
        exec_config = config or self.config
        
        # Validate command for PowerShell-specific security
        self._validate_powershell_command(command)
        
        if self.config.enable_logging:
            self.logger.debug(f"Executing PowerShell command: {command}")
            self.logger.debug(f"Using PowerShell: {self.powershell_path}")
        
        # Prepare command for PowerShell execution
        ps_command = self._prepare_powershell_command(command, exec_config)
        
        try:
            # Execute with PowerShell-specific handling
            result = self._execute_powershell_command(ps_command, exec_config)
            
            # Add PowerShell-specific metadata
            result.metadata = getattr(result, 'metadata', {})
            result.metadata.update({
                'powershell_path': self.powershell_path,
                'powershell_version': self.powershell_version,
                'is_powershell_core': self.is_powershell_core,
                'supports_remoting': self.supports_remoting,
                'available_modules': self.available_modules
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"PowerShell command execution failed: {command}")
            self.logger.error(f"Error: {str(e)}")
            raise CommandExecutionError(f"Failed to execute PowerShell command: {str(e)}")
    
    def _validate_powershell_command(self, command: str) -> None:
        """
        Validate command for PowerShell-specific security considerations.
        
        Args:
            command: The command string to validate
            
        Raises:
            SecurityError: If command fails PowerShell-specific validation
        """
        # Basic validation from base class
        self.validate_command(command)
        
        # PowerShell-specific security checks
        dangerous_ps_patterns = [
            r'\b(Invoke-Expression|Invoke-Expression|iex)\b.*[`\']',  # Code injection
            r'\b(Invoke-Command|icm)\b.*-ComputerName.*[`\']',       # Remote execution without validation
            r'\b(Add-Content|Set-Content)\b.*-Path.*`.*\\.*',       # File writing to system paths
            r'&.*[`\']{3,}',                                        # Arbitrary code execution
            r'\$\{.*[`\']',                                         # Variable expansion injection
            r'\|\s*Invoke-Expression',                              # Pipeline to code execution
        ]
        
        for pattern in dangerous_ps_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityError(f"Command contains potentially dangerous PowerShell pattern: {pattern}")
    
    def _prepare_powershell_command(self, command: str, config: CommandConfig) -> str:
        """
        Prepare command for PowerShell execution.
        
        Args:
            command: Original command string
            config: Execution configuration
            
        Returns:
            str: Prepared command for PowerShell execution
        """
        if config.shell_mode:
            # Use PowerShell to execute the command
            return f'"{self.powershell_path}" -Command "{command}"'
        else:
            return command
    
    def _execute_powershell_command(self, command: str, config: CommandConfig) -> CommandResult:
        """
        Execute command with PowerShell-specific optimizations.
        
        Args:
            command: Prepared command for execution
            config: Execution configuration
            
        Returns:
            CommandResult: Execution result with PowerShell-specific enhancements
        """
        start_time = time.time()
        working_dir = config.working_directory or self.config.working_directory
        env = {**os.environ, **self.config.environment, **config.environment}
        
        try:
            # Execute PowerShell command
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=working_dir,
                env=env,
                text=True,
                encoding=config.encoding
            )
            
            try:
                stdout, stderr = process.communicate(timeout=config.timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                stderr = stderr or f"PowerShell command timed out after {config.timeout} seconds"
                exit_code = -1
            
            execution_time = time.time() - start_time
            
            # Truncate output if too large
            if len(stdout) > config.max_output_size:
                stdout = stdout[:config.max_output_size] + "\n[Output truncated...]"
            if len(stderr) > config.max_output_size:
                stderr = stderr[:config.max_output_size] + "\n[Error output truncated...]"
            
            result = CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout or "",
                stderr=stderr or "",
                execution_time=execution_time,
                environment=env,
                working_directory=working_dir or os.getcwd(),
                timestamp=start_time,
                success=exit_code == 0
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"PowerShell execution error: {str(e)}",
                execution_time=execution_time,
                environment=env,
                working_directory=working_dir or os.getcwd(),
                timestamp=start_time,
                success=False
            )
    
    def execute_script_file(self, script_path: str, args: List[str] = None, 
                          config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a PowerShell script file.
        
        Args:
            script_path: Path to the PowerShell script file
            args: Optional arguments to pass to the script
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Execution result
        """
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"PowerShell script not found: {script_path}")
        
        # Build command to execute script
        command_parts = [f'"{self.powershell_path}"', '-File', f'"{script_path}"']
        
        if args:
            command_parts.extend(args)
        
        command = ' '.join(command_parts)
        
        return self.execute_command(command, config)
    
    def execute_script_block(self, script_block: str, args: Dict[str, Any] = None,
                           config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a PowerShell script block with arguments.
        
        Args:
            script_block: PowerShell script block to execute
            args: Optional arguments to pass to the script block
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Execution result
        """
        # Convert arguments to PowerShell format
        ps_args = ""
        if args:
            arg_list = []
            for key, value in args.items():
                if isinstance(value, str):
                    arg_list.append(f'-{key} "{value}"')
                elif isinstance(value, (int, float)):
                    arg_list.append(f'-{key} {value}')
                elif isinstance(value, bool):
                    arg_list.append(f'-{key} ${"true" if value else "false"}')
                else:
                    arg_list.append(f'-{key} "{str(value)}"')
            ps_args = ' ' + ' '.join(arg_list)
        
        # Combine script block with arguments
        full_command = script_block + ps_args
        
        return self.execute_command(full_command, config)
    
    def get_powershell_info(self) -> Dict[str, Any]:
        """
        Get detailed PowerShell-specific environment information.
        
        Returns:
            Dict[str, Any]: PowerShell environment information including
                          version, modules, and capabilities
        """
        info = {
            'powershell_path': self.powershell_path,
            'powershell_version': self.powershell_version,
            'is_powershell_core': self.is_powershell_core,
            'supports_remoting': self.supports_remoting,
            'available_modules': self.available_modules,
            'execution_policy': self._get_execution_policy(),
            'psmodule_path': self._get_psmodule_path()
        }
        
        # Merge with base environment info
        info.update(self.environment_info)
        
        return info
    
    def _get_execution_policy(self) -> str:
        """
        Get current PowerShell execution policy.
        
        Returns:
            str: Current execution policy setting
        """
        try:
            policy_command = f'"{self.powershell_path}" -Command "Get-ExecutionPolicy"'
            result = subprocess.run(
                policy_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "Unknown"
                
        except Exception:
            return "Unknown"
    
    def _get_psmodule_path(self) -> str:
        """
        Get PowerShell module path.
        
        Returns:
            str: PowerShell module search path
        """
        try:
            path_command = f'"{self.powershell_path}" -Command "$env:PSModulePath"'
            result = subprocess.run(
                path_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "Unknown"
                
        except Exception:
            return "Unknown"
    
    def test_powershell_connection(self) -> bool:
        """
        Test if PowerShell is accessible and working.
        
        Returns:
            bool: True if PowerShell is working correctly
        """
        try:
            test_command = f'"{self.powershell_path}" -Command "Write-Host ''PowerShell test''; exit 0"'
            result = subprocess.run(
                test_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def get_available_cmdlets(self, module_name: str = None) -> List[str]:
        """
        Get list of available PowerShell cmdlets.
        
        Args:
            module_name: Optional module name to filter cmdlets
            
        Returns:
            List[str]: Names of available cmdlets
        """
        try:
            if module_name:
                cmdlets_command = f'"{self.powershell_path}" -Command "Get-Command -Module {module_name} | Select-Object -ExpandProperty Name"'
            else:
                cmdlets_command = f'"{self.powershell_path}" -Command "Get-Command | Select-Object -ExpandProperty Name"'
            
            result = subprocess.run(
                cmdlets_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                cmdlets = [line.strip() for line in result.stdout.split('\n') if line.strip()]
                return cmdlets
            else:
                return []
                
        except Exception:
            return []


# Import required modules
import shutil
import re


# Export PowerShell handler
__all__ = ['PowerShellHandler']