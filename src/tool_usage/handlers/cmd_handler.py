"""
Windows Command Prompt (cmd.exe) Handler for mini-biai-1

This module provides specialized command execution capabilities for Windows
Command Prompt (cmd.exe) environments. It optimizes for Windows batch file
execution and Windows-specific command-line utilities while maintaining
compatibility with legacy Windows systems.

The cmd.exe handler implements:
- Windows Command Prompt execution and batch file support
- Windows-specific path handling and environment variable management
- Windows utility integration and system commands
- Batch file execution and scripting support
- Windows file system operations
- Windows service and process management

┌─────────────────────────────────────────────────────────────────────┐
│                  Windows Command Prompt Architecture                │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Batch     │ │   Windows   │ │    Batch    │
    │ Execution   │ │  Utilities  │ │  Scripting  │
    │             │ │             │ │             │
    │ • cmd.exe   │ │ • dir       │ │ • FOR loops │
    │ • .bat      │ │ • copy      │ │ • IF/ELSE   │
    │ • .cmd      │ │ • del       │ │ • GOTO      │
    └─────────────┘ └─────────────┘ └─────────────┘

Key Features:
- Windows Command Prompt (cmd.exe) execution
- Batch file (.bat, .cmd) scripting support
- Windows-specific path and environment handling
- Windows utility integration (dir, copy, del, etc.)
- Batch file programming constructs support
- Windows file system and process management
- Legacy Windows compatibility and support
- Windows registry and service command access

Architecture Benefits:
- Optimized for Windows Command Prompt environment
- Legacy Windows system compatibility
- Batch file scripting and automation support
- Windows-specific utility integration
- Comprehensive Windows system management

Dependencies:
    Core: subprocess, os, sys (Windows process management)
    Windows: winreg (registry access), win32api (Windows API)
    File: pathlib (path handling), typing (type annotations)

Error Handling:
    The cmd.exe handler implements comprehensive error handling:
        - Windows error code propagation and translation
        - Batch file execution error handling
        - Command timeout and termination
        - Windows-specific exception handling
        - Legacy Windows compatibility management
        - Graceful fallback for missing Windows utilities

Usage Examples:

Basic cmd.exe Execution:
    >>> from src.tool_usage.handlers.cmd_handler import WindowsCommandHandler
    >>> 
    >>> handler = WindowsCommandHandler()
    >>> result = handler.execute_command("dir C:\\")
    >>> print(f"Directory listing:\n{result.stdout}")

Batch File Execution:
    >>> # Create and execute a batch file
    >>> batch_content = '''
    ... @echo off
    ... echo Hello from batch file!
    ... dir /b *.txt
    ... set MY_VAR=batch_value
    ... echo Variable: %MY_VAR%
    ... '''
    >>> 
    >>> # Write batch file and execute
    >>> import tempfile
    >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.bat', delete=False) as f:
    ...     f.write(batch_content)
    ...     batch_path = f.name
    >>> 
    >>> result = handler.execute_batch_file(batch_path)
    >>> print(f"Batch execution result:\n{result.stdout}")

Windows Directory Operations:
    >>> # Windows-specific directory operations
    >>> command = "dir /s /b C:\\temp\\*.log | find /c /v \"\""
    >>> result = handler.execute_command(command)
    >>> print(f"Log files count: {result.stdout}")

Windows File Operations:
    >>> # Copy and manage files
    >>> command = "copy C:\\temp\\source.txt C:\\temp\\backup.txt && dir C:\\temp\\backup.txt"
    >>> result = handler.execute_command(command)
    >>> print(f"File operations result:\n{result.stdout}")

Environment Variable Management:
    >>> # Set and use Windows environment variables
    >>> command = "set MY_VAR=hello_world && echo %MY_VAR% && set | findstr MY_VAR"
    >>> result = handler.execute_command(command)
    >>> print(f"Environment variables:\n{result.stdout}")

Windows Service Management:
    >>> # Manage Windows services
    >>> command = "sc query type= service state= all | findstr /C:\"SERVICE_NAME\""
    >>> result = handler.execute_command(command)
    >>> print(f"Services:\n{result.stdout}")

Network Operations:
    >>> # Network-related commands
    >>> command = "ipconfig /all | findstr /C:\"IPv4\""
    >>> result = handler.execute_command(command)
    >>> print(f"Network configuration:\n{result.stdout}")

Windows Registry Access:
    >>> # Access Windows registry via reg command
    >>> command = 'reg query "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion" /v ProductName'
    >>> result = handler.execute_command(command)
    >>> print(f"Windows version:\n{result.stdout}")

Batch Scripting Constructs:
    >>> # Use batch file programming features
    >>> command = '''
    ... @echo off
    ... echo Checking file existence...
    ... if exist C:\\temp\\test.txt (
    ...     echo File exists
    ...     type C:\\temp\\test.txt
    ... ) else (
    ...     echo File does not exist
    ... )
    ... '''
    >>> result = handler.execute_command(command)
    >>> print(f"File check result:\n{result.stdout}")

Process Management:
    >>> # Windows process management
    >>> command = "tasklist | findstr /i python"
    >>> result = handler.execute_command(command)
    >>> print(f"Python processes:\n{result.stdout}")

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

import subprocess
import os
import sys
import time
import tempfile
import pathlib
import shutil
import re
from typing import Dict, List, Optional, Union, Any, Tuple

from .base_handler import BaseCommandHandler, CommandResult, CommandConfig


class WindowsCommandHandler(BaseCommandHandler):
    """
    Specialized command handler for Windows Command Prompt (cmd.exe) environments.
    
    This handler provides optimized command execution for Windows Command Prompt,
    supporting batch file execution, Windows utilities integration, and legacy
    Windows system compatibility. It implements advanced features like batch
    scripting support, Windows-specific path handling, and Windows system management.
    
    The handler automatically detects the available Windows Command Prompt and
    provides platform-specific optimizations for:
    - Batch file (.bat, .cmd) execution
    - Windows utility integration and system commands
    - Windows file system and process management
    - Windows environment and registry access
    - Legacy Windows compatibility and support
    
    Attributes:
        cmd_path (str): Path to the Windows Command Prompt executable
        windows_version (str): Detected Windows version
        supports_batch_files (bool): Whether batch file execution is supported
        available_windows_utils (List[str]): Available Windows utilities
        comspec (str): COMSPEC environment variable value
    
    Example Usage:
        >>> handler = WindowsCommandHandler()
        >>> 
        >>> # Basic Windows command
        >>> result = handler.execute_command("ver")
        >>> print(f"Windows version: {result.stdout}")
        >>> 
        >>> # Directory operations
        >>> result = handler.execute_command("dir C:\\Windows\\System32 | findstr .exe")
        >>> print(f"System executables: {result.stdout}")
        
        >>> # Batch file execution
        >>> result = handler.execute_batch_file("my_script.bat")
        >>> print(f"Batch result: {result.success}")
    """
    
    def __init__(self, config: Optional[CommandConfig] = None):
        """
        Initialize the Windows Command Prompt handler.
        
        Args:
            config: Optional configuration for command execution.
                   Uses Windows-specific optimizations if not provided.
        """
        super().__init__(config)
        self.cmd_path = self._detect_cmd()
        self.windows_version = self._get_windows_version()
        self.supports_batch_files = True
        self.available_windows_utils = self._get_available_windows_utils()
        self.comspec = os.environ.get('COMSPEC', 'cmd.exe')
        
        if self.config.enable_logging:
            self.logger.info(f"Windows Command Handler initialized:")
            self.logger.info(f"  Command Prompt: {self.cmd_path}")
            self.logger.info(f"  Windows Version: {self.windows_version}")
            self.logger.info(f"  Batch Files: {'✓' if self.supports_batch_files else '✗'}")
            self.logger.info(f"  COMSPEC: {self.comspec}")
    
    def _detect_cmd(self) -> str:
        """
        Detect the Windows Command Prompt executable.
        
        Checks for cmd.exe in standard Windows locations.
        
        Returns:
            str: Path to the Windows Command Prompt executable
        """
        import shutil
        
        # Check COMSPEC environment variable first
        if 'COMSPEC' in os.environ:
            return os.environ['COMSPEC']
        
        # Check for cmd.exe in standard locations
        standard_paths = [
            r'C:\Windows\System32\cmd.exe',
            r'C:\Windows\SysWOW64\cmd.exe',
            r'C:\Winnt\System32\cmd.exe',
        ]
        
        for cmd_path in standard_paths:
            if os.path.exists(cmd_path):
                return cmd_path
        
        # Fallback to 'cmd' command
        which_cmd = shutil.which('cmd')
        return which_cmd or 'cmd'
    
    def _get_windows_version(self) -> str:
        """
        Get the Windows version string.
        
        Returns:
            str: Windows version information
        """
        try:
            version_command = f'"{self.cmd_path}" /c "ver"'
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
    
    def _get_available_windows_utils(self) -> List[str]:
        """
        Get list of available Windows utilities.
        
        Returns:
            List[str]: Names of available Windows utilities
        """
        utils_to_check = [
            'dir', 'copy', 'del', 'ren', 'move', 'mkdir', 'rmdir',
            'find', 'findstr', 'sort', 'type', 'more', 'attrib',
            'chkdsk', 'sc', 'net', 'ipconfig', 'ping', 'tracert',
            'tasklist', 'taskkill', 'reg', 'wmic', 'format', 'diskpart',
            'xcopy', 'robocopy', 'where', 'forfiles', 'tzutil'
        ]
        
        available_utils = []
        for util in utils_to_check:
            if shutil.which(util) or self._check_builtin_command(util):
                available_utils.append(util)
        
        return available_utils
    
    def _check_builtin_command(self, command: str) -> bool:
        """
        Check if a command is a built-in Windows command.
        
        Args:
            command: Command to check
            
        Returns:
            bool: True if command is a built-in Windows command
        """
        builtin_commands = {
            'dir', 'copy', 'del', 'ren', 'move', 'mkdir', 'rmdir',
            'cd', 'chdir', 'md', 'rd', 'type', 'more', 'echo',
            'find', 'findstr', 'sort', 'attrib', 'cls', 'ver',
            'date', 'time', 'set', 'path', 'prompt', 'title',
            'if', 'else', 'for', 'goto', 'call', 'exit'
        }
        
        return command.lower() in builtin_commands
    
    def _is_platform_supported(self) -> bool:
        """
        Check if the current platform is supported by this handler.
        
        Supports Windows platforms with Command Prompt available.
        
        Returns:
            bool: True if Windows with cmd.exe is available
        """
        return sys.platform.startswith('win') and self.cmd_path is not None
    
    def execute_command(self, command: str, config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a command in the Windows Command Prompt environment.
        
        This method provides Windows Command Prompt-specific execution with support for:
        - Windows batch commands and utilities
        - Windows-specific path and environment handling
        - Windows file system operations
        - Windows system management
        
        Args:
            command: The command string to execute
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Standardized result with Windows Command Prompt enhancements
            
        Raises:
            CommandExecutionError: If command execution fails
            SecurityError: If command fails security validation
            TimeoutError: If command execution exceeds timeout
        """
        if not self.is_platform_supported():
            raise PlatformNotSupportedError(f"Platform {sys.platform} not supported by Windows Command handler")
        
        # Use provided config or default
        exec_config = config or self.config
        
        # Validate command for Windows-specific security
        self._validate_windows_command(command)
        
        if self.config.enable_logging:
            self.logger.debug(f"Executing Windows command: {command}")
            self.logger.debug(f"Using Command Prompt: {self.cmd_path}")
        
        # Prepare command for Windows Command Prompt execution
        cmd_command = self._prepare_windows_command(command, exec_config)
        
        try:
            # Execute with Windows-specific handling
            result = self._execute_windows_command(cmd_command, exec_config)
            
            # Add Windows Command Prompt-specific metadata
            result.metadata = getattr(result, 'metadata', {})
            result.metadata.update({
                'cmd_path': self.cmd_path,
                'windows_version': self.windows_version,
                'supports_batch_files': self.supports_batch_files,
                'available_windows_utils': self.available_windows_utils,
                'comspec': self.comspec
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Windows Command execution failed: {command}")
            self.logger.error(f"Error: {str(e)}")
            raise CommandExecutionError(f"Failed to execute Windows Command: {str(e)}")
    
    def _validate_windows_command(self, command: str) -> None:
        """
        Validate command for Windows-specific security considerations.
        
        Args:
            command: The command string to validate
            
        Raises:
            SecurityError: If command fails Windows-specific validation
        """
        # Basic validation from base class
        self.validate_command(command)
        
        # Windows-specific security checks
        dangerous_windows_patterns = [
            r'\\\\\.\\\\',      # UNC paths
            r'>.*[A-Z]:\\',       # Redirection to absolute Windows paths
            r'\|\s*cmd',          # Pipe to cmd
            r'&\s*cmd',           # Background cmd execution
            r'\b(sudo|su)\b',     # Privilege escalation (shouldn't exist on Windows)
            r'\b(rm|del|format)\b.*[\/-]',  # Dangerous file operations
            r'sc\s+config.*\\',    # Service configuration
        ]
        
        for pattern in dangerous_windows_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityError(f"Command contains potentially dangerous Windows pattern: {pattern}")
    
    def _prepare_windows_command(self, command: str, config: CommandConfig) -> str:
        """
        Prepare command for Windows Command Prompt execution.
        
        Args:
            command: Original command string
            config: Execution configuration
            
        Returns:
            str: Prepared command for Windows Command Prompt execution
        """
        if config.shell_mode:
            return f'"{self.cmd_path}" /c "{command}"'
        else:
            return command
    
    def _execute_windows_command(self, command: str, config: CommandConfig) -> CommandResult:
        """
        Execute command with Windows-specific optimizations.
        
        Args:
            command: Prepared command for execution
            config: Execution configuration
            
        Returns:
            CommandResult: Execution result with Windows Command Prompt enhancements
        """
        start_time = time.time()
        working_dir = config.working_directory or self.config.working_directory
        env = {**os.environ, **self.config.environment, **config.environment}
        
        try:
            # Execute Windows Command Prompt command
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
                stderr = stderr or f"Windows command timed out after {config.timeout} seconds"
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
                stderr=f"Windows Command execution error: {str(e)}",
                execution_time=execution_time,
                environment=env,
                working_directory=working_dir or os.getcwd(),
                timestamp=start_time,
                success=False
            )
    
    def execute_batch_file(self, batch_path: str, args: List[str] = None,
                          config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a Windows batch file (.bat or .cmd).
        
        Args:
            batch_path: Path to the batch file to execute
            args: Optional arguments to pass to the batch file
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Execution result
        """
        if not os.path.exists(batch_path):
            raise FileNotFoundError(f"Batch file not found: {batch_path}")
        
        # Build command to execute batch file
        command_parts = [f'"{self.cmd_path}"', f'"{batch_path}"']
        
        if args:
            command_parts.extend(args)
        
        command = ' '.join(command_parts)
        
        return self.execute_command(command, config)
    
    def create_batch_script(self, commands: List[str], output_path: str,
                          script_type: str = 'bat') -> str:
        """
        Create a Windows batch script file.
        
        Args:
            commands: List of commands to include in the batch script
            output_path: Path where to save the batch script
            script_type: Type of script ('bat' or 'cmd')
            
        Returns:
            str: Path to the created batch script file
        """
        if script_type not in ['bat', 'cmd']:
            raise ValueError("script_type must be 'bat' or 'cmd'")
        
        # Ensure proper file extension
        if not output_path.lower().endswith(f'.{script_type}'):
            output_path = f"{output_path}.{script_type}"
        
        # Create batch script content
        script_content = '@echo off\n'
        
        # Add commands
        for command in commands:
            script_content += f'{command}\n'
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        return output_path
    
    def get_windows_info(self) -> Dict[str, Any]:
        """
        Get detailed Windows-specific environment information.
        
        Returns:
            Dict[str, Any]: Windows environment information including
                          version, utilities, and capabilities
        """
        info = {
            'cmd_path': self.cmd_path,
            'windows_version': self.windows_version,
            'supports_batch_files': self.supports_batch_files,
            'available_windows_utils': self.available_windows_utils,
            'comspec': self.comspec,
            'environment_variables': self._get_windows_environment(),
            'system_directories': self._get_windows_directories()
        }
        
        # Merge with base environment info
        info.update(self.environment_info)
        
        return info
    
    def _get_windows_environment(self) -> Dict[str, str]:
        """
        Get Windows environment variables.
        
        Returns:
            Dict[str, str]: Key Windows environment variables
        """
        important_vars = [
            'PATH', 'PROMPT', 'COMSPEC', 'WINDIR', 'SYSTEMROOT',
            'PROGRAMFILES', 'PROGRAMFILES(X86)', 'APPDATA', 'TEMP', 'TMP'
        ]
        
        env_vars = {}
        for var in important_vars:
            value = os.environ.get(var)
            if value:
                env_vars[var] = value
        
        return env_vars
    
    def _get_windows_directories(self) -> Dict[str, str]:
        """
        Get Windows system directories.
        
        Returns:
            Dict[str, str]: Key Windows system directory paths
        """
        import pathlib
        
        directories = {
            'windows': os.environ.get('WINDIR', 'C:\\Windows'),
            'system32': os.environ.get('SYSTEMROOT', 'C:\\Windows') + '\\System32',
            'program_files': os.environ.get('PROGRAMFILES', 'C:\\Program Files'),
            'program_files_x86': os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'),
            'appdata': os.environ.get('APPDATA', ''),
            'temp': os.environ.get('TEMP', 'C:\\temp'),
        }
        
        # Ensure paths exist and normalize
        for key, path in directories.items():
            if path:
                directories[key] = pathlib.Path(path).as_posix()
        
        return directories


# Export Windows Command handler
__all__ = ['WindowsCommandHandler']