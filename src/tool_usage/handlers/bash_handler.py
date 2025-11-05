"""
Unix/Linux/macOS Shell Handler for mini-biai-1

This module provides specialized command execution capabilities for Unix-based
systems including Linux, macOS, and other Unix-like operating systems. It
optimizes for bash, zsh, sh, and other Unix shells while providing advanced
features like signal handling, process management, and Unix-specific functionality.

The Unix shell handler implements:
- Cross-platform Unix shell execution (bash, zsh, sh)
- Unix-style path handling and environment variable management
- Advanced signal handling and process control
- Unix pipeline and redirection support
- File permission and ownership management
- Unix-specific utilities and system integration

┌─────────────────────────────────────────────────────────────────────┐
│                    Unix Shell Handler Architecture                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   Shell     │ │   Unix      │ │  Process    │
    │ Execution   │ │   Utilities │ │  Management │
    │             │ │             │ │             │
    │ • Bash      │ │ • chmod     │ │ • Signals   │
    │ • zsh       │ │ • chown     │ │ • Jobs      │
    │ • sh        │ │ • find      │ │ • Timeout   │
    └─────────────┘ └─────────────┘ └─────────────┘

Key Features:
- Multi-shell support with automatic detection
- Unix-specific path and environment handling
- Advanced signal handling and process management
- Unix pipeline and I/O redirection support
- File system permission and ownership management
- Job control and background process handling
- Unix utility integration (chmod, chown, find, etc.)
- Signal handling for graceful process termination
- Cross-platform Unix compatibility (Linux, macOS, BSD)

Architecture Benefits:
- Optimized for Unix philosophy and conventions
- Advanced process and signal management
- Comprehensive Unix utility support
- Cross-platform Unix compatibility
- Advanced security and sandboxing options

Dependencies:
    Core: subprocess, os, signal, fcntl, pty (Unix process control)
    Unix: pwd, grp (user/group information), stat (file permissions)
    System: resource (resource limits), time (timing operations)
    Utils: pathlib (path handling), typing (type annotations)

Error Handling:
    The Unix shell handler implements comprehensive error handling:
        - Signal-based process termination and cleanup
        - Unix error code propagation and translation
        - Permission and access error handling
        - Resource limit enforcement and monitoring
        - Cross-platform Unix compatibility management
        - Graceful fallback for missing Unix utilities

Usage Examples:

Basic Unix Shell Execution:
    >>> from src.tool_usage.handlers.bash_handler import UnixShellHandler
    >>> 
    >>> handler = UnixShellHandler()
    >>> result = handler.execute_command("ls -la /tmp")
    >>> print(f"Exit code: {result.exit_code}")
    >>> print(f"Files:\n{result.stdout}")

Unix Pipeline Execution:
    >>> # Unix pipeline with grep and sort
    >>> command = "ps aux | grep python | grep -v grep | sort -k2 -nr"
    >>> result = handler.execute_command(command)
    >>> print(f"Python processes sorted by memory usage:")
    >>> print(result.stdout)

File Operations with Permissions:
    >>> # Create file with specific permissions
    >>> command = "touch /tmp/test.txt && chmod 644 /tmp/test.txt && ls -l /tmp/test.txt"
    >>> result = handler.execute_command(command)
    >>> print(f"File created with permissions:\n{result.stdout}")

Background Job Management:
    >>> # Start background job and manage it
    >>> handler = UnixShellHandler()
    >>> 
    >>> # Start long-running process in background
    >>> result = handler.execute_command("sleep 300 &", config=handler.config(timeout=5.0))
    >>> print(f"Background job started: {result.stdout}")

Environment Variable Management:
    >>> # Set environment variables and execute command
    >>> from src.tool_usage.handlers.base_handler import CommandConfig
    >>> 
    >>> config = CommandConfig(
    ...     environment={"PATH": "/custom/path:$PATH", "DEBUG": "1"},
    ...     working_directory="/home/user"
    ... )
    >>> result = handler.execute_command("echo $PATH && echo $DEBUG", config=config)
    >>> print(f"Environment test:\n{result.stdout}")

Signal Handling and Process Management:
    >>> # Handle long-running processes with signal management
    >>> import signal
    >>> 
    >>> def timeout_handler(signum, frame):
    ...     print("Process timeout detected!")
    >>> 
    >>> signal.signal(signal.SIGALRM, timeout_handler)
    >>> signal.alarm(10)  # 10 second timeout
    >>> 
    >>> try:
    ...     result = handler.execute_command("tail -f /var/log/syslog")
    ... finally:
    ...     signal.alarm(0)  # Cancel timeout

Unix Utility Integration:
    >>> # Use Unix utilities for file operations
    >>> command = "find /home -name '*.py' -type f -exec wc -l {} + | sort -n"
    >>> result = handler.execute_command(command)
    >>> print(f"Python files by line count:\n{result.stdout}")

Advanced Pipeline Processing:
    >>> # Complex Unix pipeline with multiple stages
    >>> command = """
    ... netstat -tuln | \\
    ...     grep LISTEN | \\
    ...     awk '{print $4}' | \\
    ...     cut -d':' -f2 | \\
    ...     sort -u | \\
    ...     while read port; do
    ...         echo "Port $port is listening"
    ...     done
    ... """
    >>> result = handler.execute_command(command)
    >>> print(f"Listening ports:\n{result.stdout}")

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

import subprocess
import os
import sys
import signal
import fcntl
import pty
import pwd
import grp
import stat
import resource
import time
import pathlib
import re
from typing import Dict, List, Optional, Union, Any, Tuple

from .base_handler import BaseCommandHandler, CommandResult, CommandConfig


class UnixShellHandler(BaseCommandHandler):
    """
    Specialized command handler for Unix/Linux/macOS shell environments.
    
    This handler provides optimized command execution for Unix-based systems,
    supporting bash, zsh, sh, and other Unix shells. It implements advanced
    features like signal handling, process management, Unix pipeline support,
    and Unix-specific utilities integration.
    
    The handler automatically detects the available shell and provides
    platform-specific optimizations for:
    - Shell execution and path handling
    - Unix utilities integration
    - Process and signal management
    - File system operations
    - Environment variable management
    
    Attributes:
        shell_path (str): Path to the detected shell executable
        shell_name (str): Name of the detected shell (bash, zsh, sh, etc.)
        supports_signals (bool): Whether the system supports Unix signals
        supports_pty (bool): Whether pseudo-terminal operations are supported
    
    Example Usage:
        >>> handler = UnixShellHandler()
        >>> 
        >>> # Basic command execution
        >>> result = handler.execute_command("echo 'Hello Unix!'")
        >>> print(f"Output: {result.stdout}")
        >>> 
        >>> # Unix pipeline execution
        >>> result = handler.execute_command("ls -1 | grep '\.py$'")
        >>> print(f"Python files: {result.stdout}")
        
        >>> # File operations with permissions
        >>> result = handler.execute_command("touch /tmp/test && chmod 755 /tmp/test")
        >>> print(f"File operation result: {result.success}")
    """
    
    def __init__(self, config: Optional[CommandConfig] = None):
        """
        Initialize the Unix shell handler.
        
        Args:
            config: Optional configuration for command execution.
                   Uses Unix-specific optimizations if not provided.
        """
        super().__init__(config)
        self.shell_path = self._detect_shell()
        self.shell_name = os.path.basename(self.shell_path)
        self.supports_signals = self._check_signal_support()
        self.supports_pty = self._check_pty_support()
        
        if self.config.enable_logging:
            self.logger.info(f"Unix Shell Handler initialized:")
            self.logger.info(f"  Shell: {self.shell_name} ({self.shell_path})")
            self.logger.info(f"  Signals: {'✓' if self.supports_signals else '✗'}")
            self.logger.info(f"  PTY: {'✓' if self.supports_pty else '✗'}")
    
    def _detect_shell(self) -> str:
        """
        Detect the best available Unix shell for execution.
        
        Checks for shells in order of preference: bash, zsh, sh.
        Falls back to the default system shell if none are found.
        
        Returns:
            str: Path to the detected shell executable
        """
        preferred_shells = ['bash', 'zsh', 'sh']
        
        # Check environment variable first
        if 'SHELL' in os.environ:
            shell_path = os.environ['SHELL']
            if os.path.exists(shell_path):
                return shell_path
        
        # Check for preferred shells in PATH
        for shell_name in preferred_shells:
            shell_path = self._find_executable(shell_name)
            if shell_path:
                return shell_path
        
        # Fallback to /bin/sh
        return '/bin/sh'
    
    def _find_executable(self, name: str) -> Optional[str]:
        """
        Find executable in system PATH.
        
        Args:
            name: Name of the executable to find
            
        Returns:
            Optional[str]: Path to executable if found, None otherwise
        """
        import shutil
        return shutil.which(name)
    
    def _check_signal_support(self) -> bool:
        """
        Check if the system supports Unix signal handling.
        
        Returns:
            bool: True if signal handling is supported
        """
        try:
            # Check if signal module is available and working
            signal.SIGTERM
            return True
        except (AttributeError, OSError):
            return False
    
    def _check_pty_support(self) -> bool:
        """
        Check if the system supports pseudo-terminal operations.
        
        Returns:
            bool: True if PTY operations are supported
        """
        try:
            # Check if pty module is available
            import pty
            return True
        except ImportError:
            return False
    
    def _is_platform_supported(self) -> bool:
        """
        Check if the current platform is supported by this handler.
        
        Supports Unix-like systems including Linux, macOS, BSD, etc.
        
        Returns:
            bool: True if platform is Unix-like, False otherwise
        """
        return sys.platform.startswith(('linux', 'darwin', 'freebsd', 'openbsd', 'netbsd'))
    
    def execute_command(self, command: str, config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a command in the Unix shell environment.
        
        This method provides Unix-specific command execution with support for:
        - Shell scripts and Unix pipelines
        - Unix utilities and system integration
        - Signal handling and process management
        - Unix-style path and environment handling
        
        Args:
            command: The command string to execute
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Standardized result with Unix-specific enhancements
            
        Raises:
            CommandExecutionError: If command execution fails
            SecurityError: If command fails security validation
            TimeoutError: If command execution exceeds timeout
        """
        if not self.is_platform_supported():
            raise PlatformNotSupportedError(f"Platform {sys.platform} not supported by Unix shell handler")
        
        # Use provided config or default
        exec_config = config or self.config
        
        # Validate command for Unix-specific security
        self._validate_unix_command(command)
        
        if self.config.enable_logging:
            self.logger.debug(f"Executing Unix command: {command}")
            self.logger.debug(f"Using shell: {self.shell_path}")
        
        # Prepare command for shell execution
        shell_command = self._prepare_shell_command(command, exec_config)
        
        try:
            # Execute with Unix-specific handling
            result = self._execute_unix_command(shell_command, exec_config)
            
            # Add Unix-specific metadata
            result.metadata = getattr(result, 'metadata', {})
            result.metadata.update({
                'shell_used': self.shell_name,
                'shell_path': self.shell_path,
                'unix_platform': sys.platform,
                'supports_signals': self.supports_signals,
                'supports_pty': self.supports_pty
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Unix command execution failed: {command}")
            self.logger.error(f"Error: {str(e)}")
            raise CommandExecutionError(f"Failed to execute Unix command: {str(e)}")
    
    def _validate_unix_command(self, command: str) -> None:
        """
        Validate command for Unix-specific security considerations.
        
        Args:
            command: The command string to validate
            
        Raises:
            SecurityError: If command fails Unix-specific validation
        """
        # Basic validation from base class
        self.validate_command(command)
        
        # Unix-specific security checks
        dangerous_unix_patterns = [
            r'\b(sudo|su)\b',  # Privilege escalation
            r'>/etc/',         # Writing to system directories
            r'rm\s+-rf\s+/',   # Recursive removal from root
            r'\|\s*sh\s*$',    # Pipe to shell
            r'&\s*sh\s*$',     # Background shell execution
        ]
        
        for pattern in dangerous_unix_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityError(f"Command contains potentially dangerous Unix pattern: {pattern}")
    
    def _prepare_shell_command(self, command: str, config: CommandConfig) -> str:
        """
        Prepare command for Unix shell execution.
        
        Args:
            command: Original command string
            config: Execution configuration
            
        Returns:
            str: Prepared command for shell execution
        """
        # Use detected shell
        if config.shell_mode:
            return f"{self.shell_path} -c '{command}'"
        else:
            return command
    
    def _execute_unix_command(self, command: str, config: CommandConfig) -> CommandResult:
        """
        Execute command with Unix-specific optimizations.
        
        Args:
            command: Prepared command for execution
            config: Execution configuration
            
        Returns:
            CommandResult: Execution result with Unix-specific enhancements
        """
        start_time = time.time()
        working_dir = config.working_directory or self.config.working_directory
        env = {**os.environ, **self.config.environment, **config.environment}
        
        try:
            if config.shell_mode and self.supports_pty:
                # Use PTY for better Unix compatibility
                return self._execute_with_pty(command, config)
            else:
                # Standard subprocess execution
                return self._execute_subprocess(command, config)
                
        except Exception as e:
            execution_time = time.time() - start_time
            
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"Unix execution error: {str(e)}",
                execution_time=execution_time,
                environment=env,
                working_directory=working_dir or os.getcwd(),
                timestamp=start_time,
                success=False
            )
    
    def _execute_with_pty(self, command: str, config: CommandConfig) -> CommandResult:
        """
        Execute command using pseudo-terminal for better Unix compatibility.
        
        Args:
            command: Command to execute
            config: Execution configuration
            
        Returns:
            CommandResult: Execution result
        """
        import pty
        import select
        import sys
        
        start_time = time.time()
        
        try:
            # Create pseudo-terminal
            master_fd, slave_fd = pty.openpty()
            
            # Set terminal attributes
            try:
                import termios
                import tty
                # Configure terminal for Unix compatibility
                attrs = termios.tcgetattr(slave_fd)
                attrs[3] = attrs[3] & ~(termios.ECHO | termios.ICANON)  # Disable echo and canonical mode
                termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
            except:
                pass
            
            # Start process in pseudo-terminal
            process = subprocess.Popen(
                command,
                shell=True,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True
            )
            
            # Close slave fd in parent process
            os.close(slave_fd)
            
            # Set master fd to non-blocking
            fcntl.fcntl(master_fd, fcntl.F_SETFL, fcntl.O_NONBLOCK)
            
            # Read output
            stdout_data = b""
            stderr_data = b""
            start_read_time = time.time()
            
            while True:
                # Check for timeout
                if time.time() - start_time > config.timeout:
                    try:
                        process.terminate()
                        time.sleep(0.1)
                        if process.poll() is None:
                            process.kill()
                    except:
                        pass
                    stderr_data += f"Command timed out after {config.timeout} seconds".encode()
                    break
                
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Read available data
                try:
                    while True:
                        data = os.read(master_fd, 1024)
                        if not data:
                            break
                        stdout_data += data
                except OSError:
                    pass
                
                # Limit read time to prevent infinite loop
                if time.time() - start_read_time > config.timeout:
                    break
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
            
            # Final read
            try:
                while True:
                    data = os.read(master_fd, 1024)
                    if not data:
                        break
                    stdout_data += data
            except OSError:
                pass
            
            # Get process status
            exit_code = process.wait() if process.poll() is None else process.returncode
            
            # Close master fd
            os.close(master_fd)
            
            execution_time = time.time() - start_time
            
            # Decode output
            try:
                stdout = stdout_data.decode(config.encoding, errors='replace')
            except:
                stdout = stdout_data.decode('utf-8', errors='replace')
            
            # Truncate output if too large
            if len(stdout) > config.max_output_size:
                stdout = stdout[:config.max_output_size] + "\n[Output truncated...]"
            
            result = CommandResult(
                command=command,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr_data.decode(config.encoding, errors='replace') if stderr_data else "",
                execution_time=execution_time,
                environment=env,
                working_directory=config.working_directory or os.getcwd(),
                timestamp=start_time,
                success=exit_code == 0
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            os.close(master_fd)
            
            return CommandResult(
                command=command,
                exit_code=-1,
                stdout="",
                stderr=f"PTY execution error: {str(e)}",
                execution_time=execution_time,
                environment=env,
                working_directory=config.working_directory or os.getcwd(),
                timestamp=start_time,
                success=False
            )
    
    def get_unix_info(self) -> Dict[str, Any]:
        """
        Get detailed Unix-specific environment information.
        
        Returns:
            Dict[str, Any]: Unix environment information including
                          shell details, user info, and system capabilities
        """
        info = {
            'shell_path': self.shell_path,
            'shell_name': self.shell_name,
            'supports_signals': self.supports_signals,
            'supports_pty': self.supports_pty,
            'user_info': self._get_user_info(),
            'group_info': self._get_group_info(),
            'resource_limits': self._get_resource_limits(),
            'available_unix_utils': self._get_available_unix_utils()
        }
        
        # Merge with base environment info
        info.update(self.environment_info)
        
        return info
    
    def _get_user_info(self) -> Dict[str, str]:
        """Get current user information."""
        try:
            user_info = pwd.getpwuid(os.getuid())
            return {
                'username': user_info.pw_name,
                'uid': str(user_info.pw_uid),
                'gid': str(user_info.pw_gid),
                'home': user_info.pw_dir,
                'shell': user_info.pw_shell
            }
        except:
            return {}
    
    def _get_group_info(self) -> Dict[str, Any]:
        """Get current group information."""
        try:
            group_info = grp.getgrgid(os.getgid())
            return {
                'groupname': group_info.gr_name,
                'gid': str(group_info.gr_gid),
                'members': group_info.gr_mem
            }
        except:
            return {}
    
    def _get_resource_limits(self) -> Dict[str, int]:
        """Get system resource limits."""
        try:
            return {
                'max_file_size': resource.getrlimit(resource.RLIMIT_FSIZE)[0],
                'max_memory': resource.getrlimit(resource.RLIMIT_AS)[0],
                'max_processes': resource.getrlimit(resource.RLIMIT_NPROC)[0],
                'max_open_files': resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            }
        except:
            return {}
    
    def _get_available_unix_utils(self) -> Dict[str, bool]:
        """Check availability of common Unix utilities."""
        utils_to_check = [
            'bash', 'zsh', 'sh', 'find', 'grep', 'awk', 'sed', 'sort', 'cut',
            'chmod', 'chown', 'ls', 'ps', 'top', 'kill', 'killall', 'ps',
            'netstat', 'ss', 'df', 'du', 'tar', 'gzip', 'unzip', 'curl', 'wget'
        ]
        
        available_utils = {}
        for util in utils_to_check:
            available_utils[util] = self._find_executable(util) is not None
        
        return available_utils
    
    def set_resource_limits(self, max_memory: int = None, max_processes: int = None, 
                          max_file_size: int = None) -> None:
        """
        Set Unix resource limits for this handler.
        
        Args:
            max_memory: Maximum memory limit in bytes
            max_processes: Maximum number of processes
            max_file_size: Maximum file size in bytes
        """
        try:
            limits = {}
            
            if max_memory is not None:
                limits[resource.RLIMIT_AS] = (max_memory, max_memory)
            
            if max_processes is not None:
                limits[resource.RLIMIT_NPROC] = (max_processes, max_processes)
            
            if max_file_size is not None:
                limits[resource.RLIMIT_FSIZE] = (max_file_size, max_file_size)
            
            for resource_type, (soft_limit, hard_limit) in limits.items():
                try:
                    resource.setrlimit(resource_type, (soft_limit, hard_limit))
                    if self.config.enable_logging:
                        self.logger.debug(f"Set resource limit for {resource_type}")
                except (ValueError, OSError) as e:
                    if self.config.enable_logging:
                        self.logger.warning(f"Failed to set resource limit: {e}")
                    
        except Exception as e:
            if self.config.enable_logging:
                self.logger.warning(f"Failed to set resource limits: {e}")


# Import sys for platform detection in _is_platform_supported
import sys


# Export Unix shell handler
__all__ = ['UnixShellHandler']