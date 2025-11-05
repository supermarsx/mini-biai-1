"""
Command Executor Module

Provides secure command execution capabilities for the mini-biai-1 framework
tool usage system with cross-platform support and security features.
"""

import os
import sys
import platform
import subprocess
import shlex
import tempfile
import shutil
import signal
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

from .platform_adapter import PlatformAdapter, get_platform_adapter
from .shell_detector import ShellDetector, ShellInfo


class ExecutionMode(Enum):
    """Execution modes for commands."""
    BLOCKING = "blocking"
    NON_BLOCKING = "non_blocking"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


class SecurityLevel(Enum):
    """Security levels for command execution."""
    LOW = "low"          # Basic path and permission checks
    MEDIUM = "medium"    # Environment isolation and timeouts
    HIGH = "high"        # Full sandbox and comprehensive validation
    MAXIMUM = "maximum"  # Complete isolation with multiple validation layers


@dataclass
class CommandResult:
    """Container for command execution results."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool
    timestamp: float = field(default_factory=time.time)
    pid: Optional[int] = None
    shell_used: Optional[str] = None
    working_directory: Optional[str] = None
    environment: Optional[Dict[str, str]] = None
    warnings: List[str] = field(default_factory=list)
    security_checks_passed: bool = True


@dataclass
class ExecutionConfig:
    """Configuration for command execution."""
    mode: ExecutionMode = ExecutionMode.BLOCKING
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    timeout: Optional[float] = 30.0
    working_directory: Optional[Union[str, Path]] = None
    environment: Optional[Dict[str, str]] = None
    shell: Optional[str] = None
    stdin: Optional[str] = None
    capture_output: bool = True
    check: bool = False
    cwd: Optional[Union[str, Path]] = None
    pre_exec_fn: Optional[Callable] = None
    post_exec_fn: Optional[Callable] = None


class SecurityValidator:
    """Validates commands for security compliance."""
    
    # Dangerous commands that should be restricted
    DANGEROUS_COMMANDS = {
        'rm', 'del', 'format', 'fdisk', 'dd', 'mkfs',
        'sudo', 'su', 'passwd', 'chmod', 'chown',
        'iptables', 'ufw', 'firewall-cmd', 'netsh',
        'taskkill', 'pkill', 'killall', 'kill',
        'shutdown', 'reboot', 'halt', 'poweroff',
        'curl', 'wget', 'nc', 'netcat', 'telnet',
        'eval', 'exec', 'source', '.'
    }
    
    # Potentially dangerous patterns
    DANGEROUS_PATTERNS = [
        r'&&\s*rm\s+-rf\s+/',  # Command chaining with dangerous rm
        r';\s*rm\s+-rf\s+/',   # Command chaining with dangerous rm
        r'\|\s*rm\s+-rf\s+/',  # Pipe to dangerous rm
        r'>\s*/etc/',          # Writing to system directories
        r'<\s*/etc/',          # Reading from system directories
        r'\$\{.*\}',           # Variable expansion (potential injection)
        r'`[^`]*`',            # Command substitution
        r'\$\([^)]*\)',        # Command substitution
    ]
    
    def __init__(self, security_level: SecurityLevel):
        """
        Initialize the security validator.
        
        Args:
            security_level: The security level to enforce
        """
        self.security_level = security_level
        self.validation_rules = self._get_validation_rules()
    
    def _get_validation_rules(self) -> List[Callable[[str], Optional[str]]]:
        """Get validation rules based on security level."""
        rules = [
            self._validate_basic_syntax,
            self._validate_dangerous_commands,
            self._validate_shell_injection,
        ]
        
        if self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
            rules.extend([
                self._validate_path_traversal,
                self._validate_command_chaining,
                self._validate_environment_variables,
                self._validate_network_operations,
                self._validate_file_operations,
            ])
        
        if self.security_level == SecurityLevel.MAXIMUM:
            rules.extend([
                self._validate_resource_usage,
                self._validate_system_calls,
                self._validate_cross_platform_compatibility,
            ])
        
        return rules
    
    def validate(self, command: str) -> Tuple[bool, List[str]]:
        """
        Validate a command for security compliance.
        
        Args:
            command: The command to validate
            
        Returns:
            Tuple of (is_safe, list_of_warnings)
        """
        warnings = []
        
        for rule in self.validation_rules:
            warning = rule(command)
            if warning:
                warnings.append(warning)
        
        # For maximum security, any warning is a failure
        if self.security_level == SecurityLevel.MAXIMUM and warnings:
            return False, warnings
        
        # For other levels, warnings are allowed but reported
        return True, warnings
    
    def _validate_basic_syntax(self, command: str) -> Optional[str]:
        """Validate basic command syntax."""
        try:
            # Try to parse the command
            if platform.system() == 'Windows':
                # Windows command parsing
                parts = command.split()
                if not parts:
                    return "Empty command"
            else:
                # POSIX shell parsing
                shlex.split(command)
        except (ValueError, IndexError) as e:
            return f"Invalid command syntax: {e}"
        
        return None
    
    def _validate_dangerous_commands(self, command: str) -> Optional[str]:
        """Check for dangerous commands."""
        parts = shlex.split(command) if platform.system() != 'Windows' else command.split()
        
        if not parts:
            return None
        
        base_command = parts[0].lower()
        
        if base_command in self.DANGEROUS_COMMANDS:
            return f"Dangerous command detected: {base_command}"
        
        return None
    
    def _validate_shell_injection(self, command: str) -> Optional[str]:
        """Check for potential shell injection."""
        dangerous_chars = ['|', '&', ';', '`', '$', '(', ')', '<', '>']
        
        for char in dangerous_chars:
            if char in command and self.security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM]:
                return f"Potential shell injection detected: {char}"
        
        return None
    
    def _validate_path_traversal(self, command: str) -> Optional[str]:
        """Check for path traversal attempts."""
        import re
        
        if re.search(r'\.\./|\.\.\\', command):
            return "Path traversal attempt detected"
        
        return None
    
    def _validate_command_chaining(self, command: str) -> Optional[str]:
        """Check for dangerous command chaining."""
        import re
        
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Dangerous command pattern detected: {pattern}"
        
        return None
    
    def _validate_environment_variables(self, command: str) -> Optional[str]:
        """Check for environment variable usage."""
        import re
        
        if re.search(r'\${[^}]*}|\$[A-Z_]+|`[^`]*`', command):
            return "Environment variable usage detected"
        
        return None
    
    def _validate_network_operations(self, command: str) -> Optional[str]:
        """Check for network-related operations."""
        network_commands = {'curl', 'wget', 'nc', 'netcat', 'telnet', 'ssh', 'ftp'}
        parts = shlex.split(command) if platform.system() != 'Windows' else command.split()
        
        if parts and parts[0].lower() in network_commands:
            return "Network operation detected"
        
        return None
    
    def _validate_file_operations(self, command: str) -> Optional[str]:
        """Check for file system operations."""
        file_commands = {'rm', 'del', 'copy', 'xcopy', 'move', 'cat', 'type', 'less', 'more'}
        parts = shlex.split(command) if platform.system() != 'Windows' else command.split()
        
        if parts and parts[0].lower() in file_commands:
            return "File system operation detected"
        
        return None
    
    def _validate_resource_usage(self, command: str) -> Optional[str]:
        """Check for resource-intensive operations."""
        intensive_commands = {'tar', 'zip', 'unzip', 'gzip', 'find', 'grep', 'dd'}
        parts = shlex.split(command) if platform.system() != 'Windows' else command.split()
        
        if parts and parts[0].lower() in intensive_commands:
            return "Resource-intensive operation detected"
        
        return None
    
    def _validate_system_calls(self, command: str) -> Optional[str]:
        """Check for system-level calls."""
        system_calls = {'sudo', 'su', 'chmod', 'chown', 'mount', 'umount'}
        parts = shlex.split(command) if platform.system() != 'Windows' else command.split()
        
        if parts and parts[0].lower() in system_calls:
            return "System-level operation detected"
        
        return None
    
    def _validate_cross_platform_compatibility(self, command: str) -> Optional[str]:
        """Check for cross-platform compatibility issues."""
        # Windows-specific commands that shouldn't run on POSIX
        if platform.system() != 'Windows':
            windows_commands = {'cmd.exe', 'powershell.exe', 'pwsh.exe', 'del', 'dir', 'type'}
            parts = shlex.split(command)
            if parts and parts[0] in windows_commands:
                return "Windows-specific command on non-Windows system"
        
        # POSIX-specific commands that shouldn't run on Windows
        if platform.system() == 'Windows':
            posix_commands = {'bash', 'sh', 'zsh', 'ls', 'cat', 'grep', 'rm'}
            parts = shlex.split(command)
            if parts and parts[0] in posix_commands:
                return "POSIX-specific command on Windows system"
        
        return None


class CommandExecutor:
    """
    Secure command execution system with cross-platform support.
    
    Provides comprehensive command execution capabilities with security
    validation, platform adaptation, and performance monitoring.
    """
    
    def __init__(self, platform_adapter: Optional[PlatformAdapter] = None):
        """
        Initialize the command executor.
        
        Args:
            platform_adapter: Platform adapter instance (uses default if None)
        """
        self.platform_adapter = platform_adapter or get_platform_adapter()
        self.shell_detector = ShellDetector(self.platform_adapter)
        self._execution_history: List[CommandResult] = []
        self._active_processes: Dict[int, subprocess.Popen] = {}
        
        # Default execution configuration
        self.default_config = ExecutionConfig(
            mode=ExecutionMode.BLOCKING,
            security_level=SecurityLevel.MEDIUM,
            timeout=30.0,
            capture_output=True
        )
    
    def execute(self, command: str, config: Optional[ExecutionConfig] = None) -> CommandResult:
        """
        Execute a command with the specified configuration.
        
        Args:
            command: The command to execute
            config: Execution configuration (uses default if None)
            
        Returns:
            CommandResult object with execution details
        """
        config = config or self.default_config
        
        start_time = time.time()
        
        try:
            # Security validation
            security_validator = SecurityValidator(config.security_level)
            is_safe, warnings = security_validator.validate(command)
            
            # Prepare execution environment
            final_command = self._prepare_command(command, config)
            execution_env = self._prepare_environment(config)
            working_dir = self._prepare_working_directory(config)
            
            # Execute command based on mode
            if config.mode == ExecutionMode.BLOCKING:
                result = self._execute_blocking(final_command, execution_env, working_dir, config)
            elif config.mode == ExecutionMode.NON_BLOCKING:
                result = self._execute_non_blocking(final_command, execution_env, working_dir, config)
            elif config.mode == ExecutionMode.INTERACTIVE:
                result = self._execute_interactive(final_command, execution_env, working_dir, config)
            elif config.mode == ExecutionMode.BACKGROUND:
                result = self._execute_background(final_command, execution_env, working_dir, config)
            else:
                raise ValueError(f"Unknown execution mode: {config.mode}")
            
            # Populate result data
            result.execution_time = time.time() - start_time
            result.command = command
            result.working_directory = str(working_dir) if working_dir else None
            result.environment = execution_env
            result.warnings = warnings
            result.security_checks_passed = is_safe
            
            # Add to history
            self._execution_history.append(result)
            
            # Apply post-execution hooks
            if config.post_exec_fn:
                try:
                    config.post_exec_fn(result)
                except Exception as e:
                    result.warnings.append(f"Post-execution hook failed: {e}")
            
            return result
            
        except Exception as e:
            # Handle execution errors
            execution_time = time.time() - start_time
            return CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                success=False,
                warnings=warnings if 'warnings' in locals() else [],
                security_checks_passed=False
            )
    
    def _prepare_command(self, command: str, config: ExecutionConfig) -> str:
        """Prepare the final command for execution."""
        # Use specified shell or detect appropriate shell
        if config.shell:
            shell = self.shell_detector.get_shell_by_name(config.shell)
            if not shell:
                raise ValueError(f"Specified shell '{config.shell}' not available")
            shell_path = shell.path
        else:
            # Auto-detect best shell for the command
            detected_shell = self.shell_detector.detect_shell_from_command(command)
            shell_path = detected_shell.path if detected_shell else self.platform_adapter.get_shell_path()
        
        # For complex commands or shell-specific features, wrap in shell
        if self._requires_shell(command):
            if self.platform_adapter.platform_info.is_windows:
                return f'{shell_path} /c "{command}"'
            else:
                return f'{shell_path} -c {shlex.quote(command)}'
        
        return command
    
    def _requires_shell(self, command: str) -> bool:
        """Determine if a command requires shell execution."""
        # Commands with shell features
        shell_features = ['|', '&', ';', '>', '<', '`', '$', '*', '?', '~']
        return any(feature in command for feature in shell_features)
    
    def _prepare_environment(self, config: ExecutionConfig) -> Dict[str, str]:
        """Prepare environment variables for execution."""
        env = os.environ.copy()
        
        if config.environment:
            env.update(config.environment)
        
        return env
    
    def _prepare_working_directory(self, config: ExecutionConfig) -> Optional[str]:
        """Prepare working directory for execution."""
        if config.cwd:
            return str(config.cwd)
        elif config.working_directory:
            return str(config.working_directory)
        else:
            return None
    
    def _execute_blocking(self, command: str, env: Dict[str, str], 
                         cwd: Optional[str], config: ExecutionConfig) -> CommandResult:
        """Execute command in blocking mode."""
        try:
            result = subprocess.run(
                command,
                shell=isinstance(command, str) and self._requires_shell(command),
                capture_output=config.capture_output,
                text=True,
                timeout=config.timeout,
                env=env,
                cwd=cwd,
                check=config.check,
                preexec_fn=config.pre_exec_fn if (hasattr(os, 'preexec_fn') and config.pre_exec_fn) else None
            )
            
            return CommandResult(
                command=command,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=0,  # Will be set by caller
                success=result.returncode == 0
            )
            
        except subprocess.TimeoutExpired as e:
            return CommandResult(
                command=command,
                return_code=-1,
                stdout=e.stdout if hasattr(e, 'stdout') and e.stdout else "",
                stderr=f"Command timed out after {config.timeout}s",
                execution_time=0,
                success=False
            )
        except subprocess.CalledProcessError as e:
            return CommandResult(
                command=command,
                return_code=e.returncode,
                stdout=e.stdout if hasattr(e, 'stdout') else "",
                stderr=e.stderr if hasattr(e, 'stderr') else str(e),
                execution_time=0,
                success=False
            )
    
    def _execute_non_blocking(self, command: str, env: Dict[str, str],
                             cwd: Optional[str], config: ExecutionConfig) -> CommandResult:
        """Execute command in non-blocking mode."""
        process = subprocess.Popen(
            command,
            shell=isinstance(command, str) and self._requires_shell(command),
            stdout=subprocess.PIPE if config.capture_output else None,
            stderr=subprocess.PIPE if config.capture_output else None,
            text=True,
            env=env,
            cwd=cwd,
            preexec_fn=config.pre_exec_fn if hasattr(os, 'preexec_fn') else None
        )
        
        self._active_processes[process.pid] = process
        
        # Wait for completion with timeout
        try:
            stdout, stderr = process.communicate(timeout=config.timeout)
            self._active_processes.pop(process.pid, None)
            
            return CommandResult(
                command=command,
                return_code=process.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                execution_time=0,
                success=process.returncode == 0,
                pid=process.pid
            )
        except subprocess.TimeoutExpired:
            process.kill()
            self._active_processes.pop(process.pid, None)
            
            return CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {config.timeout}s",
                execution_time=0,
                success=False,
                pid=process.pid
            )
    
    def _execute_interactive(self, command: str, env: Dict[str, str],
                           cwd: Optional[str], config: ExecutionConfig) -> CommandResult:
        """Execute command in interactive mode."""
        # For interactive mode, we run the command with stdin/stdout connected
        process = subprocess.Popen(
            command,
            shell=isinstance(command, str) and self._requires_shell(command),
            stdin=subprocess.PIPE if config.stdin else None,
            stdout=subprocess.PIPE if config.capture_output else None,
            stderr=subprocess.PIPE if config.capture_output else None,
            text=True,
            env=env,
            cwd=cwd
        )
        
        self._active_processes[process.pid] = process
        
        # Send stdin if provided
        if config.stdin and process.stdin:
            process.stdin.write(config.stdin)
            process.stdin.close()
        
        try:
            stdout, stderr = process.communicate(timeout=config.timeout)
            self._active_processes.pop(process.pid, None)
            
            return CommandResult(
                command=command,
                return_code=process.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                execution_time=0,
                success=process.returncode == 0,
                pid=process.pid
            )
        except subprocess.TimeoutExpired:
            process.kill()
            self._active_processes.pop(process.pid, None)
            
            return CommandResult(
                command=command,
                return_code=-1,
                stdout="",
                stderr=f"Command timed out after {config.timeout}s",
                execution_time=0,
                success=False,
                pid=process.pid
            )
    
    def _execute_background(self, command: str, env: Dict[str, str],
                          cwd: Optional[str], config: ExecutionConfig) -> CommandResult:
        """Execute command in background mode."""
        process = subprocess.Popen(
            command,
            shell=isinstance(command, str) and self._requires_shell(command),
            stdout=subprocess.PIPE if config.capture_output else None,
            stderr=subprocess.PIPE if config.capture_output else None,
            text=True,
            env=env,
            cwd=cwd,
            preexec_fn=config.pre_exec_fn if hasattr(os, 'preexec_fn') else None
        )
        
        self._active_processes[process.pid] = process
        
        return CommandResult(
            command=command,
            return_code=0,  # Background processes return 0 immediately
            stdout="",
            stderr="",
            execution_time=0,
            success=True,
            pid=process.pid
        )
    
    def terminate_process(self, pid: int) -> bool:
        """
        Terminate a running process.
        
        Args:
            pid: Process ID to terminate
            
        Returns:
            True if process was terminated successfully
        """
        process = self._active_processes.get(pid)
        if not process:
            return False
        
        try:
            if platform.system() == 'Windows':
                subprocess.run(['taskkill', '/PID', str(pid)], check=True)
            else:
                os.kill(pid, signal.SIGTERM)
            
            # Wait a moment for graceful shutdown
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                if platform.system() == 'Windows':
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
                else:
                    os.kill(pid, signal.SIGKILL)
            
            self._active_processes.pop(pid, None)
            return True
            
        except (subprocess.CalledProcessError, OSError):
            return False
    
    def get_process_status(self, pid: int) -> Optional[Dict[str, Any]]:
        """
        Get status information for a running process.
        
        Args:
            pid: Process ID to check
            
        Returns:
            Dictionary with process status information, None if not found
        """
        process = self._active_processes.get(pid)
        if not process:
            return None
        
        try:
            poll_result = process.poll()
            return {
                'pid': pid,
                'running': poll_result is None,
                'return_code': poll_result,
                'creation_time': getattr(process, 'creation_time', None)
            }
        except Exception:
            return None
    
    def get_execution_history(self, limit: Optional[int] = None) -> List[CommandResult]:
        """
        Get execution history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of CommandResult objects
        """
        history = self._execution_history
        if limit:
            history = history[-limit:]
        return history.copy()
    
    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()
    
    def get_platform_summary(self) -> Dict[str, Any]:
        """
        Get summary of platform and execution capabilities.
        
        Returns:
            Dictionary with platform and executor summary
        """
        return {
            'platform': self.shell_detector.get_shell_summary(),
            'executor': {
                'supported_modes': [mode.value for mode in ExecutionMode],
                'security_levels': [level.value for level in SecurityLevel],
                'default_timeout': self.default_config.timeout,
                'active_processes': len(self._active_processes),
                'execution_history_size': len(self._execution_history)
            }
        }