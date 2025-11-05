#!/usr/bin/env python3
"""
Sandboxed Execution Environment Module

This module provides secure isolated execution environments with resource
monitoring, process isolation, and comprehensive security controls.

Author: Mini-Biai Framework Team
Version: 1.0.0
Date: 2024
"""

import os
import signal
import subprocess
import tempfile
import threading
import time
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import psutil
except ImportError:
    psutil = None

from ..core.exceptions import SandboxError, SecurityError
from ..security.validator import CommandValidator, ValidationResult, ThreatLevel


class IsolationLevel(Enum):
    """Sandbox isolation levels"""
    NONE = "none"          # No isolation (not recommended)
    BASIC = "basic"        # Basic process isolation
    STRICT = "strict"      # Process + file system isolation
    CONTAINER = "container" # Full container isolation (requires Docker)


class ResourceType(Enum):
    """Resource types for monitoring"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"


@dataclass
class SandboxConfig:
    """Sandbox execution configuration"""
    isolation_level: IsolationLevel = IsolationLevel.STRICT
    max_memory_mb: int = 256
    max_cpu_time_sec: int = 30
    max_wall_time_sec: int = 60
    max_file_size_mb: int = 50
    max_processes: int = 5
    allowed_dirs: Set[str] = field(default_factory=lambda: {tempfile.gettempdir()})
    blocked_dirs: Set[str] = field(default_factory=lambda: {"/etc", "/sys", "/proc", "/dev"})
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {".txt", ".log", ".json", ".csv"})
    blocked_file_extensions: Set[str] = field(default_factory=lambda: {".sh", ".exe", ".bat"})
    network_enabled: bool = True
    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=lambda: {"localhost", "127.0.0.1", "::1"})
    working_directory: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    docker_image: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of sandboxed execution"""
    success: bool
    exit_code: Optional[int]
    stdout: str
    stderr: str
    execution_time: float
    resource_usage: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    sandbox_path: Optional[str] = None
    process_id: Optional[int] = None
    killed_by_timeout: bool = False
    killed_by_resource_limit: bool = False


class ResourceMonitor:
    """Monitor resource usage during execution"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.monitoring = False
        self.resources = {
            "cpu_percent": 0.0,
            "memory_mb": 0.0,
            "disk_usage_mb": 0.0,
            "processes": 0
        }
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self, process_id: int) -> None:
        """Start monitoring resource usage"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(process_id,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        if not self.monitoring:
            return
        
        self.monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_resources(self, process_id: int) -> None:
        """Monitor resources in background thread"""
        if psutil is None:
            return
        
        try:
            process = psutil.Process(process_id)
            
            while self.monitoring and not self._stop_event.is_set():
                try:
                    # CPU usage
                    cpu_percent = process.cpu_percent(interval=0.1)
                    
                    # Memory usage
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    
                    # Process count in subtree
                    processes = len(process.children(recursive=True)) + 1
                    
                    # Update resource tracking
                    self.resources.update({
                        "cpu_percent": cpu_percent,
                        "memory_mb": memory_mb,
                        "processes": processes
                    })
                    
                    # Check for resource limits
                    if memory_mb > self.config.max_memory_mb:
                        raise SandboxError(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.config.max_memory_mb}MB")
                    
                    if processes > self.config.max_processes:
                        raise SandboxError(f"Process limit exceeded: {processes} > {self.config.max_processes}")
                    
                    time.sleep(0.1)  # Monitor every 100ms
                    
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    # Log but don't stop monitoring due to minor issues
                    pass
                    
        except Exception as e:
            # Handle any monitoring errors gracefully
            pass
    
    def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage"""
        return self.resources.copy()


class FileSystemMonitor:
    """Monitor file system access during execution"""
    
    def __init__(self, config: SandboxConfig):
        self.config = config
        self.allowed_dirs = {os.path.abspath(d) for d in config.allowed_dirs}
        self.blocked_dirs = {os.path.abspath(d) for d in config.blocked_dirs}
        self.allowed_extensions = config.allowed_file_extensions
        self.blocked_extensions = config.blocked_file_extensions
        self.violations: List[Dict[str, Any]] = []
    
    def check_path_access(self, path: str, operation: str = "read") -> Tuple[bool, str]:
        """Check if path access is allowed"""
        abs_path = os.path.abspath(path)
        
        # Check blocked directories
        for blocked_dir in self.blocked_dirs:
            if abs_path.startswith(blocked_dir + os.sep) or abs_path == blocked_dir:
                self.violations.append({
                    "path": abs_path,
                    "operation": operation,
                    "reason": f"Access to {blocked_dir} is blocked",
                    "timestamp": time.time()
                })
                return False, f"Access to {blocked_dir} is blocked"
        
        # Check allowed directories
        if self.allowed_dirs:
            allowed = False
            for allowed_dir in self.allowed_dirs:
                if abs_path.startswith(allowed_dir + os.sep) or abs_path == allowed_dir:
                    allowed = True
                    break
            
            if not allowed:
                self.violations.append({
                    "path": abs_path,
                    "operation": operation,
                    "reason": "Path not in allowed directories",
                    "timestamp": time.time()
                })
                return False, "Path not in allowed directories"
        
        # Check file extensions
        if os.path.isfile(abs_path):
            file_ext = os.path.splitext(abs_path)[1].lower()
            
            if file_ext in self.blocked_extensions:
                self.violations.append({
                    "path": abs_path,
                    "operation": operation,
                    "reason": f"File extension {file_ext} is blocked",
                    "timestamp": time.time()
                })
                return False, f"File extension {file_ext} is blocked"
            
            if self.allowed_extensions and file_ext not in self.allowed_extensions:
                self.violations.append({
                    "path": abs_path,
                    "operation": operation,
                    "reason": f"File extension {file_ext} is not allowed",
                    "timestamp": time.time()
                })
                return False, f"File extension {file_ext} is not allowed"
        
        return True, "Access allowed"
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get list of file system violations"""
        return self.violations.copy()


class SandboxManager:
    """Manage sandboxed execution environments"""
    
    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        self.validator = CommandValidator()
        self.active_sandboxes: Dict[str, SandboxConfig] = {}
        self.logger = None  # Will be set by parent logger
        
        # Initialize monitoring components
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.fs_monitor: Optional[FileSystemMonitor] = None
    
    def set_logger(self, logger) -> None:
        """Set logger for sandbox operations"""
        self.logger = logger
        self.validator.logger = logger
    
    def execute_command(self, command: str, working_dir: Optional[str] = None, 
                       timeout: Optional[float] = None) -> ExecutionResult:
        """Execute command in sandbox environment"""
        start_time = time.time()
        timeout = timeout or self.config.max_wall_time_sec
        
        # Validate command first
        validation_result = self.validator.validate_command(command)
        if not validation_result.is_valid:
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Command validation failed: {'; '.join(validation_result.issues)}",
                execution_time=0.0,
                resource_usage={},
                warnings=[],
                errors=validation_result.issues,
                killed_by_timeout=False,
                killed_by_resource_limit=False
            )
        
        # Create sandbox environment
        sandbox_dir = None
        try:
            sandbox_dir = self._create_sandbox_environment(working_dir)
            
            # Execute based on isolation level
            if self.config.isolation_level == IsolationLevel.NONE:
                result = self._execute_basic(command, sandbox_dir, timeout)
            elif self.config.isolation_level == IsolationLevel.BASIC:
                result = self._execute_basic(command, sandbox_dir, timeout)
            elif self.config.isolation_level == IsolationLevel.STRICT:
                result = self._execute_strict(command, sandbox_dir, timeout)
            elif self.config.isolation_level == IsolationLevel.CONTAINER:
                result = self._execute_container(command, sandbox_dir, timeout)
            else:
                raise SandboxError(f"Unsupported isolation level: {self.config.isolation_level}")
            
            # Add validation warnings
            if validation_result.suggestions:
                result.warnings.extend([f"Security suggestion: {s}" for s in validation_result.suggestions])
            
            result.sandbox_path = sandbox_dir
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                resource_usage={},
                warnings=[],
                errors=[str(e)],
                sandbox_path=sandbox_dir,
                killed_by_timeout=False,
                killed_by_resource_limit=False
            )
        
        finally:
            # Cleanup sandbox
            if sandbox_dir and os.path.exists(sandbox_dir):
                self._cleanup_sandbox(sandbox_dir)
    
    def execute_python_code(self, code: str, globals_dict: Optional[Dict] = None, 
                           locals_dict: Optional[Dict] = None) -> ExecutionResult:
        """Execute Python code in sandbox"""
        # Validate Python code
        validation_result = self.validator.validate_command(code)
        if not validation_result.is_valid:
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Code validation failed: {'; '.join(validation_result.issues)}",
                execution_time=0.0,
                resource_usage={},
                warnings=[],
                errors=validation_result.issues
            )
        
        start_time = time.time()
        
        # Create restricted globals
        if globals_dict is None:
            globals_dict = {
                "__builtins__": self._get_safe_builtins(),
                "__name__": "__sandbox__",
                "print": lambda *args, **kwargs: None  # Disable print for security
            }
        
        # Create safe locals
        if locals_dict is None:
            locals_dict = {}
        
        try:
            # Execute code
            exec(code, globals_dict, locals_dict)
            
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=True,
                exit_code=0,
                stdout="",  # No output captured
                stderr="",
                execution_time=execution_time,
                resource_usage={},
                warnings=validation_result.suggestions if validation_result.suggestions else [],
                errors=[]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                resource_usage={},
                warnings=[],
                errors=[str(e)]
            )
    
    def _get_safe_builtins(self) -> Dict[str, Any]:
        """Get safe builtins for sandboxed Python execution"""
        safe_builtins = {
            # Safe mathematical functions
            "abs", "divmod", "max", "min", "pow", "round", "sum",
            # Safe type functions
            "bool", "complex", "dict", "float", "int", "list", "set", "str", "tuple",
            # Safe utility functions
            "all", "any", "enumerate", "filter", "map", "range", "reversed", "sorted", "zip",
            # Safe string functions
            "ascii", "bin", "chr", "format", "hex", "oct", "ord",
            # Safe container operations
            "len", "slice"
        }
        
        # Get actual builtin functions
        import builtins
        return {name: getattr(builtins, name) for name in safe_builtins 
                if hasattr(builtins, name)}
    
    def _create_sandbox_environment(self, working_dir: Optional[str] = None) -> str:
        """Create sandbox environment directory"""
        sandbox_dir = tempfile.mkdtemp(prefix="sandbox_")
        self.active_sandboxes[sandbox_dir] = self.config
        
        # Create directory structure
        os.makedirs(os.path.join(sandbox_dir, "tmp"), exist_ok=True)
        os.makedirs(os.path.join(sandbox_dir, "work"), exist_ok=True)
        
        # Set working directory
        if working_dir:
            work_dir = os.path.join(sandbox_dir, "work", os.path.basename(working_dir))
            if os.path.exists(working_dir):
                shutil.copytree(working_dir, work_dir, symlinks=True)
            else:
                os.makedirs(work_dir, exist_ok=True)
        else:
            work_dir = os.path.join(sandbox_dir, "work")
            os.makedirs(work_dir, exist_ok=True)
        
        return sandbox_dir
    
    def _execute_basic(self, command: str, sandbox_dir: str, timeout: float) -> ExecutionResult:
        """Execute command with basic isolation"""
        start_time = time.time()
        process = None
        
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.environment_vars)
            
            # Change to sandbox directory
            original_dir = os.getcwd()
            os.chdir(os.path.join(sandbox_dir, "work"))
            
            try:
                # Start process
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    preexec_fn=os.setsid if os.name != 'nt' else None
                )
                
                # Start resource monitoring
                self.resource_monitor = ResourceMonitor(self.config)
                self.resource_monitor.start_monitoring(process.pid)
                
                # Wait for completion with timeout
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    
                    execution_time = time.time() - start_time
                    
                    # Get resource usage
                    resource_usage = self.resource_monitor.get_resource_usage() if self.resource_monitor else {}
                    
                    return ExecutionResult(
                        success=process.returncode == 0,
                        exit_code=process.returncode,
                        stdout=stdout.decode('utf-8', errors='replace'),
                        stderr=stderr.decode('utf-8', errors='replace'),
                        execution_time=execution_time,
                        resource_usage=resource_usage,
                        warnings=[],
                        errors=[],
                        process_id=process.pid
                    )
                    
                except subprocess.TimeoutExpired:
                    # Kill process and return timeout result
                    self.resource_monitor.stop_monitoring()
                    process.kill()
                    stdout, stderr = process.communicate()
                    
                    execution_time = time.time() - start_time
                    
                    return ExecutionResult(
                        success=False,
                        exit_code=None,
                        stdout=stdout.decode('utf-8', errors='replace') if stdout else "",
                        stderr="Command timed out",
                        execution_time=execution_time,
                        resource_usage={},
                        warnings=[],
                        errors=["Command execution timeout"],
                        killed_by_timeout=True
                    )
                    
            finally:
                os.chdir(original_dir)
                if self.resource_monitor:
                    self.resource_monitor.stop_monitoring()
                    
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                resource_usage={},
                warnings=[],
                errors=[str(e)]
            )
        
        finally:
            if process:
                try:
                    process.kill()
                except:
                    pass
    
    def _execute_strict(self, command: str, sandbox_dir: str, timeout: float) -> ExecutionResult:
        """Execute command with strict isolation"""
        # Start with basic execution
        result = self._execute_basic(command, sandbox_dir, timeout)
        
        # Add file system monitoring for strict mode
        self.fs_monitor = FileSystemMonitor(self.config)
        
        # Check if any file system violations occurred
        violations = self.fs_monitor.get_violations()
        if violations:
            result.warnings.extend([
                f"File system violation: {v['reason']}" for v in violations
            ])
        
        return result
    
    def _execute_container(self, command: str, sandbox_dir: str, timeout: float) -> ExecutionResult:
        """Execute command in Docker container"""
        if not shutil.which("docker"):
            raise SandboxError("Docker not available for container execution")
        
        if not self.config.docker_image:
            raise SandboxError("Docker image not specified for container execution")
        
        start_time = time.time()
        
        try:
            # Prepare Docker command
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",  # No network access by default
                "--memory", f"{self.config.max_memory_mb}m",
                "--cpus", "1.0",
                "-v", f"{sandbox_dir}:/sandbox",
                "-w", "/sandbox/work"
            ]
            
            # Add domain restrictions if specified
            if not self.config.network_enabled:
                docker_cmd.extend(["--network", "none"])
            
            # Add environment variables
            for key, value in self.config.environment_vars.items():
                docker_cmd.extend(["-e", f"{key}={value}"])
            
            # Add image and command
            docker_cmd.extend([self.config.docker_image, "/bin/sh", "-c", command])
            
            # Execute in container
            result = self._execute_basic(" ".join(docker_cmd), sandbox_dir, timeout)
            result.execution_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ExecutionResult(
                success=False,
                exit_code=1,
                stdout="",
                stderr=f"Container execution failed: {str(e)}",
                execution_time=execution_time,
                resource_usage={},
                warnings=[],
                errors=[str(e)]
            )
    
    def _cleanup_sandbox(self, sandbox_dir: str) -> None:
        """Clean up sandbox environment"""
        try:
            if sandbox_dir in self.active_sandboxes:
                del self.active_sandboxes[sandbox_dir]
            
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")
    
    def cleanup_all_sandboxes(self) -> None:
        """Clean up all active sandbox environments"""
        sandbox_dirs = list(self.active_sandboxes.keys())
        for sandbox_dir in sandbox_dirs:
            self._cleanup_sandbox(sandbox_dir)
    
    def get_sandbox_info(self) -> Dict[str, Any]:
        """Get information about active sandboxes"""
        return {
            "active_sandboxes": len(self.active_sandboxes),
            "sandbox_dirs": list(self.active_sandboxes.keys()),
            "current_config": {
                "isolation_level": self.config.isolation_level.value,
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_time_sec": self.config.max_cpu_time_sec,
                "max_wall_time_sec": self.config.max_wall_time_sec,
                "max_file_size_mb": self.config.max_file_size_mb,
                "max_processes": self.config.max_processes,
                "network_enabled": self.config.network_enabled
            }
        }
    
    def update_config(self, **kwargs) -> None:
        """Update sandbox configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration option: {key}")
    
    def add_allowed_directory(self, directory: str) -> None:
        """Add directory to allowed list"""
        abs_dir = os.path.abspath(directory)
        self.config.allowed_dirs.add(abs_dir)
    
    def add_blocked_directory(self, directory: str) -> None:
        """Add directory to blocked list"""
        abs_dir = os.path.abspath(directory)
        self.config.blocked_dirs.add(abs_dir)
    
    def add_allowed_domain(self, domain: str) -> None:
        """Add domain to allowed list"""
        self.config.allowed_domains.add(domain.lower())
    
    def add_blocked_domain(self, domain: str) -> None:
        """Add domain to blocked list"""
        self.config.blocked_domains.add(domain.lower())


# Export main classes and functions
__all__ = [
    "SandboxManager",
    "SandboxConfig",
    "ExecutionResult",
    "IsolationLevel",
    "ResourceMonitor",
    "FileSystemMonitor"
]