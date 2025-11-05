"""
Security and Safety Framework for Tool Usage

This package provides comprehensive security features for secure tool execution:
- Command validation and sanitization
- Sandboxed execution environments  
- Permission checking and escalation control
- Audit logging and monitoring
- Safety modes and circuit breakers

Features:
- Prevents malicious command execution
- Provides sandboxed execution for untrusted code
- Implements role-based access control
- Comprehensive audit trail
- Configurable safety modes
- Integration with main tool usage module
"""

from .validator import CommandValidator, ValidationResult
from .sandbox import SandboxManager, SandboxConfig, ExecutionResult
from .permissions import PermissionChecker, SecurityContext, PermissionLevel
from .audit import AuditLogger, AuditEvent, AuditLevel
from .safety_modes import SafetyManager, SafetyMode, CircuitBreaker

__version__ = "1.0.0"
__all__ = [
    # Core security components
    "CommandValidator",
    "ValidationResult", 
    "SandboxManager",
    "SandboxConfig",
    "ExecutionResult",
    "PermissionChecker", 
    "SecurityContext",
    "PermissionLevel",
    "AuditLogger",
    "AuditEvent",
    "AuditLevel",
    "SafetyManager",
    "SafetyMode",
    "CircuitBreaker",
]

# Security framework constants
DEFAULT_TIMEOUT = 30.0
MAX_INPUT_LENGTH = 10000
MAX_COMMAND_LENGTH = 2048
ALLOWED_FILE_EXTENSIONS = {'.py', '.txt', '.json', '.yaml', '.yml', '.md', '.csv', '.log'}
BLOCKED_PATTERNS = [
    'rm -rf', 'sudo', 'su', 'chmod', 'chown', 'passwd',
    'eval', 'exec', 'import os', 'subprocess', 'os.system',
    '__import__', 'compile', 'globals', 'locals', 'open(',
    'file(', 'input(', 'raw_input(', 'load(','unpickle',
    'shutil.rm', 'shutil.copy', 'subprocess.Popen', 'commands.'
]

# Security configuration defaults
SECURITY_CONFIG = {
    "strict_mode": False,
    "allow_network": False,
    "allow_file_write": False,
    "max_memory_mb": 256,
    "max_execution_time": 30.0,
    "audit_level": "MEDIUM",
    "sandbox_enabled": True,
    "permission_checking": True,
}

def get_security_config():
    """Get the security configuration."""
    return SECURITY_CONFIG.copy()

def set_security_config(config: dict):
    """Update the security configuration."""
    SECURITY_CONFIG.update(config)

# Initialize global security components
_command_validator = None
_safety_manager = None

def get_command_validator() -> CommandValidator:
    """Get the global command validator instance."""
    global _command_validator
    if _command_validator is None:
        _command_validator = CommandValidator()
    return _command_validator

def get_safety_manager() -> SafetyManager:
    """Get the global safety manager instance."""
    global _safety_manager
    if _safety_manager is None:
        _safety_manager = SafetyManager()
    return _safety_manager

def is_secure_mode() -> bool:
    """Check if security is in secure mode."""
    return SECURITY_CONFIG.get("strict_mode", False)

def enable_strict_mode():
    """Enable strict security mode."""
    SECURITY_CONFIG["strict_mode"] = True

def disable_strict_mode():
    """Disable strict security mode."""
    SECURITY_CONFIG["strict_mode"] = False