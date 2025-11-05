"""
Platform-Specific Command Handlers for mini-biai-1

This comprehensive handlers package provides specialized command execution
capabilities for different operating systems and command environments. Each
handler is optimized for its specific platform while maintaining a consistent
interface across all environments.

┌─────────────────────────────────────────────────────────────────────┐
│                    Platform-Specific Command Handlers               │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌───────────┐ ┌───────────┐ ┌───────────┐
    │  Unix     │ │  Windows  │ │  Remote   │
    │  Shell    │ │  Shell    │ │  Access   │
    │           │ │           │ │           │
    │ • Bash    │ │ • PowerShell│ │ • SSH    │
    │ • zsh     │ │ • cmd.exe │ │ • SFTP   │
    │ • sh      │ │ • WSL     │ │ • SCP    │
    └───────────┘ └───────────┘ └───────────┘

Core Components:

Unix/Linux/macOS Shell Handler:
    - Bash, zsh, sh shell execution
    - Unix-style path handling and environment variables
    - Signal handling and process management
    - Unix pipeline and redirection support
    - File permission and ownership management

Windows Command Handlers:
    - PowerShell command execution and script running
    - Windows Command Prompt (cmd.exe) support
    - Windows path normalization and escaping
    - Windows-specific environment variable handling
    - Registry access and Windows service management

Windows Subsystem for Linux (WSL):
    - Seamless Linux command execution within Windows
    - Cross-platform file system access
    - Linux/Windows path translation
    - WSL-specific command routing

Remote Command Execution:
    - SSH connection management and authentication
    - SFTP file transfer capabilities
    - Remote process execution and monitoring
    - Connection pooling and session management

Key Features:
- Cross-platform compatibility with optimized platform-specific handling
- Consistent API interface across all command environments
- Advanced error handling and recovery mechanisms
- Command execution with configurable timeouts and resource limits
- Output parsing and format normalization
- Environment variable and path handling per platform
- Security features including input sanitization and command validation
- Performance optimization with connection pooling and caching
- Comprehensive logging and debugging capabilities

Architecture Benefits:
- Platform-optimized command execution performance
- Consistent error handling and reporting across environments
- Advanced security features for remote command execution
- Extensible architecture for adding new command environments
- Comprehensive input validation and sanitization
- Automatic environment detection and optimization

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .base_handler import (
    BaseCommandHandler,
    CommandResult,
    CommandConfig,
    CommandExecutionError,
    SecurityError,
    TimeoutError,
    PlatformNotSupportedError
)

from .ssh_handler import (
    SSHConnectionError,
    SSHAuthenticationError,
    SSHTimeoutError
)

from .bash_handler import UnixShellHandler
from .powershell_handler import PowerShellHandler
from .cmd_handler import WindowsCommandHandler
from .wsl_handler import WSLHandler
from .ssh_handler import SSHHandler

# Import utility functions
from .handler_utils import (
    get_handler_for_platform,
    get_handler_by_type,
    list_available_handlers,
    validate_platform_support,
    get_platform_info as get_handler_platform_info,
    get_recommended_handler,
    get_handler_capabilities
)

__all__ = [
    # Base classes and exceptions
    'BaseCommandHandler',
    'CommandResult',
    'CommandConfig',
    'CommandExecutionError',
    'SecurityError',
    'TimeoutError',
    'PlatformNotSupportedError',
    
    # SSH-specific exceptions
    'SSHConnectionError',
    'SSHAuthenticationError',
    'SSHTimeoutError',
    
    # Handler classes
    'UnixShellHandler',
    'PowerShellHandler', 
    'WindowsCommandHandler',
    'WSLHandler',
    'SSHHandler',
    
    # Utility functions
    'get_handler_for_platform',
    'get_handler_by_type',
    'list_available_handlers',
    'validate_platform_support',
    'get_handler_platform_info',
    'get_recommended_handler',
    'get_handler_capabilities'
]

__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"