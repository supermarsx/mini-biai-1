"""
SSH Remote Command Execution Handler for mini-biai-1

This module provides specialized command execution capabilities for remote
systems via SSH (Secure Shell). It enables secure remote command execution,
file transfer operations, and remote system management with comprehensive
authentication and connection management.

The SSH handler implements:
- SSH connection management and authentication
- Remote command execution with session management
- SFTP file transfer capabilities
- SSH key-based and password authentication
- Connection pooling and session management
- Remote process execution and monitoring
- Tunneling and port forwarding capabilities

┌─────────────────────────────────────────────────────────────────────┐
│                    SSH Remote Handler Architecture                  │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │   SSH       │ │   Remote    │ │    SFTP     │
    │ Connection  │ │  Command    │ │   File      │
    │             │ │ Execution   │ │  Transfer   │
    │ • Auth      │ │ • Execute   │ │ • Upload    │
    │ • Sessions  │ │ • Monitor   │ │ • Download  │
    │ • Pooling   │ │ • Timeout   │ │ • Sync      │
    └─────────────┘ └─────────────┘ └─────────────┘

Key Features:
- SSH connection management with authentication
- Remote command execution with session tracking
- SFTP file transfer and synchronization
- SSH key-based and password authentication
- Connection pooling and session management
- Remote process execution and monitoring
- SSH tunneling and port forwarding
- Advanced error handling and recovery

Architecture Benefits:
- Secure remote command execution capabilities
- Comprehensive SSH protocol implementation
- Advanced authentication and security features
- Connection pooling and performance optimization
- Cross-platform remote system management

Dependencies:
    Core: subprocess, os, sys (process management)
    SSH: paramiko (SSH library), socket (network operations)
    Security: cryptography (SSH key handling), hashlib (integrity)
    Utils: pathlib (path handling), typing (type annotations)

Error Handling:
    The SSH handler implements comprehensive error handling:
        - SSH connection failure recovery
        - Authentication error handling and retry
        - Remote command execution timeout management
        - Network connectivity and recovery
        - SSH key validation and security checks
        - Graceful fallback and error propagation

Usage Examples:

Basic SSH Connection:
    >>> from src.tool_usage.handlers.ssh_handler import SSHHandler
    >>> from src.tool_usage.handlers.base_handler import CommandConfig
    >>> 
    >>> handler = SSHHandler(
    ...     hostname="server.example.com",
    ...     username="user",
    ...     password="password"
    >>> )
    >>> 
    >>> result = handler.execute_command("ls -la /home/user")
    >>> print(f"Remote directory listing:\n{result.stdout}")

SSH Key Authentication:
    >>> # Use SSH key for authentication
    >>> handler = SSHHandler(
    ...     hostname="server.example.com",
    ...     username="user",
    ...     key_file="/home/user/.ssh/id_rsa"
    >>> )
    >>> 
    >>> result = handler.execute_command("uptime")
    >>> print(f"Server uptime: {result.stdout}")

SFTP File Transfer:
    >>> # Upload file to remote server
    >>> result = handler.upload_file("/local/file.txt", "/remote/file.txt")
    >>> print(f"Upload result: {result.success}")
    >>> 
    >>> # Download file from remote server
    >>> result = handler.download_file("/remote/backup.tar.gz", "/local/backup.tar.gz")
    >>> print(f"Download result: {result.success}")

Remote Process Management:
    >>> # Execute remote process with monitoring
    >>> result = handler.execute_command(
    ...     "long_running_script.sh",
    ...     timeout=300  # 5 minute timeout
    >>> )
    >>> print(f"Process result: {result.exit_code}")
    >>> 
    >>> # Monitor remote processes
    >>> result = handler.execute_command("ps aux | grep python")
    >>> print(f"Remote processes:\n{result.stdout}")

Connection Pooling:
    >>> # Create connection pool for multiple servers
    >>> servers = [
    ...     {"hostname": "server1.example.com", "username": "user"},
    ...     {"hostname": "server2.example.com", "username": "user"},
    ... ]
    >>> 
    >>> pool = SSHConnectionPool(servers)
    >>> result1 = pool.execute_on_server("server1.example.com", "df -h")
    >>> result2 = pool.execute_on_server("server2.example.com", "free -m")

SSH Tunneling:
    >>> # Create SSH tunnel for port forwarding
    >>> tunnel = handler.create_tunnel(
    ...     local_port=8080,
    ...     remote_host="localhost",
    ...     remote_port=80
    >>> )
    >>> 
    >>> # Use tunnel for web requests
    >>> result = handler.execute_command("curl http://localhost:8080")
    >>> print(f"Web content via tunnel: {result.stdout}")

Batch Remote Operations:
    >>> # Execute commands on multiple servers
    >>> commands = [
    ...     "git pull origin main",
    ...     "sudo systemctl restart apache2",
    ...     "tail -f /var/log/apache2/access.log"
    >>> ]
    >>> 
    >>> for cmd in commands:
    ...     result = handler.execute_command(cmd)
    ...     print(f"Command: {cmd}")
    ...     print(f"Result: {result.success}")

Remote System Monitoring:
    >>> # Monitor remote system resources
    >>> result = handler.execute_command(\"\"\"
    ... echo \"=== System Info ===\"
    ... uname -a
    ... echo \"=== CPU Info ===\"
    ... lscpu | head -10
    ... echo \"=== Memory Info ===\"
    ... free -h
    ... echo \"=== Disk Info ===\"
    ... df -h
    ... echo \"=== Network Info ===\"
    ... ip addr show
    ... \"\"\")
    >>> print(f"System monitoring:\n{result.stdout}")

SSH Configuration Management:
    >>> # Apply configuration to remote server
    >>> config_script = \"\"\"
    ... # Backup existing config
    ... sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
    ... 
    ... # Apply new configuration
    ... sudo tee /etc/nginx/nginx.conf > /dev/null <<EOF
    ... worker_processes auto;
    ... events {
    ...     worker_connections 1024;
    ... }
    ... http {
    ...     include /etc/nginx/mime.types;
    ...     default_type application/octet-stream;
    ... }
    ... EOF
    ... 
    ... # Test configuration
    ... sudo nginx -t
    ... 
    ... # Reload nginx if config is valid
    ... if [ $? -eq 0 ]; then
    ...     sudo systemctl reload nginx
    ...     echo \"Configuration applied successfully\"
    ... else
    ...     echo \"Configuration error - reverted\"
    ...     sudo mv /etc/nginx/nginx.conf.backup /etc/nginx/nginx.conf
    ... fi
    ... \"\"\"
    # result = handler.execute_command(config_script)
    # print(f"Config update result:\n{result.stdout}")

# SSH Handler Implementation
# Version 1.0.0
# Author: mini-biai-1 Team
# License: MIT

import subprocess
import os
import sys
import time
import socket
import pathlib
import tempfile
import hashlib
import re
from typing import Dict, List, Optional, Union, Any, Tuple

from .base_handler import BaseCommandHandler, CommandResult, CommandConfig
from typing import Dict, List, Optional, Union, Any, Tuple

from .base_handler import BaseCommandHandler, CommandResult, CommandConfig


class SSHConnectionError(Exception):
    """Exception raised when SSH connection fails."""
    pass


class SSHAuthenticationError(Exception):
    """Exception raised when SSH authentication fails."""
    pass


class SSHTimeoutError(Exception):
    """Exception raised when SSH operation times out."""
    pass


class SSHHandler(BaseCommandHandler):
    """
    Specialized command handler for SSH remote command execution.
    
    This handler provides secure remote command execution capabilities via SSH,
    supporting various authentication methods, connection pooling, SFTP file
    transfer, and advanced remote system management features.
    
    The handler implements:
    - SSH connection management with multiple authentication methods
    - Remote command execution with session tracking
    - SFTP file transfer and synchronization capabilities
    - Connection pooling for multiple servers
    - SSH tunneling and port forwarding
    - Remote process execution and monitoring
    
    Attributes:
        hostname (str): SSH server hostname or IP address
        username (str): SSH authentication username
        port (int): SSH server port (default: 22)
        password (str): SSH authentication password (if using password auth)
        key_file (str): Path to SSH private key file (if using key auth)
        connection_timeout (float): SSH connection timeout in seconds
        max_connections (int): Maximum number of concurrent connections
    
    Example Usage:
        >>> handler = SSHHandler(
        ...     hostname="server.example.com",
        ...     username="user",
        ...     password="password"
        >>> )
        >>> 
        >>> # Execute remote command
        >>> result = handler.execute_command("ls -la /home/user")
        >>> print(f"Remote output: {result.stdout}")
        
        >>> # Upload file via SFTP
        >>> result = handler.upload_file("/local/file.txt", "/remote/file.txt")
        >>> print(f"Upload successful: {result.success}")
    """
    
    def __init__(self, hostname: str, username: str, port: int = 22,
                 password: str = None, key_file: str = None,
                 config: Optional[CommandConfig] = None):
        """
        Initialize the SSH handler.
        
        Args:
            hostname: SSH server hostname or IP address
            username: SSH authentication username
            port: SSH server port (default: 22)
            password: SSH authentication password (optional if using key auth)
            key_file: Path to SSH private key file (optional if using password auth)
            config: Optional configuration for command execution
        """
        super().__init__(config)
        self.hostname = hostname
        self.username = username
        self.port = port
        self.password = password
        self.key_file = key_file
        self.connection_timeout = 30.0
        self.max_connections = 10
        
        # Connection pool management
        self._connection_pool = {}
        self._active_connections = 0
        
        # SSH command template
        self.ssh_base_command = self._build_ssh_command()
        
        if self.config.enable_logging:
            self.logger.info(f"SSH Handler initialized:")
            self.logger.info(f"  Hostname: {self.hostname}")
            self.logger.info(f"  Username: {self.username}")
            self.logger.info(f"  Port: {self.port}")
            self.logger.info(f"  Authentication: {'Key' if self.key_file else 'Password' if self.password else 'None'}")
    
    def _build_ssh_command(self) -> str:
        """
        Build the base SSH command with authentication options.
        
        Returns:
            str: Base SSH command with common options
        """
        cmd_parts = ['ssh']
        
        # Port specification
        cmd_parts.extend(['-p', str(self.port)])
        
        # Connection timeout
        cmd_parts.extend(['-o', f'ConnectTimeout={self.connection_timeout}'])
        
        # Authentication options
        if self.key_file:
            cmd_parts.extend(['-i', self.key_file])
        
        # Security options
        cmd_parts.extend([
            '-o', 'StrictHostKeyChecking=no',
            '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'BatchMode=yes',  # Non-interactive mode
            '-o', 'PasswordAuthentication=yes' if self.password else 'PasswordAuthentication=no'
        ])
        
        # User and host specification
        cmd_parts.append(f'{self.username}@{self.hostname}')
        
        return ' '.join(cmd_parts)
    
    def _is_platform_supported(self) -> bool:
        """
        Check if the current platform is supported by this handler.
        
        SSH is supported on all platforms with SSH client available.
        
        Returns:
            bool: True if SSH client is available
        """
        try:
            import shutil
            return shutil.which('ssh') is not None
        except Exception:
            return False
    
    def execute_command(self, command: str, config: Optional[CommandConfig] = None) -> CommandResult:
        """
        Execute a command on the remote SSH server.
        
        This method provides SSH-specific command execution with support for:
        - Remote command execution via SSH
        - SSH authentication and connection management
        - Remote system integration and utilities
        - SSH session management and pooling
        
        Args:
            command: The command string to execute remotely
            config: Optional override configuration for this execution
            
        Returns:
            CommandResult: Standardized result with SSH-specific enhancements
            
        Raises:
            CommandExecutionError: If command execution fails
            SecurityError: If command fails security validation
            TimeoutError: If command execution exceeds timeout
            SSHConnectionError: If SSH connection fails
            SSHAuthenticationError: If SSH authentication fails
        """
        if not self.is_platform_supported():
            raise PlatformNotSupportedError(f"Platform {sys.platform} not supported by SSH handler")
        
        # Use provided config or default
        exec_config = config or self.config
        
        # Validate command for SSH-specific security
        self._validate_ssh_command(command)
        
        if self.config.enable_logging:
            self.logger.debug(f"Executing SSH command on {self.hostname}: {command}")
            self.logger.debug(f"SSH command: {self.ssh_base_command}")
        
        # Prepare command for SSH execution
        ssh_command = self._prepare_ssh_command(command, exec_config)
        
        try:
            # Test SSH connection first