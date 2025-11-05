"""
Shell Detector Module

Provides cross-platform shell identification and detection capabilities
for the mini-biai-1 framework tool usage system.
"""

import os
import platform
import subprocess
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from pathlib import Path
from .platform_adapter import PlatformAdapter, get_platform_adapter


@dataclass
class ShellInfo:
    """Information about a detected shell."""
    name: str
    path: str
    version: Optional[str] = None
    is_available: bool = False
    capabilities: Optional[Set[str]] = None
    type: Optional[str] = None  # interactive, login, non-interactive, etc.


class ShellDetector:
    """
    Cross-platform shell identification and detection system.
    
    Provides comprehensive shell detection across different operating systems
    and shell types (bash, zsh, powershell, cmd, etc.).
    """
    
    def __init__(self, platform_adapter: Optional[PlatformAdapter] = None):
        """
        Initialize the shell detector.
        
        Args:
            platform_adapter: Platform adapter instance (uses default if None)
        """
        self.platform_adapter = platform_adapter or get_platform_adapter()
        self._shell_cache: Dict[str, ShellInfo] = {}
        self._available_shells: List[ShellInfo] = []
        self._current_shell: Optional[ShellInfo] = None
        
        # Common shell names and their executables
        self.known_shells = {
            'bash': {
                'posix': ['bash', 'sh'],
                'windows': ['bash.exe', 'wsl.exe', 'git-bash.exe'],
                'capabilities': {'interactive', 'login', 'scripting', 'aliases'}
            },
            'zsh': {
                'posix': ['zsh'],
                'windows': ['zsh.exe'],
                'capabilities': {'interactive', 'login', 'scripting', 'aliases', 'completion'}
            },
            'fish': {
                'posix': ['fish'],
                'windows': ['fish.exe'],
                'capabilities': {'interactive', 'login', 'scripting', 'completion', 'autosuggestions'}
            },
            'powershell': {
                'posix': ['pwsh', 'powershell'],
                'windows': ['powershell.exe', 'pwsh.exe'],
                'capabilities': {'interactive', 'scripting', 'object-oriented', 'modules'}
            },
            'cmd': {
                'posix': [],  # Not available on POSIX
                'windows': ['cmd.exe', 'cmd'],
                'capabilities': {'interactive', 'scripting', 'batch-files'}
            },
            'dash': {
                'posix': ['dash'],
                'windows': [],  # Not available on Windows
                'capabilities': {'scripting', 'posix-compliant'}
            },
            'ash': {
                'posix': ['ash', 'busybox'],
                'windows': [],  # Not typically available
                'capabilities': {'scripting', 'minimal'}
            }
        }
        
        self._initialize_shells()
    
    def _initialize_shells(self) -> None:
        """Initialize and detect available shells."""
        self._detect_available_shells()
        self._detect_current_shell()
        self._populate_shell_cache()
    
    def _detect_available_shells(self) -> None:
        """Detect all available shells on the system."""
        self._available_shells = []
        
        for shell_name, config in self.known_shells.items():
            if self.platform_adapter.platform_info.is_windows:
                shell_executables = config.get('windows', [])
            else:
                shell_executables = config.get('posix', [])
            
            for executable in shell_executables:
                shell_info = self._probe_shell(shell_name, executable)
                if shell_info and shell_info.is_available:
                    self._available_shells.append(shell_info)
                    break  # Found one, no need to check others
    
    def _probe_shell(self, shell_name: str, executable: str) -> Optional[ShellInfo]:
        """
        Probe a specific shell executable.
        
        Args:
            shell_name: Name of the shell
            executable: Executable command or path
            
        Returns:
            ShellInfo object if shell is available, None otherwise
        """
        shell_info = ShellInfo(
            name=shell_name,
            path=executable,
            capabilities=self.known_shells.get(shell_name, {}).get('capabilities', set())
        )
        
        try:
            # Check if the executable exists and is accessible
            result = subprocess.run(
                [executable, '--version'],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                shell_info.is_available = True
                shell_info.version = self._extract_version(result.stdout, result.stderr)
            else:
                # Try alternative version flags
                for version_flag in ['-v', '-V', '--version', '-version']:
                    try:
                        result = subprocess.run(
                            [executable, version_flag],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            shell_info.is_available = True
                            shell_info.version = self._extract_version(result.stdout, result.stderr)
                            break
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        continue
                
                # If version detection failed but executable exists, still mark as available
                if not shell_info.is_available and self._executable_exists(executable):
                    shell_info.is_available = True
                    shell_info.version = "Unknown"
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Try to check if executable exists in PATH
            if self._executable_exists(executable):
                shell_info.is_available = True
                shell_info.version = "Unknown"
        
        return shell_info if shell_info.is_available else None
    
    def _executable_exists(self, executable: str) -> bool:
        """Check if an executable exists and is accessible."""
        if self.platform_adapter.platform_info.is_windows:
            # Windows: check if executable exists
            return bool(self._which(executable))
        else:
            # POSIX: use 'which' or 'command -v'
            try:
                result = subprocess.run(
                    ['which', executable] if platform.system() != 'Darwin' else ['command', '-v', executable],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=3
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
    
    def _which(self, command: str) -> Optional[str]:
        """Windows equivalent of 'which' command."""
        try:
            result = subprocess.run(
                ['where', command],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                paths = result.stdout.strip().split('\n')
                return paths[0] if paths else None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Fallback: check common installation paths
        common_paths = [
            r'C:\Windows\System32',
            r'C:\Windows',
            r'C:\Program Files',
            r'C:\Program Files (x86)',
            os.path.join(os.environ.get('ProgramFiles', ''), ''),
        ]
        
        for path in common_paths:
            executable_path = os.path.join(path, f"{command}.exe")
            if os.path.exists(executable_path):
                return executable_path
        
        return None
    
    def _extract_version(self, stdout: str, stderr: str) -> str:
        """Extract version information from command output."""
        output = f"{stdout} {stderr}".strip()
        
        # Common version patterns
        version_patterns = [
            r'version\s+(\d+\.\d+\.\d+)',
            r'version\s+(\d+\.\d+)',
            r'(\d+\.\d+\.\d+)',
            r'(\d+\.\d+)',
            r'v?(\d+\.\d+\.\d+)',
            r'PowerShell\s+(v?\d+\.\d+\.\d+)',
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _detect_current_shell(self) -> None:
        """Detect the currently active shell."""
        # Method 1: Check SHELL environment variable (POSIX systems)
        if not self.platform_adapter.platform_info.is_windows:
            shell_path = os.environ.get('SHELL', '')
            if shell_path:
                shell_name = os.path.basename(shell_path)
                current_shell = self._get_shell_by_name(shell_name)
                if current_shell:
                    self._current_shell = current_shell
                    return
        
        # Method 2: Check parent process information (cross-platform)
        current_shell = self._detect_from_parent_process()
        if current_shell:
            self._current_shell = current_shell
            return
        
        # Method 3: Fall back to default shell
        default_shell = self.get_default_shell()
        if default_shell:
            self._current_shell = default_shell
    
    def _detect_from_parent_process(self) -> Optional[ShellInfo]:
        """Detect shell from parent process information."""
        try:
            if self.platform_adapter.platform_info.is_windows:
                # Windows: use wmic or tasklist
                return self._detect_windows_parent_shell()
            else:
                # POSIX: read /proc or use ps command
                return self._detect_posix_parent_shell()
        except Exception:
            # If detection fails, return None
            return None
    
    def _detect_posix_parent_shell(self) -> Optional[ShellInfo]:
        """Detect shell on POSIX systems via parent process."""
        try:
            # Try to read parent process name from /proc/self/stat (Linux)
            if os.path.exists('/proc/self/stat'):
                with open('/proc/self/stat', 'r') as f:
                    stat_data = f.read()
                    # Parse the stat data (fields are space-separated)
                    fields = stat_data.split(')')
                    if len(fields) >= 2:
                        process_info = fields[1].strip()
                        # Field 2 contains the parent process name
                        parts = process_info.split(' ')
                        if len(parts) >= 2:
                            parent_name = parts[1].strip('()')
                            return self._get_shell_by_name(parent_name)
            
            # Fallback: use ps command
            try:
                result = subprocess.run(
                    ['ps', '-p', str(os.getppid()), '-o', 'comm='],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=3
                )
                if result.returncode == 0:
                    parent_name = result.stdout.strip()
                    return self._get_shell_by_name(parent_name)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        except Exception:
            pass
        
        return None
    
    def _detect_windows_parent_shell(self) -> Optional[ShellInfo]:
        """Detect shell on Windows via parent process."""
        try:
            # Try using PowerShell to get parent process
            if self._which('powershell'):
                ps_script = '''
                $parent = Get-WmiObject -Class Win32_Process | Where-Object { $_.ProcessId -eq $PID }
                $grandparent = Get-WmiObject -Class Win32_Process | Where-Object { $_.ProcessId -eq $parent.ParentProcessId }
                if ($grandparent) { $grandparent.Name }
                '''
                
                result = subprocess.run(
                    ['powershell', '-Command', ps_script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    parent_name = result.stdout.strip()
                    # Remove .exe extension if present
                    if parent_name.endswith('.exe'):
                        parent_name = parent_name[:-4]
                    return self._get_shell_by_name(parent_name)
        except Exception:
            pass
        
        return None
    
    def _get_shell_by_name(self, name: str) -> Optional[ShellInfo]:
        """Get shell information by shell name."""
        # Clean the name
        clean_name = os.path.basename(name).replace('.exe', '')
        
        # Check available shells
        for shell in self._available_shells:
            if shell.name == clean_name:
                return shell
        
        return None
    
    def _populate_shell_cache(self) -> None:
        """Populate the shell cache with detected shells."""
        for shell in self._available_shells:
            self._shell_cache[shell.name] = shell
            # Also cache by path
            if shell.path:
                self._shell_cache[shell.path] = shell
    
    def get_available_shells(self) -> List[ShellInfo]:
        """
        Get list of all available shells.
        
        Returns:
            List of ShellInfo objects for available shells
        """
        return self._available_shells.copy()
    
    def get_current_shell(self) -> Optional[ShellInfo]:
        """
        Get information about the currently active shell.
        
        Returns:
            ShellInfo object for the current shell, None if undetectable
        """
        return self._current_shell
    
    def get_shell_by_name(self, name: str) -> Optional[ShellInfo]:
        """
        Get shell information by name.
        
        Args:
            name: Shell name (e.g., 'bash', 'powershell', 'zsh')
            
        Returns:
            ShellInfo object if found, None otherwise
        """
        return self._shell_cache.get(name)
    
    def get_default_shell(self) -> Optional[ShellInfo]:
        """
        Get the default shell for the platform.
        
        Returns:
            ShellInfo object for the default shell
        """
        if self.platform_adapter.platform_info.is_windows:
            # Windows: prefer PowerShell, then cmd
            preferred = ['powershell', 'cmd']
        else:
            # POSIX: prefer user's shell, then bash, then sh
            preferred = []
            if self._current_shell:
                preferred.append(self._current_shell.name)
            preferred.extend(['bash', 'sh'])
        
        for shell_name in preferred:
            shell = self._shell_cache.get(shell_name)
            if shell and shell.is_available:
                return shell
        
        # Ultimate fallback: first available shell
        return self._available_shells[0] if self._available_shells else None
    
    def is_shell_available(self, shell_name: str) -> bool:
        """
        Check if a specific shell is available.
        
        Args:
            shell_name: Name of the shell to check
            
        Returns:
            True if shell is available, False otherwise
        """
        shell = self._shell_cache.get(shell_name)
        return shell is not None and shell.is_available
    
    def get_shell_capabilities(self, shell_name: str) -> Set[str]:
        """
        Get capabilities of a specific shell.
        
        Args:
            shell_name: Name of the shell
            
        Returns:
            Set of capability strings
        """
        shell = self._shell_cache.get(shell_name)
        return shell.capabilities if shell else set()
    
    def detect_shell_from_command(self, command: str) -> Optional[ShellInfo]:
        """
        Detect which shell is best suited to run a specific command.
        
        Args:
            command: The command to be executed
            
        Returns:
            ShellInfo object for the best matching shell
        """
        # Simple heuristic: prefer shells that are available and have scripting capabilities
        for shell in self._available_shells:
            if shell.capabilities and 'scripting' in shell.capabilities:
                return shell
        
        # Fallback to default shell
        return self.get_default_shell()
    
    def get_shell_summary(self) -> Dict[str, Any]:
        """
        Get a summary of shell detection results.
        
        Returns:
            Dictionary containing shell detection summary
        """
        return {
            'platform': {
                'name': self.platform_adapter.platform_info.name,
                'version': self.platform_adapter.platform_info.version,
                'is_windows': self.platform_adapter.platform_info.is_windows,
                'is_posix': self.platform_adapter.platform_info.is_posix
            },
            'available_shells': [
                {
                    'name': shell.name,
                    'path': shell.path,
                    'version': shell.version,
                    'capabilities': list(shell.capabilities) if shell.capabilities else []
                }
                for shell in self._available_shells
            ],
            'current_shell': {
                'name': self._current_shell.name if self._current_shell else None,
                'path': self._current_shell.path if self._current_shell else None,
                'version': self._current_shell.version if self._current_shell else None
            } if self._current_shell else None,
            'default_shell': {
                'name': self.get_default_shell().name if self.get_default_shell() else None,
                'path': self.get_default_shell().path if self.get_default_shell() else None
            } if self.get_default_shell() else None,
            'total_available': len(self._available_shells)
        }