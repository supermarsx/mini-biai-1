"""
Tool Registry Module

Provides comprehensive tool management and discovery capabilities
for the mini-biai-1 framework tool usage system.
"""

import os
import sys
import json
import yaml
import subprocess
import platform
import importlib.util
import inspect
from typing import Dict, List, Optional, Union, Any, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone

from .platform_adapter import PlatformAdapter, get_platform_adapter
from .shell_detector import ShellDetector


class ToolCategory(Enum):
    """Categories of tools that can be managed."""
    SYSTEM = "system"
    DEVELOPMENT = "development"
    NETWORK = "network"
    FILE_MANAGEMENT = "file_management"
    TEXT_PROCESSING = "text_processing"
    DATA_PROCESSING = "data_processing"
    MULTIMEDIA = "multimedia"
    SECURITY = "security"
    AUTOMATION = "automation"
    UTILITIES = "utilities"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """Status of a tool in the registry."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    OUTDATED = "outdated"
    CONFLICTING = "conflicting"
    INSTALLED = "installed"
    MISSING = "missing"


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    category: Optional[ToolCategory] = None
    status: ToolStatus = ToolStatus.UNAVAILABLE
    path: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    usage_examples: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None
    license: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_verified: Optional[datetime] = None
    verification_errors: List[str] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)


class ToolDiscovery:
    """Discovers available tools on the system."""
    
    def __init__(self, platform_adapter: PlatformAdapter):
        """
        Initialize the tool discovery system.
        
        Args:
            platform_adapter: Platform adapter instance
        """
        self.platform_adapter = platform_adapter
        self._builtin_tools = self._load_builtin_tools()
        self._system_paths = self._get_system_paths()
    
    def _load_builtin_tools(self) -> Dict[str, Dict[str, Any]]:
        """Load built-in tool definitions."""
        return {
            # System tools
            'ls': {
                'category': ToolCategory.SYSTEM,
                'description': 'List directory contents',
                'platforms': ['linux', 'macos'],
                'capabilities': ['list_files', 'detailed_view']
            },
            'dir': {
                'category': ToolCategory.SYSTEM,
                'description': 'List directory contents (Windows)',
                'platforms': ['windows'],
                'capabilities': ['list_files', 'detailed_view']
            },
            'pwd': {
                'category': ToolCategory.SYSTEM,
                'description': 'Print working directory',
                'platforms': ['linux', 'macos'],
                'capabilities': ['current_directory']
            },
            'cd': {
                'category': ToolCategory.SYSTEM,
                'description': 'Change directory',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['change_directory']
            },
            'mkdir': {
                'category': ToolCategory.FILE_MANAGEMENT,
                'description': 'Create directories',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['create_directory', 'recursive_create']
            },
            'rm': {
                'category': ToolCategory.FILE_MANAGEMENT,
                'description': 'Remove files or directories',
                'platforms': ['linux', 'macos'],
                'capabilities': ['remove_files', 'remove_directories', 'recursive_remove']
            },
            'del': {
                'category': ToolCategory.FILE_MANAGEMENT,
                'description': 'Delete files (Windows)',
                'platforms': ['windows'],
                'capabilities': ['remove_files']
            },
            
            # Development tools
            'git': {
                'category': ToolCategory.DEVELOPMENT,
                'description': 'Distributed version control system',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['version_control', 'branching', 'merging', 'history']
            },
            'python': {
                'category': ToolCategory.DEVELOPMENT,
                'description': 'Python interpreter',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['script_execution', 'package_management', 'development']
            },
            'pip': {
                'category': ToolCategory.DEVELOPMENT,
                'description': 'Python package installer',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['package_management', 'installation']
            },
            'node': {
                'category': ToolCategory.DEVELOPMENT,
                'description': 'Node.js runtime',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['javascript_execution', 'server_development']
            },
            'npm': {
                'category': ToolCategory.DEVELOPMENT,
                'description': 'Node package manager',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['package_management', 'javascript_dependencies']
            },
            
            # Network tools
            'curl': {
                'category': ToolCategory.NETWORK,
                'description': 'Command line tool for transferring data with URLs',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['http_requests', 'file_download', 'api_calls']
            },
            'wget': {
                'category': ToolCategory.NETWORK,
                'description': 'Network downloader',
                'platforms': ['linux', 'macos'],
                'capabilities': ['file_download', 'recursive_download']
            },
            'ping': {
                'category': ToolCategory.NETWORK,
                'description': 'Test network connectivity',
                'platforms': ['linux', 'macos', 'windows'],
                'capabilities': ['network_testing', 'connectivity_check']
            },
            
            # Text processing
            'grep': {
                'category': ToolCategory.TEXT_PROCESSING,
                'description': 'Search text patterns',
                'platforms': ['linux', 'macos'],
                'capabilities': ['pattern_matching', 'text_search', 'regex']
            },
            'find': {
                'category': ToolCategory.FILE_MANAGEMENT,
                'description': 'Search for files and directories',
                'platforms': ['linux', 'macos'],
                'capabilities': ['file_search', 'recursive_search', 'pattern_matching']
            },
            'cat': {
                'category': ToolCategory.TEXT_PROCESSING,
                'description': 'Concatenate and display files',
                'platforms': ['linux', 'macos'],
                'capabilities': ['file_display', 'text_concatenation']
            },
            'type': {
                'category': ToolCategory.TEXT_PROCESSING,
                'description': 'Display file contents (Windows)',
                'platforms': ['windows'],
                'capabilities': ['file_display']
            },
            
            # Process management
            'ps': {
                'category': ToolCategory.SYSTEM,
                'description': 'Process status',
                'platforms': ['linux', 'macos'],
                'capabilities': ['process_listing', 'system_monitoring']
            },
            'tasklist': {
                'category': ToolCategory.SYSTEM,
                'description': 'List running processes (Windows)',
                'platforms': ['windows'],
                'capabilities': ['process_listing', 'system_monitoring']
            },
            'kill': {
                'category': ToolCategory.SYSTEM,
                'description': 'Terminate processes',
                'platforms': ['linux', 'macos'],
                'capabilities': ['process_termination', 'signal_sending']
            },
            'taskkill': {
                'category': ToolCategory.SYSTEM,
                'description': 'Terminate processes (Windows)',
                'platforms': ['windows'],
                'capabilities': ['process_termination', 'force_kill']
            }
        }
    
    def _get_system_paths(self) -> List[str]:
        """Get system paths where tools might be located."""
        paths = []
        
        # Add PATH environment variable paths
        path_env = os.environ.get('PATH', '')
        paths.extend(path_env.split(self.platform_adapter.get_path_separator()))
        
        # Add common installation directories
        if self.platform_adapter.platform_info.is_windows:
            # Windows common paths
            paths.extend([
                r'C:\Windows\System32',
                r'C:\Windows',
                r'C:\Program Files',
                r'C:\Program Files (x86)',
                r'C:\Windows\SysWOW64',
            ])
        else:
            # POSIX common paths
            paths.extend([
                '/usr/bin',
                '/usr/local/bin',
                '/bin',
                '/sbin',
                '/usr/sbin',
                '/opt/local/bin',
                '/opt/bin',
            ])
        
        # Remove duplicates and filter existing paths
        return [p for p in set(paths) if os.path.exists(p)]
    
    def discover_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Discover a specific tool on the system.
        
        Args:
            tool_name: Name of the tool to discover
            
        Returns:
            ToolMetadata object if found, None otherwise
        """
        tool_name = tool_name.lower()
        
        # Check built-in tools first
        if tool_name in self._builtin_tools:
            builtin_info = self._builtin_tools[tool_name]
            
            # Check if tool is available
            is_available = self._check_tool_availability(tool_name)
            
            # Get actual path if available
            tool_path = self._find_tool_path(tool_name) if is_available else None
            
            # Get version if available
            version = self._get_tool_version(tool_name) if is_available else None
            
            # Check platform compatibility
            platform_name = self._get_platform_name()
            platform_compatible = platform_name in builtin_info.get('platforms', [])
            
            return ToolMetadata(
                name=tool_name,
                version=version,
                description=builtin_info.get('description'),
                category=builtin_info.get('category'),
                status=ToolStatus.AVAILABLE if is_available and platform_compatible else ToolStatus.UNAVAILABLE,
                path=tool_path,
                platforms=builtin_info.get('platforms', []),
                capabilities=builtin_info.get('capabilities', []),
                tags=[tool_name, builtin_info.get('category', ToolCategory.UTILITIES).value],
                last_verified=datetime.now(timezone.utc) if is_available else None,
                verification_errors=[] if is_available else ['Tool not found or not accessible']
            )
        
        return None
    
    def _check_tool_availability(self, tool_name: str) -> bool:
        """Check if a tool is available on the system."""
        try:
            # Try to find the tool using 'which' (POSIX) or 'where' (Windows)
            if self.platform_adapter.platform_info.is_windows:
                result = subprocess.run(
                    ['where', tool_name],
                    capture_output=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ['which', tool_name],
                    capture_output=True,
                    timeout=5
                )
            
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def _find_tool_path(self, tool_name: str) -> Optional[str]:
        """Find the actual path to a tool."""
        for path in self._system_paths:
            tool_path = os.path.join(path, tool_name)
            if self.platform_adapter.platform_info.is_windows:
                # Check for .exe extension on Windows
                exe_path = f"{tool_path}.exe"
                if os.path.exists(exe_path):
                    return exe_path
            else:
                # Check regular executable on POSIX
                if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
                    return tool_path
        
        return None
    
    def _get_tool_version(self, tool_name: str) -> Optional[str]:
        """Get version information for a tool."""
        version_commands = ['--version', '-v', '-V', 'version', '--ver']
        
        for version_cmd in version_commands:
            try:
                result = subprocess.run(
                    [tool_name, version_cmd],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    return self._parse_version_output(result.stdout, result.stderr)
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
                continue
        
        return None
    
    def _parse_version_output(self, stdout: str, stderr: str) -> str:
        """Parse version information from command output."""
        output = f"{stdout} {stderr}".strip()
        
        # Common version patterns
        import re
        
        version_patterns = [
            r'version\s+(\d+\.\d+\.\d+)',
            r'version\s+(\d+\.\d+)',
            r'(\d+\.\d+\.\d+)',
            r'v?(\d+\.\d+\.\d+)',
        ]
        
        for pattern in version_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return "Unknown"
    
    def _get_platform_name(self) -> str:
        """Get normalized platform name."""
        system = self.platform_adapter.platform_info.name.lower()
        
        if system == 'darwin':
            return 'macos'
        elif system in ['linux', 'windows']:
            return system
        else:
            return system
    
    def discover_all_tools(self, categories: Optional[List[ToolCategory]] = None) -> List[ToolMetadata]:
        """
        Discover all available tools.
        
        Args:
            categories: Optional list of categories to limit discovery to
            
        Returns:
            List of ToolMetadata objects
        """
        tools = []
        
        for tool_name in self._builtin_tools:
            tool_info = self._builtin_tools[tool_name]
            
            # Filter by categories if specified
            if categories and tool_info.get('category') not in categories:
                continue
            
            tool_metadata = self.discover_tool(tool_name)
            if tool_metadata:
                tools.append(tool_metadata)
        
        return tools


class ToolRegistry:
    """
    Comprehensive tool management and discovery system.
    
    Provides capabilities for registering, discovering, managing, and
    utilizing tools across different platforms and environments.
    """
    
    def __init__(self, platform_adapter: Optional[PlatformAdapter] = None):
        """
        Initialize the tool registry.
        
        Args:
            platform_adapter: Platform adapter instance (uses default if None)
        """
        self.platform_adapter = platform_adapter or get_platform_adapter()
        self.shell_detector = ShellDetector(self.platform_adapter)
        self.discovery = ToolDiscovery(self.platform_adapter)
        
        # Registry storage
        self._registered_tools: Dict[str, ToolMetadata] = {}
        self._tool_aliases: Dict[str, str] = {}
        self._category_index: Dict[ToolCategory, Set[str]] = {}
        self._capability_index: Dict[str, Set[str]] = {}
        
        # Configuration
        self.auto_discovery = True
        self.cache_discoveries = True
        self.auto_update_status = True
        
        # Initialize with built-in tools
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Initialize the registry with built-in tools."""
        if self.auto_discovery:
            self.discover_and_register_tools()
    
    def register_tool(self, tool_metadata: ToolMetadata) -> bool:
        """
        Register a tool in the registry.
        
        Args:
            tool_metadata: Tool metadata to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            tool_name = tool_metadata.name.lower()
            
            # Update status if auto-update is enabled
            if self.auto_update_status:
                self._update_tool_status(tool_metadata)
            
            # Store in registry
            self._registered_tools[tool_name] = tool_metadata
            
            # Update indices
            self._update_indices(tool_metadata)
            
            # Register aliases
            for alias in tool_metadata.aliases:
                self._tool_aliases[alias.lower()] = tool_name
            
            return True
            
        except Exception:
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        tool_name = tool_name.lower()
        
        if tool_name not in self._registered_tools:
            return False
        
        tool_metadata = self._registered_tools[tool_name]
        
        # Remove from main registry
        del self._registered_tools[tool_name]
        
        # Remove from category index
        if tool_metadata.category:
            category_tools = self._category_index.get(tool_metadata.category, set())
            category_tools.discard(tool_name)
            if not category_tools:
                del self._category_index[tool_metadata.category]
        
        # Remove from capability index
        for capability in tool_metadata.capabilities:
            capability_tools = self._capability_index.get(capability, set())
            capability_tools.discard(tool_name)
            if not capability_tools:
                del self._capability_index[capability]
        
        # Remove aliases
        aliases_to_remove = [alias for alias, orig in self._tool_aliases.items() if orig == tool_name]
        for alias in aliases_to_remove:
            del self._tool_aliases[alias]
        
        return True
    
    def discover_and_register_tools(self, categories: Optional[List[ToolCategory]] = None) -> int:
        """
        Discover and register all available tools.
        
        Args:
            categories: Optional list of categories to limit discovery to
            
        Returns:
            Number of tools successfully registered
        """
        discovered_tools = self.discovery.discover_all_tools(categories)
        registered_count = 0
        
        for tool_metadata in discovered_tools:
            if self.register_tool(tool_metadata):
                registered_count += 1
        
        return registered_count
    
    def get_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Get tool metadata by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolMetadata object if found, None otherwise
        """
        tool_name = tool_name.lower()
        
        # Check direct name
        if tool_name in self._registered_tools:
            return self._registered_tools[tool_name]
        
        # Check alias
        actual_name = self._tool_aliases.get(tool_name)
        if actual_name:
            return self._registered_tools.get(actual_name)
        
        return None
    
    def get_tools_by_category(self, category: ToolCategory) -> List[ToolMetadata]:
        """
        Get all tools in a specific category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List of ToolMetadata objects in the category
        """
        tool_names = self._category_index.get(category, set())
        return [self._registered_tools[name] for name in tool_names if name in self._registered_tools]
    
    def get_tools_by_capability(self, capability: str) -> List[ToolMetadata]:
        """
        Get all tools that provide a specific capability.
        
        Args:
            capability: Capability to filter by
            
        Returns:
            List of ToolMetadata objects with the capability
        """
        tool_names = self._capability_index.get(capability, set())
        return [self._registered_tools[name] for name in tool_names if name in self._registered_tools]
    
    def search_tools(self, query: str) -> List[ToolMetadata]:
        """
        Search for tools by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of ToolMetadata objects matching the query
        """
        query = query.lower()
        results = []
        
        for tool_metadata in self._registered_tools.values():
            # Search in name
            if query in tool_metadata.name.lower():
                results.append(tool_metadata)
                continue
            
            # Search in description
            if tool_metadata.description and query in tool_metadata.description.lower():
                results.append(tool_metadata)
                continue
            
            # Search in tags
            if any(query in tag.lower() for tag in tool_metadata.tags):
                results.append(tool_metadata)
                continue
            
            # Search in aliases
            if any(query in alias.lower() for alias in tool_metadata.aliases):
                results.append(tool_metadata)
                continue
        
        return results
    
    def update_tool_status(self, tool_name: str) -> bool:
        """
        Update the status of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if update successful, False otherwise
        """
        tool_metadata = self.get_tool(tool_name)
        if not tool_metadata:
            return False
        
        return self._update_tool_status(tool_metadata)
    
    def _update_tool_status(self, tool_metadata: ToolMetadata) -> bool:
        """Update the status of a tool."""
        try:
            # Check availability
            is_available = self.discovery._check_tool_availability(tool_metadata.name)
            
            if is_available:
                tool_metadata.status = ToolStatus.AVAILABLE
                tool_metadata.last_verified = datetime.now(timezone.utc)
                tool_metadata.verification_errors = []
                
                # Update path if needed
                if not tool_metadata.path:
                    tool_metadata.path = self.discovery._find_tool_path(tool_metadata.name)
                
                # Update version if needed
                if not tool_metadata.version:
                    tool_metadata.version = self.discovery._get_tool_version(tool_metadata.name)
            else:
                tool_metadata.status = ToolStatus.UNAVAILABLE
                tool_metadata.verification_errors.append('Tool not accessible')
            
            return True
            
        except Exception as e:
            tool_metadata.verification_errors.append(f'Status update failed: {e}')
            return False
    
    def _update_indices(self, tool_metadata: ToolMetadata) -> None:
        """Update search indices for a tool."""
        tool_name = tool_metadata.name.lower()
        
        # Update category index
        if tool_metadata.category:
            if tool_metadata.category not in self._category_index:
                self._category_index[tool_metadata.category] = set()
            self._category_index[tool_metadata.category].add(tool_name)
        
        # Update capability index
        for capability in tool_metadata.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(tool_name)
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the registry contents.
        
        Returns:
            Dictionary containing registry summary
        """
        category_counts = {cat.value: len(tools) for cat, tools in self._category_index.items()}
        capability_counts = {cap: len(tools) for cap, tools in self._capability_index.items()}
        
        available_tools = [t for t in self._registered_tools.values() if t.status == ToolStatus.AVAILABLE]
        
        return {
            'total_tools': len(self._registered_tools),
            'available_tools': len(available_tools),
            'unavailable_tools': len(self._registered_tools) - len(available_tools),
            'tool_aliases': len(self._tool_aliases),
            'categories': {
                'total_categories': len(self._category_index),
                'category_counts': category_counts
            },
            'capabilities': {
                'total_capabilities': len(self._capability_index),
                'capability_counts': capability_counts
            },
            'platform': {
                'name': self.platform_adapter.platform_info.name,
                'is_windows': self.platform_adapter.platform_info.is_windows,
                'is_posix': self.platform_adapter.platform_info.is_posix
            },
            'configuration': {
                'auto_discovery': self.auto_discovery,
                'cache_discoveries': self.cache_discoveries,
                'auto_update_status': self.auto_update_status
            }
        }
    
    def export_registry(self, filepath: Union[str, Path], format: str = 'json') -> bool:
        """
        Export the registry to a file.
        
        Args:
            filepath: Path to export file
            format: Export format ('json' or 'yaml')
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            data = {
                'metadata': {
                    'exported_at': datetime.now(timezone.utc).isoformat(),
                    'platform': self.platform_adapter.platform_info.name,
                    'format': format,
                    'version': '1.0'
                },
                'tools': {name: asdict(tool) for name, tool in self._registered_tools.items()},
                'aliases': self._tool_aliases,
                'indices': {
                    'categories': {cat.value: list(tools) for cat, tools in self._category_index.items()},
                    'capabilities': {cap: list(tools) for cap, tools in self._capability_index.items()}
                }
            }
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == 'yaml':
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
            
        except Exception:
            return False
    
    def import_registry(self, filepath: Union[str, Path]) -> bool:
        """
        Import a registry from a file.
        
        Args:
            filepath: Path to import file
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return False
            
            with open(filepath, 'r') as f:
                if filepath.suffix.lower() == '.json':
                    data = json.load(f)
                elif filepath.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            # Import tools
            for name, tool_data in data.get('tools', {}).items():
                tool_metadata = ToolMetadata(**tool_data)
                self.register_tool(tool_metadata)
            
            # Import aliases
            self._tool_aliases.update(data.get('aliases', {}))
            
            # Rebuild indices
            self._rebuild_indices()
            
            return True
            
        except Exception:
            return False
    
    def _rebuild_indices(self) -> None:
        """Rebuild search indices from registered tools."""
        self._category_index.clear()
        self._capability_index.clear()
        
        for tool_metadata in self._registered_tools.values():
            self._update_indices(tool_metadata)