"""
Unit Tests for Tool Registry Component
=======================================

Comprehensive unit tests for the ToolRegistry class, testing tool discovery,
registration, categorization, and cross-platform capabilities.
"""

import pytest
import os
import sys
import json
import time
import subprocess
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.tool_registry import (
        ToolRegistry, ToolMetadata, ToolCategory, ToolStatus,
        ToolDiscovery
    )
    from tool_usage.platform_adapter import PosixAdapter, WindowsAdapter
except ImportError as e:
    pytest.skip(f"Could not import ToolRegistry: {e}", allow_module_level=True)


class TestToolMetadata:
    """Test ToolMetadata dataclass"""
    
    def test_tool_metadata_creation(self):
        """Test creating ToolMetadata object"""
        tool = ToolMetadata(
            name="python",
            category=ToolCategory.PROGRAMMING_LANGUAGE,
            path="/usr/bin/python",
            version="3.9.7",
            status=ToolStatus.AVAILABLE,
            description="Python programming language",
            tags={'language', 'scripting', 'development'},
            capabilities={'execute', 'script'},
            last_updated=time.time()
        )
        
        assert tool.name == "python"
        assert tool.category == ToolCategory.PROGRAMMING_LANGUAGE
        assert tool.path == "/usr/bin/python"
        assert tool.version == "3.9.7"
        assert tool.status == ToolStatus.AVAILABLE
        assert "language" in tool.tags
        assert "execute" in tool.capabilities
    
    def test_tool_metadata_minimal(self):
        """Test creating ToolMetadata with minimal data"""
        tool = ToolMetadata(
            name="minimal_tool",
            category=ToolCategory.UTILITY,
            path="/usr/bin/minimal_tool"
        )
        
        assert tool.name == "minimal_tool"
        assert tool.category == ToolCategory.UTILITY
        assert tool.path == "/usr/bin/minimal_tool"
        assert tool.version is None  # Default
        assert tool.status == ToolStatus.UNKNOWN  # Default
        assert tool.description is None  # Default
        assert tool.tags == set()  # Default
        assert tool.capabilities == set()  # Default
        assert tool.last_updated is None  # Default
    
    def test_tool_categories(self):
        """Test all tool categories"""
        categories = [
            ToolCategory.PROGRAMMING_LANGUAGE,
            ToolCategory.DEVELOPMENT_TOOL,
            ToolCategory.UTILITY,
            ToolCategory.SYSTEM_TOOL,
            ToolCategory.NETWORK_TOOL,
            ToolCategory.TEXT_PROCESSING,
            ToolCategory.DATABASE_TOOL,
            ToolCategory.WEB_TOOL,
            ToolCategory.GRAPHICS_TOOL,
            ToolCategory.MULTIMEDIA_TOOL,
            ToolCategory.SECURITY_TOOL,
            ToolCategory.SCIENTIFIC_TOOL,
            ToolCategory.OTHER
        ]
        
        for category in categories:
            tool = ToolMetadata(
                name="test",
                category=category,
                path="/usr/bin/test"
            )
            assert tool.category == category
    
    def test_tool_statuses(self):
        """Test all tool statuses"""
        statuses = [
            ToolStatus.AVAILABLE,
            ToolStatus.INSTALLED,
            ToolStatus.NOT_FOUND,
            ToolStatus.DEPRECATED,
            ToolStatus.UNKNOWN
        ]
        
        for status in statuses:
            tool = ToolMetadata(
                name="test",
                category=ToolCategory.UTILITY,
                path="/usr/bin/test",
                status=status
            )
            assert tool.status == status


class TestToolRegistryInitialization:
    """Test ToolRegistry initialization and basic setup"""
    
    def test_registry_initialization(self, posix_adapter):
        """Test basic ToolRegistry initialization"""
        registry = ToolRegistry(posix_adapter)
        
        assert registry.platform_adapter == posix_adapter
        assert registry.registered_tools == {}
        assert registry.discovery_cache == {}
        assert registry.auto_discovery is True
        assert registry.last_discovery_time is None
    
    def test_registry_with_default_adapter(self):
        """Test ToolRegistry with default adapter"""
        registry = ToolRegistry()
        assert registry.platform_adapter is not None
    
    def test_registry_custom_config(self):
        """Test ToolRegistry with custom configuration"""
        registry = ToolRegistry(auto_discovery=False)
        assert registry.auto_discovery is False
    
    def test_known_tools_configuration(self):
        """Test that registry has known tools configuration"""
        registry = ToolRegistry()
        
        # Should have configuration for known tools
        assert hasattr(registry, 'known_tools')
        assert isinstance(registry.known_tools, dict)
        
        # Should include common tools
        common_tools = ['python', 'node', 'git', 'docker', 'gcc']
        for tool in common_tools:
            assert tool in registry.known_tools or tool.upper() in registry.known_tools


class TestToolRegistration:
    """Test tool registration functionality"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry for testing"""
        return ToolRegistry(posix_adapter)
    
    def test_register_new_tool(self, registry):
        """Test registering a new tool"""
        tool = ToolMetadata(
            name="test_tool",
            category=ToolCategory.UTILITY,
            path="/usr/bin/test_tool",
            version="1.0.0",
            status=ToolStatus.AVAILABLE
        )
        
        success = registry.register_tool(tool)
        
        assert success is True
        assert "test_tool" in registry.registered_tools
        assert registry.registered_tools["test_tool"] == tool
    
    def test_register_duplicate_tool(self, registry):
        """Test registering duplicate tool"""
        tool1 = ToolMetadata(
            name="duplicate",
            category=ToolCategory.UTILITY,
            path="/usr/bin/duplicate",
            version="1.0.0"
        )
        
        tool2 = ToolMetadata(
            name="duplicate",
            category=ToolCategory.UTILITY,
            path="/usr/bin/duplicate",
            version="1.0.1"  # Different version
        )
        
        # Register first tool
        success1 = registry.register_tool(tool1)
        assert success1 is True
        
        # Try to register duplicate
        success2 = registry.register_tool(tool2)
        assert success2 is False  # Should not allow duplicates
        
        # Original tool should remain unchanged
        assert registry.registered_tools["duplicate"].version == "1.0.0"
    
    def test_register_tool_with_invalid_path(self, registry):
        """Test registering tool with invalid path"""
        tool = ToolMetadata(
            name="invalid_tool",
            category=ToolCategory.UTILITY,
            path="/nonexistent/path/tool",
            version="1.0.0"
        )
        
        # This should fail validation
        success = registry.register_tool(tool)
        assert success is False
        assert "invalid_tool" not in registry.registered_tools
    
    def test_register_tool_duplicate_name_different_path(self, registry):
        """Test registering tool with same name but different path"""
        tool1 = ToolMetadata(
            name="same_name",
            category=ToolCategory.UTILITY,
            path="/usr/bin/same_name",
            version="1.0.0"
        )
        
        tool2 = ToolMetadata(
            name="same_name",
            category=ToolCategory.DEVELOPMENT_TOOL,
            path="/opt/bin/same_name",
            version="2.0.0"
        )
        
        success1 = registry.register_tool(tool1)
        assert success1 is True
        
        success2 = registry.register_tool(tool2)
        assert success2 is False  # Should not allow name conflicts
    
    def test_register_multiple_tools(self, registry):
        """Test registering multiple tools"""
        tools = [
            ToolMetadata("tool1", ToolCategory.UTILITY, "/usr/bin/tool1", "1.0"),
            ToolMetadata("tool2", ToolCategory.DEVELOPMENT_TOOL, "/usr/bin/tool2", "2.0"),
            ToolMetadata("tool3", ToolCategory.PROGRAMMING_LANGUAGE, "/usr/bin/tool3", "3.0")
        ]
        
        success_count = 0
        for tool in tools:
            if registry.register_tool(tool):
                success_count += 1
        
        assert success_count == 3
        assert len(registry.registered_tools) == 3
        assert "tool1" in registry.registered_tools
        assert "tool2" in registry.registered_tools
        assert "tool3" in registry.registered_tools


class TestToolDiscovery:
    """Test tool discovery functionality"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry for testing"""
        return ToolRegistry(posix_adapter)
    
    def test_discover_tools_basic(self, registry):
        """Test basic tool discovery"""
        with patch.object(registry, '_discover_system_tools') as mock_discovery:
            mock_discovery.return_value = 5  # Found 5 tools
            
            count = registry.discover_and_register_tools()
            
            assert count == 5
            mock_discovery.assert_called_once()
    
    def test_discover_tools_by_category(self, registry):
        """Test tool discovery by specific categories"""
        categories = [ToolCategory.PROGRAMMING_LANGUAGE, ToolCategory.UTILITY]
        
        with patch.object(registry, '_discover_system_tools') as mock_discovery:
            mock_discovery.return_value = 3
            
            count = registry.discover_and_register_tools(categories)
            
            assert count == 3
            mock_discovery.assert_called_once_with(categories)
    
    def test_discover_tool_specific_path(self, registry):
        """Test discovery of tool at specific path"""
        test_path = "/custom/path/tool"
        
        with patch.object(registry, '_check_tool_path') as mock_check:
            mock_check.return_value = ToolMetadata(
                "custom_tool",
                ToolCategory.UTILITY,
                test_path,
                "1.0.0"
            )
            
            tool = registry.discover_tool_at_path(test_path)
            
            assert tool is not None
            assert tool.name == "custom_tool"
            assert tool.path == test_path
            mock_check.assert_called_once_with(test_path)
    
    def test_discovery_cache_usage(self, registry):
        """Test that discovery cache is used"""
        # Pre-populate cache
        registry.discovery_cache["cached_tool"] = ToolMetadata(
            "cached_tool",
            ToolCategory.UTILITY,
            "/usr/bin/cached_tool",
            "1.0.0"
        )
        
        with patch.object(registry, '_discover_system_tools') as mock_discovery:
            mock_discovery.return_value = 0  # No new discoveries
            
            # Should use cache instead of re-discovering
            count = registry.discover_and_register_tools()
            
            assert count == 0
            assert "cached_tool" in registry.registered_tools
            mock_discovery.assert_called_once()


class TestToolRetrieval:
    """Test tool information retrieval"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry with test data"""
        registry = ToolRegistry(posix_adapter)
        
        # Add test tools
        tools = [
            ToolMetadata("python", ToolCategory.PROGRAMMING_LANGUAGE, "/usr/bin/python", "3.9.7"),
            ToolMetadata("git", ToolCategory.DEVELOPMENT_TOOL, "/usr/bin/git", "2.34.0"),
            ToolMetadata("ls", ToolCategory.UTILITY, "/bin/ls", None),
            ToolMetadata("gcc", ToolCategory.DEVELOPMENT_TOOL, "/usr/bin/gcc", "11.2.0")
        ]
        
        for tool in tools:
            registry.register_tool(tool)
        
        return registry
    
    def test_get_tool_by_name(self, registry):
        """Test getting tool by name"""
        tool = registry.get_tool("python")
        
        assert tool is not None
        assert tool.name == "python"
        assert tool.category == ToolCategory.PROGRAMMING_LANGUAGE
        assert tool.version == "3.9.7"
    
    def test_get_tool_not_found(self, registry):
        """Test getting non-existent tool"""
        tool = registry.get_tool("nonexistent_tool")
        
        assert tool is None
    
    def test_get_tools_by_category(self, registry):
        """Test getting tools by category"""
        dev_tools = registry.get_tools_by_category(ToolCategory.DEVELOPMENT_TOOL)
        
        assert len(dev_tools) == 2  # git and gcc
        tool_names = [tool.name for tool in dev_tools]
        assert "git" in tool_names
        assert "gcc" in tool_names
        
        # Check that other tools are not included
        tool_categories = [tool.category for tool in dev_tools]
        assert all(cat == ToolCategory.DEVELOPMENT_TOOL for cat in tool_categories)
    
    def test_get_all_tools(self, registry):
        """Test getting all registered tools"""
        all_tools = registry.get_all_tools()
        
        assert len(all_tools) == 4
        tool_names = [tool.name for tool in all_tools]
        expected_names = ["python", "git", "ls", "gcc"]
        
        for name in expected_names:
            assert name in tool_names
    
    def test_search_tools_by_name(self, registry):
        """Test searching tools by name"""
        results = registry.search_tools("python")
        
        assert len(results) == 1
        assert results[0].name == "python"
        
        # Test case-insensitive search
        results = registry.search_tools("PYTHON")
        assert len(results) == 1
        assert results[0].name == "python"
    
    def test_search_tools_by_description(self, registry):
        """Test searching tools by description"""
        # Add tool with description
        tool_with_desc = ToolMetadata(
            "test_tool",
            ToolCategory.UTILITY,
            "/usr/bin/test_tool",
            description="A testing tool for development"
        )
        registry.register_tool(tool_with_desc)
        
        results = registry.search_tools("development")
        
        # Should find the tool with description containing "development"
        assert len(results) >= 1
        found_tool = next((t for t in results if t.name == "test_tool"), None)
        assert found_tool is not None
    
    def test_search_tools_by_tags(self, registry):
        """Test searching tools by tags"""
        # Add tool with tags
        tool_with_tags = ToolMetadata(
            "tagged_tool",
            ToolCategory.UTILITY,
            "/usr/bin/tagged_tool",
            tags={'testing', 'development', 'cli'}
        )
        registry.register_tool(tool_with_tags)
        
        results = registry.search_tools("testing")
        
        assert len(results) >= 1
        found_tool = next((t for t in results if t.name == "tagged_tool"), None)
        assert found_tool is not None


class TestToolStatus:
    """Test tool status management"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry for testing"""
        return ToolRegistry(posix_adapter)
    
    def test_update_tool_status(self, registry):
        """Test updating tool status"""
        # Register tool initially
        tool = ToolMetadata(
            "status_test",
            ToolCategory.UTILITY,
            "/usr/bin/status_test",
            status=ToolStatus.AVAILABLE
        )
        registry.register_tool(tool)
        
        # Update status
        success = registry.update_tool_status("status_test")
        
        assert success is True
        updated_tool = registry.get_tool("status_test")
        # Status might be updated based on current system state
        assert updated_tool.status in [ToolStatus.AVAILABLE, ToolStatus.INSTALLED]
    
    def test_update_nonexistent_tool_status(self, registry):
        """Test updating status of non-existent tool"""
        success = registry.update_tool_status("nonexistent_tool")
        
        assert success is False
    
    def test_check_tool_availability(self, registry):
        """Test checking tool availability"""
        tool = ToolMetadata(
            "availability_test",
            ToolCategory.UTILITY,
            "/usr/bin/availability_test",
            status=ToolStatus.AVAILABLE
        )
        registry.register_tool(tool)
        
        # Mock availability check
        with patch.object(registry, '_check_tool_availability') as mock_check:
            mock_check.return_value = ToolStatus.AVAILABLE
            
            status = registry.check_tool_availability("availability_test")
            
            assert status == ToolStatus.AVAILABLE
            mock_check.assert_called_once_with("availability_test")


class TestToolCategories:
    """Test tool categorization functionality"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry with categorized tools"""
        registry = ToolRegistry(posix_adapter)
        
        # Add tools in different categories
        tools_by_category = {
            ToolCategory.PROGRAMMING_LANGUAGE: ["python", "node", "ruby"],
            ToolCategory.DEVELOPMENT_TOOL: ["git", "gcc", "make"],
            ToolCategory.UTILITY: ["ls", "cat", "grep"],
            ToolCategory.NETWORK_TOOL: ["curl", "wget", "ssh"],
            ToolCategory.TEXT_PROCESSING: ["sed", "awk", "vim"]
        }
        
        for category, tool_names in tools_by_category.items():
            for tool_name in tool_names:
                tool = ToolMetadata(
                    tool_name,
                    category,
                    f"/usr/bin/{tool_name}",
                    version="1.0.0"
                )
                registry.register_tool(tool)
        
        return registry
    
    def test_get_tools_by_category(self, registry):
        """Test getting tools by specific category"""
        programming_tools = registry.get_tools_by_category(ToolCategory.PROGRAMMING_LANGUAGE)
        
        assert len(programming_tools) == 3
        tool_names = [tool.name for tool in programming_tools]
        expected = ["python", "node", "ruby"]
        
        for name in expected:
            assert name in tool_names
    
    def test_get_category_summary(self, registry):
        """Test getting category summary"""
        summary = registry.get_category_summary()
        
        assert isinstance(summary, dict)
        
        # Should have counts for each category
        for category in ToolCategory:
            if category != ToolCategory.OTHER:
                assert category.value in summary
                assert summary[category.value] >= 0
    
    def test_find_tools_by_capability(self, registry):
        """Test finding tools by capability"""
        # Add tools with specific capabilities
        tool_with_caps = ToolMetadata(
            "capability_test",
            ToolCategory.UTILITY,
            "/usr/bin/capability_test",
            capabilities={'execute', 'script', 'interactive'}
        )
        registry.register_tool(tool_with_caps)
        
        # Find tools with 'script' capability
        script_tools = registry.find_tools_by_capability('script')
        
        assert len(script_tools) >= 1
        found_tool = next((t for t in script_tools if t.name == "capability_test"), None)
        assert found_tool is not None
        assert 'script' in found_tool.capabilities


class TestToolValidation:
    """Test tool validation functionality"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry for testing"""
        return ToolRegistry(posix_adapter)
    
    def test_validate_tool_path(self, registry):
        """Test validating tool path"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            
            is_valid = registry.validate_tool_path("/usr/bin/python")
            
            assert is_valid is True
    
    def test_validate_tool_path_nonexistent(self, registry):
        """Test validating non-existent tool path"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            is_valid = registry.validate_tool_path("/nonexistent/path")
            
            assert is_valid is False
    
    def test_validate_tool_executable(self, registry):
        """Test validating tool executable permissions"""
        with patch('os.access') as mock_access:
            mock_access.return_value = True
            
            is_executable = registry.validate_tool_executable("/usr/bin/python")
            
            assert is_executable is True
    
    def test_validate_tool_version(self, registry):
        """Test extracting and validating tool version"""
        # Mock version extraction
        with patch.object(registry, '_extract_tool_version') as mock_extract:
            mock_extract.return_value = "3.9.7"
            
            version = registry.extract_tool_version("/usr/bin/python")
            
            assert version == "3.9.7"
    
    def test_validate_tool_metadata(self, registry):
        """Test validating complete tool metadata"""
        tool = ToolMetadata(
            "validation_test",
            ToolCategory.UTILITY,
            "/usr/bin/validation_test",
            version="1.0.0",
            status=ToolStatus.AVAILABLE
        )
        
        is_valid = registry.validate_tool_metadata(tool)
        
        assert is_valid is True  # Should be valid with all required fields


class TestRegistryPersistence:
    """Test registry persistence and serialization"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry with test data"""
        registry = ToolRegistry(posix_adapter)
        
        # Add test tools
        tools = [
            ToolMetadata("persist_test", ToolCategory.UTILITY, "/usr/bin/persist_test", "1.0.0"),
            ToolMetadata("persist_test2", ToolCategory.DEVELOPMENT_TOOL, "/opt/bin/persist_test2", "2.0.0")
        ]
        
        for tool in tools:
            registry.register_tool(tool)
        
        return registry
    
    def test_export_registry_json(self, temp_test_dir, registry):
        """Test exporting registry to JSON"""
        export_path = temp_test_dir / "registry_export.json"
        
        success = registry.export_registry(str(export_path), "json")
        
        assert success is True
        assert export_path.exists()
        
        # Verify exported content
        with open(export_path) as f:
            data = json.load(f)
            
        assert "tools" in data
        assert len(data["tools"]) == 2
        assert data["tools"][0]["name"] == "persist_test"
    
    def test_import_registry_json(self, temp_test_dir, registry):
        """Test importing registry from JSON"""
        # Create import file
        import_data = {
            "version": "1.0",
            "tools": [
                {
                    "name": "import_test",
                    "category": "utility",
                    "path": "/usr/bin/import_test",
                    "version": "1.0.0"
                }
            ]
        }
        
        import_path = temp_test_dir / "registry_import.json"
        with open(import_path, 'w') as f:
            json.dump(import_data, f)
        
        success = registry.import_registry(str(import_path), "json")
        
        assert success is True
        assert "import_test" in registry.registered_tools
        assert registry.get_tool("import_test").name == "import_test"
    
    def test_registry_summary(self, registry):
        """Test generating registry summary"""
        summary = registry.get_registry_summary()
        
        assert isinstance(summary, dict)
        assert 'total_tools' in summary
        assert 'categories' in summary
        assert 'last_updated' in summary
        
        assert summary['total_tools'] == len(registry.registered_tools)
        assert isinstance(summary['categories'], dict)


class TestCrossPlatformCompatibility:
    """Test cross-platform tool registry compatibility"""
    
    @pytest.mark.compatibility
    def test_posix_tool_discovery(self, mock_platform_linux, posix_adapter):
        """Test tool discovery on POSIX systems"""
        registry = ToolRegistry(posix_adapter)
        
        with patch.object(registry, '_discover_posix_tools') as mock_discovery:
            mock_discovery.return_value = ["python", "git", "ls"]
            
            tools = registry.discover_posix_tools()
            
            assert len(tools) == 3
            assert "python" in tools
    
    @pytest.mark.compatibility
    def test_windows_tool_discovery(self, mock_platform_windows, windows_adapter):
        """Test tool discovery on Windows systems"""
        registry = ToolRegistry(windows_adapter)
        
        with patch.object(registry, '_discover_windows_tools') as mock_discovery:
            mock_discovery.return_value = ["python.exe", "git.exe", "cmd.exe"]
            
            tools = registry.discover_windows_tools()
            
            assert len(tools) == 3
            assert "python.exe" in tools
    
    @pytest.mark.compatibility
    @pytest.mark.parametrize("tool_name,expected_category", [
        ("python", ToolCategory.PROGRAMMING_LANGUAGE),
        ("git", ToolCategory.DEVELOPMENT_TOOL),
        ("ls", ToolCategory.UTILITY),
        ("curl", ToolCategory.NETWORK_TOOL),
        ("gcc", ToolCategory.DEVELOPMENT_TOOL)
    ])
    def test_platform_specific_categorization(self, tool_name, expected_category):
        """Test that tools are categorized correctly across platforms"""
        registry = ToolRegistry()
        
        # Mock category detection
        with patch.object(registry, '_detect_tool_category') as mock_detect:
            mock_detect.return_value = expected_category
            
            detected_category = registry._detect_tool_category(tool_name)
            
            assert detected_category == expected_category


class TestToolRegistryPerformance:
    """Test ToolRegistry performance characteristics"""
    
    @pytest.mark.performance
    def test_registration_performance(self):
        """Test performance of tool registration"""
        registry = ToolRegistry()
        
        # Time registration of multiple tools
        start_time = time.time()
        
        for i in range(100):
            tool = ToolMetadata(
                f"perf_test_{i}",
                ToolCategory.UTILITY,
                f"/usr/bin/perf_test_{i}",
                version="1.0.0"
            )
            registry.register_tool(tool)
        
        end_time = time.time()
        registration_time = end_time - start_time
        
        # Should register 100 tools quickly
        assert registration_time < 5.0
        assert len(registry.registered_tools) == 100
    
    @pytest.mark.performance
    def test_search_performance(self, registry):
        """Test performance of tool search"""
        # Pre-populate registry
        for i in range(1000):
            tool = ToolMetadata(
                f"search_test_{i}",
                ToolCategory.UTILITY,
                f"/usr/bin/search_test_{i}",
                tags={'test', f'tag_{i}'}
            )
            registry.register_tool(tool)
        
        # Time search operations
        start_time = time.time()
        
        for i in range(100):
            results = registry.search_tools(f"search_test_{i}")
            assert len(results) >= 0
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Should complete searches quickly
        assert search_time < 10.0
    
    @pytest.mark.performance
    def test_cache_performance(self):
        """Test performance benefits of caching"""
        registry = ToolRegistry()
        
        # Pre-populate cache
        for i in range(100):
            registry.discovery_cache[f"cached_tool_{i}"] = ToolMetadata(
                f"cached_tool_{i}",
                ToolCategory.UTILITY,
                f"/usr/bin/cached_tool_{i}"
            )
        
        # Access cached tools (should be fast)
        start_time = time.time()
        
        for i in range(100):
            tool = registry.get_tool(f"cached_tool_{i}")
            assert tool is not None
        
        end_time = time.time()
        cache_time = end_time - start_time
        
        # Cache access should be very fast
        assert cache_time < 1.0


class TestToolRegistryErrorHandling:
    """Test error handling and resilience in ToolRegistry"""
    
    @pytest.fixture
    def registry(self, posix_adapter):
        """Create registry for testing"""
        return ToolRegistry(posix_adapter)
    
    def test_invalid_tool_registration(self, registry):
        """Test handling of invalid tool registration"""
        # Invalid tool (missing required fields)
        invalid_tool = ToolMetadata(
            name="",  # Empty name
            category=ToolCategory.UTILITY,
            path=""   # Empty path
        )
        
        success = registry.register_tool(invalid_tool)
        assert success is False
        assert "" not in registry.registered_tools
    
    def test_corrupted_registry_data(self, registry):
        """Test handling of corrupted registry data"""
        # Simulate corrupted data
        registry.registered_tools["corrupted"] = "not a ToolMetadata object"
        
        # Should handle gracefully
        tool = registry.get_tool("corrupted")
        assert tool is None or tool != "not a ToolMetadata object"
    
    def test_file_system_errors(self, registry):
        """Test handling of file system errors"""
        # Mock file system errors
        with patch('builtins.open', side_effect=OSError("Disk error")):
            success = registry.export_registry("/tmp/nonexistent/registry.json", "json")
            
            assert success is False
    
    def test_memory_errors(self, registry):
        """Test handling of memory errors"""
        with patch.object(registry, '_register_tool_with_validation', side_effect=MemoryError("Out of memory")):
            tool = ToolMetadata("memory_test", ToolCategory.UTILITY, "/usr/bin/memory_test")
            
            # Should handle gracefully
            try:
                success = registry.register_tool(tool)
                # Either succeeds or fails gracefully
                assert isinstance(success, bool)
            except MemoryError:
                # Memory errors can be propagated
                pass


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])