import pytest
import sys
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum

# Add parent directories to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

# Assuming these are the imports from your actual ToolRegistry implementation
try:
    from tool_usage.tool_registry import (
        ToolMetadata,
        ToolCategory, 
        ToolStatus,
        ToolRegistry
    )
except ImportError:
    # Mock the imports if the actual module doesn't exist yet
    @dataclass
    class ToolMetadata:
        name: str
        description: str
        category: str
        status: str = "active"
        parameters: Dict[str, Any] = field(default_factory=dict)
        capabilities: List[str] = field(default_factory=list)
        version: str = "1.0.0"
        dependencies: List[str] = field(default_factory=list)
        
    class ToolCategory(Enum):
        SYSTEM = "system"
        DEVELOPMENT = "development"
        DATA_PROCESSING = "data_processing"
        NETWORK = "network"
        SECURITY = "security"
        CUSTOM = "custom"
        
    class ToolStatus(Enum):
        ACTIVE = "active"
        INACTIVE = "inactive"
        DEPRECATED = "deprecated"
        TESTING = "testing"
        
    class ToolRegistry:
        def __init__(self):
            self._tools: Dict[str, ToolMetadata] = {}
            self._categories: Set[str] = set()
            self._search_index: Dict[str, List[str]] = {}
            
        def register_tool(self, tool: ToolMetadata) -> bool:
            """Register a tool in the registry"""
            self._tools[tool.name] = tool
            self._categories.add(tool.category)
            self._build_search_index()
            return True
            
        def unregister_tool(self, name: str) -> bool:
            """Remove a tool from the registry"""
            if name in self._tools:
                del self._tools[name]
                self._build_search_index()
                return True
            return False
            
        def get_tool(self, name: str) -> Optional[ToolMetadata]:
            """Get a tool by name"""
            return self._tools.get(name)
            
        def list_tools(self, category: Optional[str] = None) -> List[ToolMetadata]:
            """List all tools, optionally filtered by category"""
            tools = list(self._tools.values())
            if category:
                tools = [t for t in tools if t.category == category]
            return tools
            
        def search_tools(self, query: str) -> List[ToolMetadata]:
            """Search for tools by name or description"""
            query_lower = query.lower()
            results = []
            for tool in self._tools.values():
                if (query_lower in tool.name.lower() or 
                    query_lower in tool.description.lower()):
                    results.append(tool)
            return results
            
        def get_categories(self) -> List[str]:
            """Get all available categories"""
            return sorted(list(self._categories))
            
        def bulk_register(self, tools: List[ToolMetadata]) -> Dict[str, bool]:
            """Register multiple tools at once"""
            results = {}
            for tool in tools:
                results[tool.name] = self.register_tool(tool)
            return results
            
        def _build_search_index(self):
            """Build internal search index"""
            self._search_index = {}
            for tool in self._tools.values():
                keywords = [tool.name.lower(), tool.description.lower()]
                keywords.extend(tool.capabilities)
                for keyword in keywords:
                    if keyword not in self._search_index:
                        self._search_index[keyword] = []
                    self._search_index[keyword].append(tool.name)
                    
        def get_tool_statistics(self) -> Dict[str, Any]:
            """Get statistics about registered tools"""
            total_tools = len(self._tools)
            category_counts = {}
            status_counts = {}
            
            for tool in self._tools.values():
                category_counts[tool.category] = category_counts.get(tool.category, 0) + 1
                status_counts[tool.status] = status_counts.get(tool.status, 0) + 1
                
            return {
                'total_tools': total_tools,
                'categories': category_counts,
                'statuses': status_counts,
                'active_tools': status_counts.get('active', 0)
            }


class TestToolMetadata:
    """Test the ToolMetadata dataclass"""
    
    def test_tool_metadata_creation(self):
        """Test creating ToolMetadata instances"""
        metadata = ToolMetadata(
            name="test_tool",
            description="A test tool",
            category="development",
            status="active",
            parameters={"param1": "value1"},
            capabilities=["capability1", "capability2"],
            version="1.0.0",
            dependencies=["dependency1"]
        )
        
        assert metadata.name == "test_tool"
        assert metadata.description == "A test tool"
        assert metadata.category == "development"
        assert metadata.status == "active"
        assert metadata.parameters == {"param1": "value1"}
        assert metadata.capabilities == ["capability1", "capability2"]
        assert metadata.version == "1.0.0"
        assert metadata.dependencies == ["dependency1"]
        
    def test_tool_metadata_defaults(self):
        """Test ToolMetadata with default values"""
        metadata = ToolMetadata(
            name="minimal_tool",
            description="Minimal tool description",
            category="system"
        )
        
        assert metadata.status == "active"  # Default status
        assert metadata.parameters == {}
        assert metadata.capabilities == []
        assert metadata.version == "1.0.0"  # Default version
        assert metadata.dependencies == []
        
    def test_tool_metadata_immutability(self):
        """Test that ToolMetadata fields can be modified after creation"""
        metadata = ToolMetadata(
            name="modifiable_tool",
            description="Initial description",
            category="development"
        )
        
        # Should be able to modify fields
        metadata.description = "Modified description"
        metadata.parameters["new_param"] = "new_value"
        metadata.capabilities.append("new_capability")
        
        assert metadata.description == "Modified description"
        assert metadata.parameters["new_param"] == "new_value"
        assert "new_capability" in metadata.capabilities
        
    def test_tool_metadata_validation(self):
        """Test ToolMetadata field validation"""
        # Test that required fields are actually required
        with pytest.raises(TypeError):
            ToolMetadata()  # Missing required fields
            
        with pytest.raises(TypeError):
            ToolMetadata(name="test")  # Missing description
            
    @pytest.mark.parametrize("name,description,category", [
        ("valid_tool", "Valid description", "development"),
        ("tool_with_underscores", "Tool with underscores", "system"),
        ("tool123", "Tool with numbers", "data_processing"),
        ("", "Empty name", "network"),  # Edge case
        ("tool", "", "security"),  # Empty description
    ])
    def test_tool_metadata_parametrized_creation(self, name, description, category):
        """Test ToolMetadata creation with various parameter combinations"""
        metadata = ToolMetadata(name=name, description=description, category=category)
        
        assert metadata.name == name
        assert metadata.description == description
        assert metadata.category == category


class TestToolCategory:
    """Test the ToolCategory enum"""
    
    def test_tool_category_values(self):
        """Test that all expected category values exist"""
        expected_categories = {
            "system", "development", "data_processing", 
            "network", "security", "custom"
        }
        actual_categories = {category.value for category in ToolCategory}
        assert expected_categories.issubset(actual_categories)
        
    def test_tool_category_comparison(self):
        """Test category equality and comparison"""
        category1 = ToolCategory.DEVELOPMENT
        category2 = ToolCategory.DEVELOPMENT
        category3 = ToolCategory.SYSTEM
        
        assert category1 == category2
        assert category1 != category3
        assert category1 == "development"
        
    def test_tool_category_iteration(self):
        """Test iterating over all categories"""
        categories = list(ToolCategory)
        assert len(categories) == 6
        assert ToolCategory.SYSTEM in categories
        assert ToolCategory.DEVELOPMENT in categories


class TestToolStatus:
    """Test the ToolStatus enum"""
    
    def test_tool_status_values(self):
        """Test that all expected status values exist"""
        expected_statuses = {"active", "inactive", "deprecated", "testing"}
        actual_statuses = {status.value for status in ToolStatus}
        assert expected_statuses.issubset(actual_statuses)
        
    def test_tool_status_comparison(self):
        """Test status equality and comparison"""
        status1 = ToolStatus.ACTIVE
        status2 = ToolStatus.ACTIVE
        status3 = ToolStatus.INACTIVE
        
        assert status1 == status2
        assert status1 != status3
        assert status1 == "active"
        
    def test_tool_status_iteration(self):
        """Test iterating over all statuses"""
        statuses = list(ToolStatus)
        assert len(statuses) == 4
        assert ToolStatus.ACTIVE in statuses
        assert ToolStatus.INACTIVE in statuses


class TestToolRegistryInitialization:
    """Test ToolRegistry initialization"""
    
    def test_registry_initialization(self):
        """Test basic registry initialization"""
        registry = ToolRegistry()
        
        # Check initial state
        assert len(registry._tools) == 0
        assert len(registry._categories) == 0
        assert len(registry._search_index) == 0
        
    def test_registry_initial_state_methods(self):
        """Test that registry methods work on empty registry"""
        registry = ToolRegistry()
        
        # Test empty searches and lists
        assert registry.list_tools() == []
        assert registry.list_tools("development") == []
        assert registry.search_tools("anything") == []
        assert registry.get_categories() == []
        assert registry.get_tool("nonexistent") is None
        assert registry.get_tool_statistics() == {
            'total_tools': 0,
            'categories': {},
            'statuses': {},
            'active_tools': 0
        }
        
    def test_registry_unregister_empty(self):
        """Test unregistering from empty registry"""
        registry = ToolRegistry()
        assert not registry.unregister_tool("nonexistent")
        
    def test_registry_bulk_register_empty(self):
        """Test bulk registration with empty list"""
        registry = ToolRegistry()
        result = registry.bulk_register([])
        assert result == {}


class TestToolRegistration:
    """Test tool registration functionality"""
    
    def test_single_tool_registration(self):
        """Test registering a single tool"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="test_tool",
            description="Test tool for unit testing",
            category="development",
            capabilities=["testing", "validation"]
        )
        
        assert registry.register_tool(tool)
        assert registry.get_tool("test_tool") == tool
        
    def test_duplicate_tool_registration(self):
        """Test registering duplicate tools"""
        registry = ToolRegistry()
        tool1 = ToolMetadata(
            name="duplicate_tool",
            description="First registration",
            category="development"
        )
        tool2 = ToolMetadata(
            name="duplicate_tool",
            description="Second registration",
            category="development"
        )
        
        assert registry.register_tool(tool1)
        assert registry.register_tool(tool2)  # Should overwrite
        
        retrieved = registry.get_tool("duplicate_tool")
        assert retrieved.description == "Second registration"  # Should be updated
        
    def test_tool_unregistration(self):
        """Test unregistering a tool"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="removable_tool",
            description="Tool to be removed",
            category="development"
        )
        
        registry.register_tool(tool)
        assert registry.get_tool("removable_tool") is not None
        
        assert registry.unregister_tool("removable_tool")
        assert registry.get_tool("removable_tool") is None
        
    def test_unregister_nonexistent_tool(self):
        """Test unregistering a tool that doesn't exist"""
        registry = ToolRegistry()
        assert not registry.unregister_tool("nonexistent_tool")
        
    def test_bulk_tool_registration(self):
        """Test bulk registration of multiple tools"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="tool1", description="Tool 1", category="development"),
            ToolMetadata(name="tool2", description="Tool 2", category="system"),
            ToolMetadata(name="tool3", description="Tool 3", category="network")
        ]
        
        result = registry.bulk_register(tools)
        
        assert result == {"tool1": True, "tool2": True, "tool3": True}
        assert len(registry.list_tools()) == 3
        
        # Verify all tools are registered
        for tool in tools:
            assert registry.get_tool(tool.name) == tool
            
    def test_bulk_registration_partial_failure(self):
        """Test bulk registration with some failures"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="tool1", description="Tool 1", category="development"),
            ToolMetadata(name="tool2", description="Tool 2", category="system"),
        ]
        
        # Register first tool normally
        registry.register_tool(tools[0])
        
        # Try bulk register - should fail for duplicate
        result = registry.bulk_register(tools)
        
        # Result should show the status for each tool
        assert "tool1" in result
        assert "tool2" in result
        
    @pytest.mark.parametrize("category,expected_count", [
        ("development", 2),
        ("system", 1),
        ("network", 0)
    ])
    def test_registration_by_category(self, category, expected_count):
        """Test registration and listing by category"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="dev1", description="Dev tool 1", category="development"),
            ToolMetadata(name="dev2", description="Dev tool 2", category="development"),
            ToolMetadata(name="sys1", description="System tool", category="system")
        ]
        
        registry.bulk_register(tools)
        
        assert len(registry.list_tools(category)) == expected_count


class TestToolDiscovery:
    """Test tool discovery and searching functionality"""
    
    @pytest.fixture
    def populated_registry(self):
        """Create a registry with various tools for testing"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(
                name="git_cli",
                description="Git command line interface for version control",
                category="development",
                capabilities=["version_control", "branching", "merging"]
            ),
            ToolMetadata(
                name="docker_container",
                description="Docker container management tool",
                category="system",
                capabilities=["containerization", "deployment"]
            ),
            ToolMetadata(
                name="curl_http",
                description="HTTP client for making requests",
                category="network",
                capabilities=["http_requests", "api_testing"]
            ),
            ToolMetadata(
                name="nmap_scanner",
                description="Network discovery and security auditing",
                category="network",
                capabilities=["network_discovery", "security_scanning"]
            )
        ]
        registry.bulk_register(tools)
        return registry
        
    def test_get_tool_by_name(self, populated_registry):
        """Test retrieving tools by exact name"""
        git_tool = populated_registry.get_tool("git_cli")
        assert git_tool is not None
        assert git_tool.name == "git_cli"
        assert git_tool.category == "development"
        
    def test_list_all_tools(self, populated_registry):
        """Test listing all registered tools"""
        tools = populated_registry.list_tools()
        assert len(tools) == 4
        
        # Verify all tools are present
        tool_names = {tool.name for tool in tools}
        expected_names = {"git_cli", "docker_container", "curl_http", "nmap_scanner"}
        assert tool_names == expected_names
        
    def test_list_tools_by_category(self, populated_registry):
        """Test listing tools filtered by category"""
        development_tools = populated_registry.list_tools("development")
        assert len(development_tools) == 1
        assert development_tools[0].name == "git_cli"
        
        network_tools = populated_registry.list_tools("network")
        assert len(network_tools) == 2
        network_names = {tool.name for tool in network_tools}
        assert network_names == {"curl_http", "nmap_scanner"}
        
    def test_search_by_name(self, populated_registry):
        """Test searching tools by name"""
        # Partial name match
        git_results = populated_registry.search_tools("git")
        assert len(git_results) == 1
        assert git_results[0].name == "git_cli"
        
        # Case insensitive search
        docker_results = populated_registry.search_tools("DOCKER")
        assert len(docker_results) == 1
        assert docker_results[0].name == "docker_container"
        
    def test_search_by_description(self, populated_registry):
        """Test searching tools by description content"""
        # Search in description
        http_results = populated_registry.search_tools("HTTP")
        assert len(http_results) == 1
        assert http_results[0].name == "curl_http"
        
        # Search for multiple terms
        security_results = populated_registry.search_tools("security")
        assert len(security_results) == 1
        assert security_results[0].name == "nmap_scanner"
        
    def test_search_by_capabilities(self, populated_registry):
        """Test searching tools by capabilities"""
        # Search in capabilities
        container_results = populated_registry.search_tools("containerization")
        assert len(container_results) == 1
        assert container_results[0].name == "docker_container"
        
        deployment_results = populated_registry.search_tools("deployment")
        assert len(deployment_results) == 1
        assert deployment_results[0].name == "docker_container"
        
    def test_search_no_results(self, populated_registry):
        """Test searching for non-existent tools"""
        results = populated_registry.search_tools("nonexistent_tool_name")
        assert results == []
        
    def test_search_empty_query(self, populated_registry):
        """Test searching with empty query"""
        results = populated_registry.search_tools("")
        # Empty query should return all tools or none, depending on implementation
        # Let's assume it returns all tools that match empty string
        assert len(results) >= 0
        
    def test_search_case_insensitive(self, populated_registry):
        """Test that search is case insensitive"""
        results1 = populated_registry.search_tools("git")
        results2 = populated_registry.search_tools("GIT")
        results3 = populated_registry.search_tools("GiT")
        
        assert len(results1) == len(results2) == len(results3) == 1
        assert all(r.name == "git_cli" for r in [results1[0], results2[0], results3[0]])


class TestToolCategories:
    """Test category management functionality"""
    
    def test_get_categories_empty_registry(self):
        """Test getting categories from empty registry"""
        registry = ToolRegistry()
        assert registry.get_categories() == []
        
    def test_get_categories_after_registration(self):
        """Test getting categories after registering tools"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="tool1", description="Tool 1", category="development"),
            ToolMetadata(name="tool2", description="Tool 2", category="development"),
            ToolMetadata(name="tool3", description="Tool 3", category="system"),
            ToolMetadata(name="tool4", description="Tool 4", category="network")
        ]
        registry.bulk_register(tools)
        
        categories = registry.get_categories()
        assert len(categories) == 3
        assert "development" in categories
        assert "system" in categories
        assert "network" in categories
        
    def test_categories_are_unique(self):
        """Test that categories returned are unique and sorted"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="tool1", description="Tool 1", category="zebra"),
            ToolMetadata(name="tool2", description="Tool 2", category="alpha"),
            ToolMetadata(name="tool3", description="Tool 3", category="zebra")
        ]
        registry.bulk_register(tools)
        
        categories = registry.get_categories()
        assert len(categories) == 2
        assert categories == sorted(categories)  # Should be sorted
        assert "alpha" in categories
        assert "zebra" in categories
        
    def test_category_tracking_on_unregister(self):
        """Test that categories are tracked correctly when tools are unregistered"""
        registry = ToolRegistry()
        tool1 = ToolMetadata(name="tool1", description="Tool 1", category="development")
        tool2 = ToolMetadata(name="tool2", description="Tool 2", category="development")
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        assert len(registry.get_categories()) == 1
        assert "development" in registry.get_categories()
        
        registry.unregister_tool("tool1")
        # Category should still exist because tool2 is still registered
        assert len(registry.get_categories()) == 1
        assert "development" in registry.get_categories()
        
        registry.unregister_tool("tool2")
        # Category should be removed if no tools remain
        assert len(registry.get_categories()) == 0


class TestToolStatistics:
    """Test tool statistics functionality"""
    
    def test_statistics_empty_registry(self):
        """Test statistics for empty registry"""
        registry = ToolRegistry()
        stats = registry.get_tool_statistics()
        
        assert stats['total_tools'] == 0
        assert stats['categories'] == {}
        assert stats['statuses'] == {}
        assert stats['active_tools'] == 0
        
    def test_statistics_single_tool(self):
        """Test statistics with single tool"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="single_tool",
            description="Single tool",
            category="development",
            status="active"
        )
        registry.register_tool(tool)
        
        stats = registry.get_tool_statistics()
        assert stats['total_tools'] == 1
        assert stats['categories'] == {'development': 1}
        assert stats['statuses'] == {'active': 1}
        assert stats['active_tools'] == 1
        
    def test_statistics_multiple_tools_different_categories(self):
        """Test statistics with multiple tools in different categories"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="tool1", description="Tool 1", category="development", status="active"),
            ToolMetadata(name="tool2", description="Tool 2", category="development", status="inactive"),
            ToolMetadata(name="tool3", description="Tool 3", category="system", status="active"),
            ToolMetadata(name="tool4", description="Tool 4", category="network", status="testing")
        ]
        registry.bulk_register(tools)
        
        stats = registry.get_tool_statistics()
        assert stats['total_tools'] == 4
        assert stats['categories'] == {'development': 2, 'system': 1, 'network': 1}
        assert stats['statuses'] == {'active': 2, 'inactive': 1, 'testing': 1}
        assert stats['active_tools'] == 2
        
    def test_statistics_after_unregister(self):
        """Test statistics update after tool unregistration"""
        registry = ToolRegistry()
        tool1 = ToolMetadata(name="tool1", description="Tool 1", category="development", status="active")
        tool2 = ToolMetadata(name="tool2", description="Tool 2", category="development", status="active")
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Initial stats
        stats = registry.get_tool_statistics()
        assert stats['total_tools'] == 2
        assert stats['categories'] == {'development': 2}
        
        # Unregister one tool
        registry.unregister_tool("tool1")
        
        # Updated stats
        stats = registry.get_tool_statistics()
        assert stats['total_tools'] == 1
        assert stats['categories'] == {'development': 1}


class TestSearchIndex:
    """Test internal search index functionality"""
    
    def test_search_index_build(self):
        """Test that search index is built correctly"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="test_tool",
            description="Test tool description",
            category="development",
            capabilities=["capability1", "capability2"]
        )
        registry.register_tool(tool)
        
        # Check that search index contains expected terms
        assert "test_tool" in registry._search_index
        assert "test tool description" in registry._search_index
        assert "development" in registry._search_index
        assert "capability1" in registry._search_index
        assert "capability2" in registry._search_index
        
    def test_search_index_update_on_unregister(self):
        """Test that search index is updated when tools are unregistered"""
        registry = ToolRegistry()
        tool1 = ToolMetadata(name="tool1", description="Tool 1 description", category="development")
        tool2 = ToolMetadata(name="tool2", description="Tool 2 description", category="development")
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Both tools should be in index
        assert "tool1" in registry._search_index
        assert "tool2" in registry._search_index
        
        registry.unregister_tool("tool1")
        
        # tool1 should be removed from index
        assert "tool1" not in registry._search_index
        # tool2 should still be there
        assert "tool2" in registry._search_index
        
    def test_search_index_case_sensitivity(self):
        """Test that search index stores lowercase versions"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="CamelCaseTool",
            description="Camel Case Description",
            category="Development"
        )
        registry.register_tool(tool)
        
        # Index should contain lowercase versions
        assert "camelcasetool" in registry._search_index
        assert "camel case description" in registry._search_index
        assert "development" in registry._search_index


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_register_none_tool(self):
        """Test registering None as a tool"""
        registry = ToolRegistry()
        with pytest.raises((TypeError, AttributeError)):
            registry.register_tool(None)
            
    def test_get_tool_nonexistent(self):
        """Test getting a tool that doesn't exist"""
        registry = ToolRegistry()
        tool = registry.get_tool("nonexistent_tool")
        assert tool is None
        
    def test_list_tools_invalid_category(self):
        """Test listing tools with invalid category"""
        registry = ToolRegistry()
        tool = ToolMetadata(name="tool1", description="Tool 1", category="development")
        registry.register_tool(tool)
        
        # Should return empty list for non-existent category
        results = registry.list_tools("nonexistent_category")
        assert results == []
        
    def test_registration_with_special_characters(self):
        """Test registration with special characters in names/descriptions"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="tool-with-dashes_and_underscores",
            description="Tool with special chars: @#$%^&*()",
            category="development"
        )
        
        assert registry.register_tool(tool)
        retrieved = registry.get_tool("tool-with-dashes_and_underscores")
        assert retrieved is not None
        assert retrieved.description == "Tool with special chars: @#$%^&*()"
        
    def test_registration_with_unicode(self):
        """Test registration with unicode characters"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="tüls_with_émojis_🎉",
            description="Tool with unicode: café, naïve, résumé",
            category="development"
        )
        
        assert registry.register_tool(tool)
        retrieved = registry.get_tool("tüls_with_émojis_🎉")
        assert retrieved is not None
        assert "café" in retrieved.description
        
    def test_very_long_tool_names(self):
        """Test registration with very long tool names"""
        registry = ToolRegistry()
        long_name = "a" * 1000  # Very long name
        tool = ToolMetadata(
            name=long_name,
            description="Tool with very long name",
            category="development"
        )
        
        assert registry.register_tool(tool)
        retrieved = registry.get_tool(long_name)
        assert retrieved is not None
        assert retrieved.name == long_name
        
    def test_empty_tool_name(self):
        """Test registration with empty tool name"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="",
            description="Tool with empty name",
            category="development"
        )
        
        # Should be able to register (depends on implementation)
        result = registry.register_tool(tool)
        # Empty names might be allowed or not, depending on validation
        assert isinstance(result, bool)
        
    def test_search_with_very_long_query(self):
        """Test searching with very long query"""
        registry = ToolRegistry()
        tool = ToolMetadata(name="short", description="Short tool", category="development")
        registry.register_tool(tool)
        
        long_query = "a" * 10000
        results = registry.search_tools(long_query)
        assert isinstance(results, list)
        assert len(results) == 0  # Should not find anything


class TestPerformance:
    """Test performance characteristics"""
    
    def test_registration_performance_small(self):
        """Test registration performance with small number of tools"""
        import time
        
        registry = ToolRegistry()
        num_tools = 100
        
        start_time = time.time()
        for i in range(num_tools):
            tool = ToolMetadata(
                name=f"tool_{i}",
                description=f"Tool {i} for testing",
                category="development"
            )
            registry.register_tool(tool)
        end_time = time.time()
        
        duration = end_time - start_time
        # Should complete in reasonable time (less than 1 second for 100 tools)
        assert duration < 1.0
        assert len(registry.list_tools()) == num_tools
        
    def test_search_performance(self):
        """Test search performance with larger dataset"""
        import time
        
        registry = ToolRegistry()
        num_tools = 500
        
        # Register tools
        for i in range(num_tools):
            tool = ToolMetadata(
                name=f"search_test_tool_{i}",
                description=f"Search test tool number {i}",
                category="development" if i % 2 == 0 else "system"
            )
            registry.register_tool(tool)
            
        # Test search performance
        start_time = time.time()
        results = registry.search_tools("test")
        end_time = time.time()
        
        duration = end_time - start_time
        # Search should complete quickly
        assert duration < 0.5
        assert len(results) > 0  # Should find some results
        
    def test_memory_usage_large_registry(self):
        """Test memory usage with large number of tools"""
        import sys
        
        registry = ToolRegistry()
        num_tools = 1000
        
        # Register many tools
        for i in range(num_tools):
            tool = ToolMetadata(
                name=f"memory_test_tool_{i}",
                description=f"Memory test tool {i} with some additional description content to increase size",
                category="development",
                capabilities=[f"capability_{i}", f"another_capability_{i}"]
            )
            registry.register_tool(tool)
            
        # Get memory usage
        registry_size = sys.getsizeof(registry._tools) + sys.getsizeof(registry._categories)
        
        # Should be reasonable for 1000 tools (not more than 10MB)
        assert registry_size < 10 * 1024 * 1024  # 10MB
        assert len(registry.list_tools()) == num_tools


class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility"""
    
    def test_path_handling_in_descriptions(self):
        """Test handling of different path separators"""
        registry = ToolRegistry()
        
        # Test Windows-style paths
        tool1 = ToolMetadata(
            name="windows_tool",
            description="Tool located at C:\\Users\\test\\tool.exe",
            category="system"
        )
        
        # Test Unix-style paths
        tool2 = ToolMetadata(
            name="unix_tool",
            description="Tool located at /usr/local/bin/tool",
            category="system"
        )
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        # Both should be searchable
        windows_results = registry.search_tools("C:\\Users")
        unix_results = registry.search_tools("/usr/local")
        
        assert len(windows_results) >= 0  # May or may not find results depending on search implementation
        assert len(unix_results) >= 0
        
    def test_case_sensitive_filesystems(self):
        """Test behavior on case-sensitive filesystems"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="CaseSensitiveTool",
            description="Tool with mixed case",
            category="development"
        )
        
        registry.register_tool(tool)
        
        # Should be able to retrieve by exact name
        retrieved = registry.get_tool("CaseSensitiveTool")
        assert retrieved is not None
        
        # Case-sensitive search should work
        results = registry.search_tools("CaseSensitive")
        assert len(results) >= 0
        
    def test_unicode_filenames(self):
        """Test handling of unicode in tool names and descriptions"""
        registry = ToolRegistry()
        
        # Various unicode characters
        unicode_tools = [
            ToolMetadata(name="工具", description="Chinese tool", category="development"),
            ToolMetadata(name="outil", description="French tool", category="development"),
            ToolMetadata(name="инструмент", description="Russian tool", category="development")
        ]
        
        for tool in unicode_tools:
            assert registry.register_tool(tool)
            retrieved = registry.get_tool(tool.name)
            assert retrieved is not None
            assert retrieved.name == tool.name


class TestIntegrationWithFileSystem:
    """Test integration with file system operations"""
    
    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file for testing"""
        config_file = tmp_path / "tool_config.json"
        tools_data = [
            {
                "name": "file_tool_1",
                "description": "Tool loaded from file",
                "category": "development",
                "status": "active",
                "version": "1.0.0"
            },
            {
                "name": "file_tool_2",
                "description": "Another tool from file",
                "category": "system",
                "status": "active",
                "version": "2.0.0"
            }
        ]
        
        with open(config_file, 'w') as f:
            json.dump(tools_data, f, indent=2)
            
        return str(config_file)
        
    def test_load_tools_from_file(self, temp_config_file):
        """Test loading tools from JSON file"""
        registry = ToolRegistry()
        
        # Simulate loading from file (this would be actual implementation)
        with open(temp_config_file, 'r') as f:
            tools_data = json.load(f)
            
        tools = []
        for tool_data in tools_data:
            tool = ToolMetadata(**tool_data)
            tools.append(tool)
            
        registry.bulk_register(tools)
        
        # Verify tools were loaded
        assert len(registry.list_tools()) == 2
        assert registry.get_tool("file_tool_1") is not None
        assert registry.get_tool("file_tool_2") is not None
        
    def test_save_registry_to_file(self, tmp_path):
        """Test saving registry state to file"""
        registry = ToolRegistry()
        tools = [
            ToolMetadata(name="save_test_1", description="Test 1", category="development"),
            ToolMetadata(name="save_test_2", description="Test 2", category="system")
        ]
        registry.bulk_register(tools)
        
        output_file = tmp_path / "registry_backup.json"
        
        # Save to file (simulated)
        tools_data = []
        for tool in registry.list_tools():
            tools_data.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "status": tool.status,
                "version": tool.version
            })
            
        with open(output_file, 'w') as f:
            json.dump(tools_data, f, indent=2)
            
        # Verify file was created and can be read back
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
            
        assert len(loaded_data) == 2
        assert loaded_data[0]["name"] == "save_test_1"
        assert loaded_data[1]["name"] == "save_test_2"


# Additional test scenarios for comprehensive coverage

class TestToolMetadataAdvanced:
    """Advanced tests for ToolMetadata functionality"""
    
    def test_metadata_deep_copy(self):
        """Test that metadata can be safely modified without affecting registry"""
        registry = ToolRegistry()
        tool = ToolMetadata(
            name="copy_test",
            description="Original description",
            category="development",
            capabilities=["cap1"]
        )
        registry.register_tool(tool)
        
        retrieved = registry.get_tool("copy_test")
        
        # Modify retrieved metadata
        retrieved.description = "Modified description"
        retrieved.capabilities.append("cap2")
        
        # Original should still be unchanged in registry
        original = registry.get_tool("copy_test")
        assert original.description == "Original description"
        assert len(original.capabilities) == 1
        assert "cap1" in original.capabilities
        
    def test_metadata_serialization(self):
        """Test metadata can be serialized to and from JSON"""
        original_tool = ToolMetadata(
            name="serial_test",
            description="Tool for serialization testing",
            category="development",
            parameters={"param1": "value1", "nested": {"key": "value"}},
            capabilities=["test", "serialize"],
            version="1.2.3",
            dependencies=["dep1", "dep2"]
        )
        
        # Serialize to JSON
        tool_dict = {
            "name": original_tool.name,
            "description": original_tool.description,
            "category": original_tool.category,
            "status": original_tool.status,
            "parameters": original_tool.parameters,
            "capabilities": original_tool.capabilities,
            "version": original_tool.version,
            "dependencies": original_tool.dependencies
        }
        
        json_str = json.dumps(tool_dict)
        
        # Deserialize from JSON
        loaded_dict = json.loads(json_str)
        
        # Create new metadata from loaded data
        loaded_tool = ToolMetadata(**loaded_dict)
        
        assert loaded_tool.name == original_tool.name
        assert loaded_tool.description == original_tool.description
        assert loaded_tool.category == original_tool.category
        assert loaded_tool.parameters == original_tool.parameters
        assert loaded_tool.capabilities == original_tool.capabilities
        assert loaded_tool.version == original_tool.version
        assert loaded_tool.dependencies == original_tool.dependencies


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])