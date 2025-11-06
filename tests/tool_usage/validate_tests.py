#!/usr/bin/env python3
"""
Test Validation Script for Tool Usage Module

This script validates the test structure and completeness of the tool_usage module test suite.
It performs checks to ensure all test files exist, have proper structure, and follow conventions.

Features:
- Validates test directory structure
- Checks test file naming conventions
- Verifies test class and method existence
- Validates docstring presence
- Ensures test coverage requirements
- Reports missing or incomplete test components

Usage:
    python validate_tests.py [--verbose] [--fix] [--coverage]
    
    --verbose: Show detailed validation output
    --fix: Automatically fix minor issues
    --coverage: Generate test coverage analysis
"""

import ast
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ValidationLevel(Enum):
    """Validation levels for test checks."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    file_path: str
    check_type: str
    status: str
    message: str
    line_number: int = 0
    severity: str = "info"


@dataclass
class TestFileInfo:
    """Information about a test file."""
    file_path: str
    module_name: str
    has_test_class: bool
    test_methods: List[str]
    docstring: str
    imports: List[str]
    coverage_score: float = 0.0


class TestValidator:
    """Main test validation class."""
    
    def __init__(self, root_path: str = "tests/tool_usage", validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.root_path = Path(root_path)
        self.validation_level = validation_level
        self.results: List[ValidationResult] = []
        self.test_files: List[TestFileInfo] = []
        self.required_test_patterns = {
            "unit": r"test_.*\.py$",
            "integration": r"test_.*_integration\.py$",
            "security": r"test_.*_security\.py$",
            "performance": r"test_.*_performance\.py$",
            "compatibility": r"test_.*_compatibility\.py$"
        }
        
    def validate_all(self, verbose: bool = False) -> bool:
        """Run all validation checks."""
        print(f"🔍 Validating test suite at {self.root_path}")
        print(f"📋 Validation level: {self.validation_level.value}")
        print("-" * 60)
        
        # Check directory structure
        self._validate_directory_structure()
        
        # Check test file naming
        self._validate_file_naming()
        
        # Analyze test files
        self._analyze_test_files()
        
        # Check coverage requirements
        self._check_coverage_requirements()
        
        # Check documentation
        self._check_documentation()
        
        # Generate summary
        return self._generate_summary(verbose)
    
    def _validate_directory_structure(self):
        """Validate the test directory structure."""
        expected_dirs = [
            "unit",
            "integration", 
            "security",
            "performance",
            "compatibility",
            "manual_tests",
            "test_helpers"
        ]
        
        for dir_name in expected_dirs:
            dir_path = self.root_path / dir_name
            if not dir_path.exists():
                self._add_result(
                    str(dir_path), "directory_structure", "error",
                    f"Missing required directory: {dir_name}"
                )
            else:
                self._add_result(
                    str(dir_path), "directory_structure", "pass",
                    f"Directory exists: {dir_name}"
                )
    
    def _validate_file_naming(self):
        """Validate test file naming conventions."""
        test_files = list(self.root_path.rglob("test_*.py"))
        
        for file_path in test_files:
            relative_path = str(file_path.relative_to(self.root_path))
            
            # Check naming patterns
            matches_pattern = any(
                re.match(pattern, file_path.name) 
                for pattern in self.required_test_patterns.values()
            )
            
            if not matches_pattern and self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                self._add_result(
                    relative_path, "naming", "warning",
                    f"File doesn't match expected naming pattern"
                )
            else:
                self._add_result(
                    relative_path, "naming", "pass",
                    f"File naming convention follows standard"
                )
    
    def _analyze_test_files(self):
        """Analyze individual test files for structure."""
        test_files = list(self.root_path.rglob("test_*.py"))
        
        for file_path in test_files:
            try:
                relative_path = str(file_path.relative_to(self.root_path))
                file_info = self._parse_test_file(file_path)
                self.test_files.append(file_info)
                
                # Validate file content
                self._validate_file_content(file_info)
                
            except Exception as e:
                self._add_result(
                    str(file_path), "parsing", "error",
                    f"Failed to parse file: {str(e)}"
                )
    
    def _parse_test_file(self, file_path: Path) -> TestFileInfo:
        """Parse a test file and extract information."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        module_name = file_path.stem
        test_methods = []
        test_classes = []
        docstring = ast.get_docstring(tree) or ""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                test_classes.append(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                        test_methods.append(item.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return TestFileInfo(
            file_path=str(file_path),
            module_name=module_name,
            has_test_class=len(test_classes) > 0,
            test_methods=test_methods,
            docstring=docstring,
            imports=imports
        )
    
    def _validate_file_content(self, file_info: TestFileInfo):
        """Validate the content of a test file."""
        # Check for test classes
        if not file_info.has_test_class:
            self._add_result(
                file_info.file_path, "structure", "warning",
                "No test class found (expected class starting with 'Test')"
            )
        
        # Check for test methods
        if len(file_info.test_methods) == 0:
            self._add_result(
                file_info.file_path, "structure", "error",
                "No test methods found (expected methods starting with 'test_')"
            )
        
        # Check for module docstring
        if not file_info.docstring and self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
            self._add_result(
                file_info.file_path, "documentation", "warning",
                "Missing module docstring"
            )
        
        # Check for required imports
        required_imports = ["import unittest", "import pytest", "import mock"]
        has_required_import = any(imp in file_info.imports for imp in required_imports)
        
        if not has_required_import:
            self._add_result(
                file_info.file_path, "structure", "warning",
                "Missing common test framework imports"
            )
    
    def _check_coverage_requirements(self):
        """Check test coverage requirements."""
        if self.validation_level == ValidationLevel.BASIC:
            return
        
        # Calculate coverage metrics
        total_files = len(self.test_files)
        files_with_classes = len([f for f in self.test_files if f.has_test_class])
        files_with_methods = len([f for f in self.test_files if len(f.test_methods) > 0])
        
        if total_files > 0:
            class_coverage = (files_with_classes / total_files) * 100
            method_coverage = (files_with_methods / total_files) * 100
            
            if class_coverage < 80:
                self._add_result(
                    "coverage", "coverage", "warning",
                    f"Test class coverage: {class_coverage:.1f}% (expected >= 80%)"
                )
            
            if method_coverage < 90:
                self._add_result(
                    "coverage", "coverage", "warning", 
                    f"Test method coverage: {method_coverage:.1f}% (expected >= 90%)"
                )
    
    def _check_documentation(self):
        """Check documentation requirements."""
        if self.validation_level == ValidationLevel.BASIC:
            return
        
        # Check for README files
        readme_files = list(self.root_path.rglob("README.md"))
        if not readme_files:
            self._add_result(
                str(self.root_path), "documentation", "warning",
                "Missing README.md in test directory"
            )
        
        # Check for test documentation files
        doc_files = list(self.root_path.rglob("*.md"))
        if len(doc_files) < 3 and self.validation_level == ValidationLevel.COMPREHENSIVE:
            self._add_result(
                "documentation", "documentation", "info",
                f"Limited documentation files found ({len(doc_files)} files)"
            )
    
    def _generate_summary(self, verbose: bool = False) -> bool:
        """Generate validation summary."""
        errors = [r for r in self.results if r.status == "error"]
        warnings = [r for r in self.results if r.status == "warning"]
        passes = [r for r in self.results if r.status == "pass"]
        
        print(f"\n📊 Validation Summary:")
        print(f"   ✅ Passed: {len(passes)}")
        print(f"   ⚠️  Warnings: {len(warnings)}")
        print(f"   ❌ Errors: {len(errors)}")
        print(f"   📁 Test files analyzed: {len(self.test_files)}")
        
        if verbose:
            print(f"\n📝 Detailed Results:")
            for result in self.results:
                icon = {"error": "❌", "warning": "⚠️", "pass": "✅"}[result.status]
                print(f"   {icon} {result.file_path}: {result.message}")
        
        # Determine overall status
        if errors:
            print(f"\n❌ Validation FAILED - {len(errors)} errors found")
            return False
        elif warnings:
            print(f"\n⚠️  Validation PASSED with warnings - {len(warnings)} warnings found")
            return True
        else:
            print(f"\n✅ Validation PASSED - All checks successful")
            return True
    
    def _add_result(self, file_path: str, check_type: str, status: str, message: str, line_number: int = 0):
        """Add a validation result."""
        result = ValidationResult(
            file_path=file_path,
            check_type=check_type,
            status=status,
            message=message,
            line_number=line_number
        )
        self.results.append(result)
    
    def generate_report(self, output_file: str = "test_validation_report.md"):
        """Generate a detailed validation report."""
        report_lines = [
            "# Test Validation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nValidation Level: {self.validation_level.value}",
            "",
            "## Summary",
            f"- Total files analyzed: {len(self.test_files)}",
            f"- Validation results: {len(self.results)}",
            f"- Errors: {len([r for r in self.results if r.status == 'error'])}",
            f"- Warnings: {len([r for r in self.results if r.status == 'warning'])}",
            f"- Passed: {len([r for r in self.results if r.status == 'pass'])}",
            "",
            "## Test Files",
        ]
        
        for file_info in self.test_files:
            report_lines.extend([
                f"\n### {file_info.file_path}",
                f"- Module: {file_info.module_name}",
                f"- Has test class: {file_info.has_test_class}",
                f"- Test methods: {len(file_info.test_methods)}",
                f"- Has docstring: {bool(file_info.docstring)}",
                f"- Imports: {len(file_info.imports)}",
                ""
            ])
        
        report_lines.extend([
            "## Validation Results",
            ""
        ])
        
        for result in self.results:
            icon = {"error": "❌", "warning": "⚠️", "pass": "✅"}[result.status]
            report_lines.extend([
                f"### {icon} {result.file_path}",
                f"**Check:** {result.check_type}",
                f"**Status:** {result.status}",
                f"**Message:** {result.message}",
                f"**Line:** {result.line_number}",
                ""
            ])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Validation report generated: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate tool_usage test suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--level", choices=["basic", "standard", "comprehensive"], 
                       default="standard", help="Validation level")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues automatically")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage analysis")
    parser.add_argument("--report", type=str, help="Generate detailed report file")
    parser.add_argument("--path", type=str, default="tests/tool_usage", 
                       help="Path to test directory")
    
    args = parser.parse_args()
    
    # Set validation level
    validation_level = ValidationLevel(args.level)
    
    # Create validator
    validator = TestValidator(args.path, validation_level)
    
    # Run validation
    success = validator.validate_all(args.verbose)
    
    # Generate report if requested
    if args.report:
        validator.generate_report(args.report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        from datetime import datetime
    except ImportError:
        datetime = None
    
    main()