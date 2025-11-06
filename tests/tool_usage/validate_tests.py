#!/usr/bin/env python3
"""
Simple Test Validation Script

This script validates that the test files exist and have basic structure
without requiring complex dependencies.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if filepath.exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False

def check_directory_structure():
    """Check the test directory structure."""
    print("🔍 Checking Test Directory Structure")
    print("=" * 50)
    
    base_dir = Path("tests/tool_usage")
    checks = []
    
    # Check main directory
    checks.append(check_file_exists(base_dir, "Test directory"))
    
    # Check conftest.py
    checks.append(check_file_exists(base_dir / "conftest.py", "conftest.py"))
    
    # Check unit test files
    unit_dir = base_dir / "unit"
    checks.append(check_file_exists(unit_dir, "Unit tests directory"))
    
    unit_files = [
        "test_shell_detector.py",
        "test_command_executor.py", 
        "test_tool_registry.py",
        "test_usage_optimizer.py",
        "test_platform_adapter.py"
    ]
    
    for file in unit_files:
        checks.append(check_file_exists(unit_dir / file, f"Unit test: {file}"))
    
    # Check integration tests
    integration_dir = base_dir / "integration"
    checks.append(check_file_exists(integration_dir, "Integration tests directory"))
    checks.append(check_file_exists(integration_dir / "test_tool_usage_integration.py", "Integration test"))
    
    # Check security tests
    security_dir = base_dir / "security"
    checks.append(check_file_exists(security_dir, "Security tests directory"))
    checks.append(check_file_exists(security_dir / "test_security_safety.py", "Security test"))
    
    # Check compatibility tests
    compatibility_dir = base_dir / "compatibility"
    checks.append(check_file_exists(compatibility_dir, "Compatibility tests directory"))
    checks.append(check_file_exists(compatibility_dir / "test_cross_platform.py", "Compatibility test"))
    
    # Check performance tests
    performance_dir = base_dir / "performance"
    checks.append(check_file_exists(performance_dir, "Performance tests directory"))
    checks.append(check_file_exists(performance_dir / "test_performance_load.py", "Performance test"))
    
    # Check manual tests
    manual_dir = base_dir / "manual_tests"
    checks.append(check_file_exists(manual_dir, "Manual tests directory"))
    checks.append(check_file_exists(manual_dir / "run_manual_tests.py", "Manual test runner"))
    
    # Check helper files
    checks.append(check_file_exists(base_dir / "quick_test_runner.py", "Quick test runner"))
    checks.append(check_file_exists(base_dir / "README.md", "Test documentation"))
    
    return all(checks)

def check_file_content(filepath, description):
    """Check basic file content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = len(content.split('\n'))
        size = len(content)
        
        print(f"📄 {description}: {lines} lines, {size} characters")
        return True
    except Exception as e:
        print(f"❌ Error reading {description}: {e}")
        return False

def validate_test_files():
    """Validate that test files have proper content."""
    print("\n🔍 Validating Test File Content")
    print("=" * 50)
    
    base_dir = Path("tests/tool_usage")
    checks = []
    
    # Validate conftest.py
    conftest_file = base_dir / "conftest.py"
    if conftest_file.exists():
        checks.append(check_file_content(conftest_file, "conftest.py"))
        
        # Check for key components
        with open(conftest_file, 'r') as f:
            content = f.read()
        
        if "pytest.fixture" in content:
            print("✅ conftest.py contains pytest fixtures")
        else:
            print("⚠️ conftest.py may be missing pytest fixtures")
        
        if "@pytest.fixture" in content or "def " in content and "fixture" in content:
            print("✅ conftest.py appears to have fixture definitions")
        else:
            print("⚠️ conftest.py may be missing fixture definitions")
    
    # Validate unit tests
    unit_dir = base_dir / "unit"
    if unit_dir.exists():
        unit_files = [
            "test_shell_detector.py",
            "test_command_executor.py",
            "test_tool_registry.py", 
            "test_usage_optimizer.py",
            "test_platform_adapter.py"
        ]
        
        for file in unit_files:
            filepath = unit_dir / file
            if filepath.exists():
                checks.append(check_file_content(filepath, f"Unit test: {file}"))
                
                # Check for pytest patterns
                with open(filepath, 'r') as f:
                    content = f.read()
                
                if "def test_" in content:
                    print(f"✅ {file} contains test functions")
                else:
                    print(f"⚠️ {file} may be missing test functions")
                
                if "class Test" in content:
                    print(f"✅ {file} contains test classes")
                else:
                    print(f"⚠️ {file} may be missing test classes")
    
    # Validate other test categories
    test_categories = [
        ("integration", "test_tool_usage_integration.py"),
        ("security", "test_security_safety.py"),
        ("compatibility", "test_cross_platform.py"),
        ("performance", "test_performance_load.py")
    ]
    
    for category, filename in test_categories:
        filepath = base_dir / category / filename
        if filepath.exists():
            checks.append(check_file_content(filepath, f"{category.title()} test"))
    
    # Validate manual tests
    manual_file = base_dir / "manual_tests" / "run_manual_tests.py"
    if manual_file.exists():
        checks.append(check_file_content(manual_file, "Manual test runner"))
    
    # Validate helper scripts
    helper_files = [
        ("quick_test_runner.py", "Quick test runner"),
        ("README.md", "Documentation")
    ]
    
    for filename, description in helper_files:
        filepath = base_dir / filename
        if filepath.exists():
            checks.append(check_file_content(filepath, description))
    
    return all(checks)

def estimate_line_counts():
    """Estimate total line counts across test files."""
    print("\n📊 Estimating Test Coverage")
    print("=" * 50)
    
    base_dir = Path("tests/tool_usage")
    total_lines = 0
    file_counts = {}
    
    test_files = [
        "conftest.py",
        "unit/test_shell_detector.py",
        "unit/test_command_executor.py",
        "unit/test_tool_registry.py",
        "unit/test_usage_optimizer.py",
        "unit/test_platform_adapter.py",
        "integration/test_tool_usage_integration.py",
        "security/test_security_safety.py",
        "compatibility/test_cross_platform.py",
        "performance/test_performance_load.py",
        "manual_tests/run_manual_tests.py",
        "quick_test_runner.py",
        "README.md"
    ]
    
    for test_file in test_files:
        filepath = base_dir / test_file
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                file_counts[test_file] = lines
                total_lines += lines
                print(f"📄 {test_file}: {lines} lines")
            except Exception as e:
                print(f"❌ Error reading {test_file}: {e}")
        else:
            print(f"❌ File not found: {test_file}")
    
    print(f"\n📈 Total estimated lines: {total_lines}")
    print(f"📁 Number of test files: {len(file_counts)}")
    
    # Estimate coverage based on lines (rough approximation)
    estimated_coverage_potential = min(95, total_lines // 10)  # Rough estimate
    print(f"🎯 Estimated coverage potential: ~{estimated_coverage_potential}%")
    
    return total_lines, file_counts

def run_syntax_check():
    """Check syntax of Python test files."""
    print("\n🔍 Running Syntax Check")
    print("=" * 50)
    
    base_dir = Path("tests/tool_usage")
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    syntax_ok = True
    for filepath in python_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                compile(f.read(), str(filepath), 'exec')
            print(f"✅ {filepath.name}: Syntax OK")
        except SyntaxError as e:
            print(f"❌ {filepath.name}: Syntax Error - {e}")
            syntax_ok = False
        except Exception as e:
            print(f"⚠️ {filepath.name}: Other error - {e}")
    
    return syntax_ok

def main():
    """Main validation function."""
    print("🚀 Tool Usage Test Suite Validation")
    print("=" * 60)
    
    # Change to project root if not already there
    if not Path("tests/tool_usage").exists():
        print("❌ tests/tool_usage directory not found.")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Run all checks
    checks_passed = 0
    total_checks = 4
    
    if check_directory_structure():
        checks_passed += 1
    
    if validate_test_files():
        checks_passed += 1
    
    if run_syntax_check():
        checks_passed += 1
    
    total_lines, file_counts = estimate_line_counts()
    checks_passed += 1  # This check always "passes"
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Checks passed: {checks_passed}/{total_checks}")
    
    if checks_passed == total_checks:
        print("🎉 All validation checks passed!")
        print("\n📋 Next steps:")
        print("1. Run: python tests/tool_usage/quick_test_runner.py")
        print("2. Run: python -m pytest tests/tool_usage/ -v")
        print("3. Run: python tests/tool_usage/manual_tests/run_manual_tests.py")
        return True
    else:
        print("⚠️ Some validation checks failed.")
        print("Please review the errors above and fix any issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
