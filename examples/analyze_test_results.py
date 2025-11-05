#!/usr/bin/env python3
"""
Test Results Analysis Tool
==========================

Analyzes test results and generates reports.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def analyze_test_results(results_file: str = "test_results.json"):
    """Analyze test results from file."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print("=== Test Results Analysis ===")
        print(f"Total tests: {len(results)}")
        
        passed = sum(1 for r in results if r.get('status') == 'passed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {passed / len(results) * 100:.1f}%")
        
    except FileNotFoundError:
        print(f"Test results file {results_file} not found")
    except json.JSONDecodeError:
        print(f"Invalid JSON in {results_file}")

if __name__ == "__main__":
    analyze_test_results()