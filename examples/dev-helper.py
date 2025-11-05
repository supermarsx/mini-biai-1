#!/usr/bin/env python3
"""
Developer Helper Utilities
==========================

Utility functions and helpers for mini-biai-1 development.
Provides common tools for testing, debugging, and development.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class DevHelper:
    """Developer helper utilities class."""
    
    def __init__(self):
        """Initialize developer helper."""
        self.log = logging.getLogger(__name__)
        
    def check_imports(self, module_names: List[str]) -> Dict[str, bool]:
        """Check if modules can be imported successfully."""
        results = {}
        
        for module_name in module_names:
            try:
                __import__(module_name)
                results[module_name] = True
                print(f"✓ {module_name}: OK")
            except ImportError as e:
                results[module_name] = False
                print(f"✗ {module_name}: FAILED - {e}")
        
        return results
    
    def generate_test_config(self, config_type: str = "quick") -> Dict[str, Any]:
        """Generate test configuration based on type."""
        configs = {
            "quick": {
                "model": {"d_model": 256},
                "memory": {"stm": {"max_tokens": 1024}}
            },
            "standard": {
                "model": {"d_model": 512},
                "memory": {"stm": {"max_tokens": 4096}}
            }
        }
        
        return configs.get(config_type, configs["quick"])
    
    def time_function(self, func, *args, **kwargs):
        """Time a function execution."""
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        self.log.info(f"Function {func.__name__} took {end - start:.4f}s")
        return result
    
    def save_debug_info(self, filepath: str, data: Dict[str, Any]):
        """Save debug information to file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Debug info saved to {filepath}")

def main():
    """Main developer helper function."""
    helper = DevHelper()
    
    print("=== Developer Helper Utilities ===")
    print()
    
    # Check core imports
    print("1. Checking Core Imports")
    print("-" * 30)
    
    core_modules = [
        "numpy",
        "torch",
        "faiss",
        "transformers"
    ]
    
    import_results = helper.check_imports(core_modules)
    print()
    
    # Generate test configuration
    print("2. Test Configuration Generation")
    print("-" * 30)
    
    config = helper.generate_test_config("quick")
    print(f"Generated config: {json.dumps(config, indent=2)}")
    print()
    
    print("=== Helper Ready ===")

if __name__ == "__main__":
    main()