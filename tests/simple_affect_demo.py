#!/usr/bin/env python3
"""
Simple Affect Demo
==================

Demonstrates basic affect detection functionality.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def simple_affect_demo():
    """Run simple affect demonstration."""
    print("=== Simple Affect Demo ===")
    
    # Simulate affect detection
    emotions = ['happy', 'sad', 'neutral', 'excited']
    
    for i, emotion in enumerate(emotions):
        print(f"Detected emotion: {emotion}")
    
    print("✓ Affect detection demo completed")

if __name__ == "__main__":
    simple_affect_demo()