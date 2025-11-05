"""
Mini-BIAI-1: Brain-Inspired Modular AI Framework

A comprehensive brain-inspired AI framework featuring neuromorphic computing,
spiking neural networks, and advanced memory systems.

Author: Mini-BIAI-1 Team
License: MIT
Version: 0.3.0
"""

__version__ = "0.3.0"
__author__ = "Mini-BIAI-1 Team"
__license__ = "MIT"
__description__ = "Brain-Inspired Modular AI Framework"

# Import main classes for easy access
try:
    from .coordinator import Coordinator
    from .memory import MemorySystem
    from .optimization import Optimizer
except ImportError:
    # Placeholder imports for development
    pass

# Package metadata
PACKAGE_INFO = {
    "version": __version__,
    "author": __author__,
    "license": __license__,
    "description": __description__,
}

def get_version():
    """Get the current version of Mini-BIAI-1."""
    return __version__

def get_package_info():
    """Get package metadata information."""
    return PACKAGE_INFO