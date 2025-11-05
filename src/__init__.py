"""
mini-biai-1 - A Brain-Inspired Computational Model

mini-biai-1 is a comprehensive framework that combines spiking neural networks,
hierarchical memory architecture, and transformer-based language processing
for neuromorphic computing applications.
"""

__version__ = "0.1.0"
__author__ = "mini-biai-1 Team"
__email__ = "team@mini-biai-1.org"

# Import all major modules
from . import coordinator
from . import memory
from . import language
from . import interfaces
from . import inference
from . import training
from . import utils

# Expose key classes
from .coordinator import SpikingRouter
from .memory import ShortTermMemory
from .language import LinearTextProcessor
from .inference import InferencePipeline, create_pipeline
from .training import RoutingTrainer, SyntheticRoutingDataset
from .utils import PerformanceProfiler

__all__ = [
    "coordinator",
    "memory", 
    "language",
    "interfaces",
    "inference",
    "training",
    "utils",
    "SpikingRouter",
    "ShortTermMemory",
    "LinearTextProcessor",
    "InferencePipeline",
    "create_pipeline",
    "RoutingTrainer",
    "SyntheticRoutingDataset",
    "PerformanceProfiler"
]