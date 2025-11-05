"""
Comprehensive Performance Optimization Suite

This module provides industrial-grade performance optimization tools for LLM inference
including 4-bit quantization, model pruning, TensorRT compilation, memory optimization,
and comprehensive benchmarking tools.

Target: ≥30 tok/s on RTX 4090 with maintained 5-15% spike rate and biological plausibility.

Author: mini-biai-1 Team
License: MIT
"""

from .quantization.awq_quantizer import AWQQuantizer, QuantizationConfig
from .quantization.gptq_quantizer import GPTQQuantizer
from .pruning.model_pruner import ModelPruner, PruningConfig
from .tensorrt.optimizer import TensorRTOptimizer, TensorRTConfig
from .memory.optimizer import MemoryOptimizer, MemoryConfig
from .energy.monitor import EnergyMonitor, EnergyConfig
from .benchmarking.metrics import (
    TokenThroughputBenchmark, 
    TTFTBenchmark, 
    MemoryBenchmark,
    EnergyBenchmark,
    BiologicalPlausibilityValidator
)
from .framework import OptimizationFramework, OptimizationResult

__version__ = "1.0.0"
__all__ = [
    "AWQQuantizer", "GPTQQuantizer", "QuantizationConfig",
    "ModelPruner", "PruningConfig", 
    "TensorRTOptimizer", "TensorRTConfig",
    "MemoryOptimizer", "MemoryConfig",
    "EnergyMonitor", "EnergyConfig",
    "TokenThroughputBenchmark", "TTFTBenchmark", "MemoryBenchmark", 
    "EnergyBenchmark", "BiologicalPlausibilityValidator",
    "OptimizationFramework", "OptimizationResult"
]