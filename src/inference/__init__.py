"""
mini-biai-1 - Inference Pipeline Package

This package provides end-to-end inference processing with:
- Tokenization and STM/LTM memory systems
- Spiking neural network-based routing
- Comprehensive logging and performance monitoring
- Modular, replaceable components
"""

from .pipeline import (
    InferencePipeline,
    PipelineConfig,
    ProcessingTrace,
    RetrievalResult,
    RoutingDecision,
    Tokenizer,
    ShortTermMemory,
    LongTermMemory,
    MultiExpertRouterWrapper,
    LanguageModule,
    PerformanceMonitor,
    create_pipeline
)

__version__ = "1.0.0"
__author__ = "mini-biai-1 Team"

__all__ = [
    "InferencePipeline",
    "PipelineConfig", 
    "ProcessingTrace",
    "RetrievalResult",
    "RoutingDecision",
    "Tokenizer",
    "ShortTermMemory",
    "LongTermMemory",
    "MultiExpertRouterWrapper",
    "LanguageModule",
    "PerformanceMonitor",
    "create_pipeline"
]