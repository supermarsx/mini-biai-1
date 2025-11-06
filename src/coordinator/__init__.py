"""Coordinator Module - Main entry point and exports for the MiniMax AI coordinator system."""

from .optimized_snn import (
    OptimizedSNNParameters,
    SurrogateGradient,
    SparseTensorManager,
    EventDrivenLIF,
    SpikePatternOptimizer,
    EnergyEfficientSpikingLayer,
    OptimizedSpikingRouter,
    HardwareOptimizer
)
from .spiking_router import (
    LIFNeuron,
    LinearEncoder,
    RoutingHead,
    LIFParameters,
    HardwareCompatibilityChecker
)
from .step3_integration import (
    VisionExpert,
    AffectModulation,
    SymbolicReasoning,
    HierarchicalMemory,
    UnifiedStep3System
)
from .optimization_integration import (
    UnifiedOptimizationSystem,
    PerformanceBenchmarker,
    DeploymentConfig
)
from .memory_efficient_training import (
    MemoryEfficientSNNTrainer,
    GradientCheckpointing,
    MixedPrecisionTrainer,
    SparseGradientComputation
)
from .cuda_kernels import (
    SparseCUDAMatrix,
    CUDASparseDenseMult,
    CUDAEventDrivenUpdate,
    MemoryCoalescer
)
from .benchmark_profiler import (
    PerformanceProfiler,
    LatencyMeasurer,
    MemoryProfiler,
    EnergyProfiler,
    SpikeRateMonitor,
    HardwareUtilizationTracker
)

__all__ = [
    # Core SNN Components
    'OptimizedSNNParameters',
    'SurrogateGradient', 
    'SparseTensorManager',
    'EventDrivenLIF',
    'SpikePatternOptimizer',
    'EnergyEfficientSpikingLayer',
    'OptimizedSpikingRouter',
    'HardwareOptimizer',
    
    # Router Components
    'LIFNeuron',
    'LinearEncoder', 
    'RoutingHead',
    'LIFParameters',
    'HardwareCompatibilityChecker',
    
    # Step 3 Integration
    'VisionExpert',
    'AffectModulation',
    'SymbolicReasoning',
    'HierarchicalMemory',
    'UnifiedStep3System',
    
    # Optimization Integration
    'UnifiedOptimizationSystem',
    'PerformanceBenchmarker',
    'DeploymentConfig',
    
    # Memory Efficient Training
    'MemoryEfficientSNNTrainer',
    'GradientCheckpointing',
    'MixedPrecisionTrainer',
    'SparseGradientComputation',
    
    # CUDA Kernels
    'SparseCUDAMatrix',
    'CUDASparseDenseMult',
    'CUDAEventDrivenUpdate', 
    'MemoryCoalescer',
    
    # Profiling
    'PerformanceProfiler',
    'LatencyMeasurer',
    'MemoryProfiler',
    'EnergyProfiler',
    'SpikeRateMonitor',
    'HardwareUtilizationTracker'
]