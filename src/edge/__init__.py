from typing import Dict, List, Optional, Union, Any
import os
import sys
import logging
import warnings
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

# Enable all optimizations by default
__all__ = [
    'ModelQuantizer',
    'EdgeOptimizer', 
    'EdgeAccelerator',
    'ONNXExporter',
    'TensorRTOptimizer',
    'WebAssemblyDeployer',
    'ProgressiveLoader',
    'BatteryOptimizer',
    'EdgeOptimizationDemo',
    'get_edge_optimization_config',
    'detect_edge_device',
    'optimize_for_platform',
    'benchmark_edge_performance'
]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import all components
try:
    from .quantization import ModelQuantizer
except ImportError as e:
    logger.warning(f"Could not import ModelQuantizer: {e}")
    ModelQuantizer = None

try:
    from .optimization import EdgeOptimizer
except ImportError as e:
    logger.warning(f"Could not import EdgeOptimizer: {e}")
    EdgeOptimizer = None

try:
    from .accelerator import EdgeAccelerator
except ImportError as e:
    logger.warning(f"Could not import EdgeAccelerator: {e}")
    EdgeAccelerator = None

try:
    from .onnx_export import ONNXExporter
except ImportError as e:
    logger.warning(f"Could not import ONNXExporter: {e}")
    ONNXExporter = None

try:
    from .tensorrt import TensorRTOptimizer
except ImportError as e:
    logger.warning(f"Could not import TensorRTOptimizer: {e}")
    TensorRTOptimizer = None

try:
    from .webassembly import WebAssemblyDeployer
except ImportError as e:
    logger.warning(f"Could not import WebAssemblyDeployer: {e}")
    WebAssemblyDeployer = None

try:
    from .progressive import ProgressiveLoader
except ImportError as e:
    logger.warning(f"Could not import ProgressiveLoader: {e}")
    ProgressiveLoader = None

try:
    from .battery import BatteryOptimizer
except ImportError as e:
    logger.warning(f"Could not import BatteryOptimizer: {e}")
    BatteryOptimizer = None

try:
    from .edge_optimization_demo import EdgeOptimizationDemo
except ImportError as e:
    logger.warning(f"Could not import EdgeOptimizationDemo: {e}")
    EdgeOptimizationDemo = None


def get_edge_optimization_config() -> Dict[str, Any]:
    """
    Get default configuration for edge optimization.
    
    Returns:
        Dict containing default optimization settings
    """
    return {
        'quantization': {
            'enabled': True,
            'precision': 'int8',
            'calibration_samples': 100,
            'preserve_accuracy': True
        },
        'optimization': {
            'pruning': {
                'enabled': True,
                'method': 'magnitude',
                'sparsity': 0.3
            },
            'memory': {
                'enabled': True,
                'optimize_batch_size': True,
                'gradient_checkpointing': False
            }
        },
        'acceleration': {
            'hardware_acceleration': True,
            'preferred_backends': ['tensorrt', 'nnapi', 'coreml', 'openvino'],
            'fallback_to_cpu': True
        },
        'power': {
            'battery_aware': True,
            'thermal_throttling': True,
            'performance_scaling': True
        },
        'progressive': {
            'enabled': True,
            'chunk_size': 'auto',
            'priority_based': True
        }
    }


def detect_edge_device() -> Dict[str, Any]:
    """
    Detect edge device capabilities and constraints.
    
    Returns:
        Dict containing device information
    """
    import platform
    import psutil
    
    device_info = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }
    
    # System resources
    try:
        device_info.update({
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        })
    except:
        logger.warning("Could not detect system resources")
    
    # Platform-specific detection
    if platform.system() == 'Linux':
        # Check for ARM architecture
        if 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower():
            device_info['edge_type'] = 'mobile_arm'
        else:
            device_info['edge_type'] = 'linux_x86'
    elif platform.system() == 'Darwin':
        device_info['edge_type'] = 'apple_silicon' if 'arm' in platform.machine().lower() else 'macos_x86'
    elif platform.system() == 'Windows':
        device_info['edge_type'] = 'windows_x86'
    else:
        device_info['edge_type'] = 'unknown'
    
    # Hardware acceleration capabilities
    device_info['acceleration_backends'] = []
    
    # Check for TensorRT
    try:
        import tensorrt
        device_info['acceleration_backends'].append('tensorrt')
    except ImportError:
        pass
    
    # Check for ONNX Runtime
    try:
        import onnxruntime as ort
        device_info['acceleration_backends'].append('onnxruntime')
        # Check available providers
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            device_info['acceleration_backends'].append('cuda')
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            device_info['acceleration_backends'].append('tensorrt_ort')
    except ImportError:
        pass
    
    return device_info


def optimize_for_platform(model: Any, platform: str = 'auto') -> Any:
    """
    Apply platform-specific optimizations to a model.
    
    Args:
        model: Model to optimize
        platform: Target platform ('auto', 'mobile', 'web', 'embedded')
    
    Returns:
        Optimized model
    """
    if platform == 'auto':
        device_info = detect_edge_device()
        platform = device_info.get('edge_type', 'unknown')
    
    logger.info(f"Optimizing model for platform: {platform}")
    
    # Apply quantization
    if ModelQuantizer is not None:
        quantizer = ModelQuantizer()
        if platform in ['mobile_arm', 'apple_silicon']:
            model = quantizer.quantize_model(model, precision='int8')
    
    # Apply optimization
    if EdgeOptimizer is not None:
        optimizer = EdgeOptimizer()
        if platform in ['mobile_arm', 'embedded']:
            model = optimizer.prune_model(model, sparsity=0.2)
    
    # Apply acceleration
    if EdgeAccelerator is not None:
        accelerator = EdgeAccelerator()
        if platform == 'web':
            # Web-specific optimizations
            pass
        elif platform in ['mobile_arm', 'apple_silicon']:
            # Mobile-specific acceleration
            pass
    
    return model


def benchmark_edge_performance(model: Any, test_input: Any = None) -> Dict[str, float]:
    """
    Benchmark model performance on edge devices.
    
    Args:
        model: Model to benchmark
        test_input: Test input data (optional)
    
    Returns:
        Dict containing performance metrics
    """
    import time
    import numpy as np
    
    logger.info("Starting edge performance benchmark...")
    
    # Device detection
    device_info = detect_edge_device()
    logger.info(f"Device: {device_info['edge_type']}")
    logger.info(f"Available backends: {device_info['acceleration_backends']}")
    
    # Generate test input if not provided
    if test_input is None:
        if hasattr(model, 'input_shape'):
            shape = model.input_shape
        else:
            shape = (1, 3, 224, 224)  # Default for vision models
        test_input = np.random.randn(*shape).astype(np.float32)
    
    results = {}
    
    # CPU baseline
    try:
        start_time = time.time()
        for _ in range(10):
            _ = model(test_input)
        cpu_time = (time.time() - start_time) / 10
        results['cpu_latency_ms'] = cpu_time * 1000
    except Exception as e:
        logger.warning(f"CPU benchmark failed: {e}")
        results['cpu_latency_ms'] = None
    
    # Hardware acceleration benchmarks
    if 'tensorrt' in device_info['acceleration_backends'] and TensorRTOptimizer is not None:
        try:
            trt_optimizer = TensorRTOptimizer()
            trt_model = trt_optimizer.optimize_model(model)
            
            start_time = time.time()
            for _ in range(10):
                _ = trt_model(test_input)
            trt_time = (time.time() - start_time) / 10
            results['tensorrt_latency_ms'] = trt_time * 1000
            results['speedup_vs_cpu'] = results['cpu_latency_ms'] / results['tensorrt_latency_ms'] if results['cpu_latency_ms'] else None
        except Exception as e:
            logger.warning(f"TensorRT benchmark failed: {e}")
    
    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        results['memory_usage_mb'] = memory_info.rss / 1024 / 1024
    except:
        results['memory_usage_mb'] = None
    
    return results


# Version information
__version__ = "1.0.0"
__author__ = "Edge Optimization Team"
__email__ = "edge-optimization@company.com"
__description__ = "Comprehensive edge deployment optimization library"

# Feature flags
FEATURES = {
    'quantization': ModelQuantizer is not None,
    'optimization': EdgeOptimizer is not None,
    'acceleration': EdgeAccelerator is not None,
    'onnx_export': ONNXExporter is not None,
    'tensorrt': TensorRTOptimizer is not None,
    'webassembly': WebAssemblyDeployer is not None,
    'progressive': ProgressiveLoader is not None,
    'battery': BatteryOptimizer is not None,
    'demo': EdgeOptimizationDemo is not None
}


def print_feature_status():
    """Print status of all edge optimization features."""
    print("\n=== Edge Optimization Features Status ===")
    for feature, available in FEATURES.items():
        status = "✓" if available else "✗"
        print(f"{status} {feature.replace('_', ' ').title()}")
    print("\n" + "="*45)


if __name__ == "__main__":
    print_feature_status()
    
    # Run quick detection
    device_info = detect_edge_device()
    print(f"\nDetected Device: {device_info['edge_type']}")
    print(f"Available Backends: {', '.join(device_info['acceleration_backends'])}")
    
    # Show configuration
    config = get_edge_optimization_config()
    print(f"\nDefault Configuration:")
    for category, settings in config.items():
        print(f"  {category}: {settings}")