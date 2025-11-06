"""
Edge Deployment Optimization Module

This module provides comprehensive optimization capabilities for deploying machine learning
models on resource-constrained edge devices including mobile and embedded systems.


Key Features:
- INT8 and FP16 quantization with calibration
- ONNX and TensorRT optimization
- WebAssembly deployment for browsers
- Progressive model loading
- Hardware acceleration (NNAPI, CoreML, Metal)
- Battery and power consumption optimization
- Memory-efficient inference
- Real-time performance on edge devices

Platforms Supported:
- iOS (CoreML, Metal)
- Android (NNAPI, TensorFlow Lite)
- Embedded Linux (TensorRT, ONNX)
- Web (WebAssembly)


Author: MiniMax AI
Version: 1.0.0
"""


import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class EdgePlatform(Enum):
    """"Supported edge platforms."""
    IOS = "ios"
    ANDROID = "android"
    LINUX_EMBEDDED = "linux_embedded"
    WEB_WASM = "web_wasm"
    WINDOWS_IOT = "windows_iot"
    RASPBERRY_PI = "raspberry_pi"



class QuantizationType(Enum):
    """Quantization types for edge deployment."""
    INT8 = "int8"
    FP16 = "fp16"
    DYNAMIC = "dynamic"
    POST_TRAINING = "post_training"



class AccelerationBackend(Enum):
    """Hardware acceleration backends."""
    COREML = "coreml"
    NNAPI = "nnapi"
    METAL = "metal"
    TENSORRT = "tensorrt"
    OPENCL = "opencl"
    CUDA = "cuda"



@dataclass
class EdgeDevice:
    """Edge device specifications."""
    platform: EdgePlatform
    name: str
    cpu: str
    gpu: Optional[str] = None
    memory_gb: float = 2.0
    storage_gb: float = 16.0
    max_power_watts: float = 5.0
    supported_backends: List[AccelerationBackend] = None
    constraints: Dict[str, Any] = None

    def __post_init__(self):
        if self.supported_backends is None:
            self.supported_backends = []
        if self.constraints is None:
            self.constraints = {}



@dataclass
class OptimizationConfig:
    """Configuration for edge optimization."""
    quantization_type: Optional[QuantizationType] = None
    target_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    target_accuracy: Optional[float] = None
    battery_optimization: bool = True
    enable_progressive_loading: bool = True
    compression_ratio: Optional[float] = None
    batch_size: int = 1
    threads: Optional[int] = None



# Initialize core components
try:
    from .quantization import Quantizer
    from .optimization import MobileOptimizer
    from .onnx_export import ONNXExporter
    from .tensorrt import TensorRTOptimizer
    from .webassembly import WebAssemblyDeployer
    from .progressive import ProgressiveLoader
    from .accelerator import EdgeAccelerator
    from .battery import BatteryOptimizer

    __all__ = [
        "Quantizer",
        "MobileOptimizer", 
        "ONNXExporter",
        "TensorRTOptimizer",
        "WebAssemblyDeployer",
        "ProgressiveLoader",
        "EdgeAccelerator",
        "BatteryOptimizer",
        "EdgePlatform",
        "QuantizationType",
        "AccelerationBackend",
        "EdgeDevice",
        "OptimizationConfig"
    ]
    
    logger.info("Edge optimization module initialized successfully")

except ImportError as e:
    logger.warning(f"Some edge optimization components failed to import: {e}")
    __all__ = []



def create_edge_deployer(device: EdgeDevice, config: OptimizationConfig) -> Dict[str, Any]:
    """
    Factory function to create appropriate edge deployer based on device and configuration.
    
    Args:
        device: Target edge device specifications
        config: Optimization configuration
        
    Returns:
        Dictionary containing appropriate deployer instances
    """
    deployers = {}
    
    try:
        # Always create quantizer
        if config.quantization_type:
            deployers['quantizer'] = Quantizer()
        
        # Create platform-specific optimizers
        if device.platform == EdgePlatform.IOS:
            deployers['optimizer'] = MobileOptimizer(backend=AccelerationBackend.COREML)
            deployers['accelerator'] = EdgeAccelerator([AccelerationBackend.COREML, AccelerationBackend.METAL])
        elif device.platform == EdgePlatform.ANDROID:
            deployers['optimizer'] = MobileOptimizer(backend=AccelerationBackend.NNAPI)
            deployers['accelerator'] = EdgeAccelerator([AccelerationBackend.NNAPI, AccelerationBackend.OPENCL])
        elif device.platform == EdgePlatform.LINUX_EMBEDDED:
            deployers['optimizer'] = MobileOptimizer(backend=AccelerationBackend.TENSORRT)
            deployers['tensorrt'] = TensorRTOptimizer()
        elif device.platform == EdgePlatform.WEB_WASM:
            deployers['webassembly'] = WebAssemblyDeployer()
        
        # Always create common components
        deployers['progressive'] = ProgressiveLoader()
        deployers['battery'] = BatteryOptimizer()
        
        if config.quantization_type in [QuantizationType.INT8, QuantizationType.FP16]:
            deployers['onnx'] = ONNXExporter()
            
    except Exception as e:
        logger.error(f"Failed to create edge deployer: {e}")
        raise
    
    return deployers


def get_supported_devices() -> List[EdgeDevice]:
    """
    Get list of supported edge devices with pre-configured specifications.
    
    Returns:
        List of EdgeDevice instances
    """
    return [
        EdgeDevice(
            platform=EdgePlatform.IOS,
            name="iPhone 14 Pro",
            cpu="Apple A16 Bionic",
            gpu="Apple GPU",
            memory_gb=6.0,
            supported_backends=[AccelerationBackend.COREML, AccelerationBackend.METAL]
        ),
        EdgeDevice(
            platform=EdgePlatform.ANDROID,
            name="Samsung Galaxy S23",
            cpu="Snapdragon 8 Gen 2",
            gpu="Adreno 740",
            memory_gb=8.0,
            supported_backends=[AccelerationBackend.NNAPI, AccelerationBackend.OPENCL]
        ),
        EdgeDevice(
            platform=EdgePlatform.LINUX_EMBEDDED,
            name="NVIDIA Jetson Orin",
            cpu="ARM Cortex-A78AE",
            gpu="NVIDIA Ampere",
            memory_gb=8.0,
            supported_backends=[AccelerationBackend.TENSORRT, AccelerationBackend.CUDA]
        ),
        EdgeDevice(
            platform=EdgePlatform.WEB_WASM,
            name="Web Browser",
            cpu="x86_64/ARM64",
            memory_gb=4.0,
            supported_backends=[]
        ),
        EdgeDevice(
            platform=EdgePlatform.RASPBERRY_PI,
            name="Raspberry Pi 4",
            cpu="ARM Cortex-A72",
            memory_gb=4.0,
            supported_backends=[AccelerationBackend.OPENCL]
        )
    ]



def benchmark_edge_deployment(model_path: str, device: EdgeDevice, 
                            config: OptimizationConfig) -> Dict[str, Any]:
    """
    Benchmark edge deployment performance.
    
    Args:
        model_path: Path to the model
        device: Target device
        config: Optimization configuration
        
    Returns:
        Benchmark results dictionary
    """
    logger.info(f"Starting edge deployment benchmark for {device.name}")
    
    # Create deployers
    deployers = create_edge_deployer(device, config)
    
    # Run benchmark
    results = {
        "device": device.name,
        "platform": device.platform.value,
        "config": {
            "quantization": config.quantization_type.value if config.quantization_type else None,
            "target_latency": config.target_latency_ms,
            "max_memory": config.max_memory_mb
        },
        "results": {}
    }
    
    # Benchmark quantization
    if 'quantizer' in deployers:
        try:
            quant_result = deployers['quantizer'].quantize_model(model_path, config.quantization_type)
            results["results"]["quantization"] = {
                "size_reduction": quant_result.get("size_reduction", 0),
                "accuracy_impact": quant_result.get("accuracy_impact", 0),
                "speed_improvement": quant_result.get("speed_improvement", 0)
            }
        except Exception as e:
            results["results"]["quantization"] = {"error": str(e)}
    
    # Benchmark hardware acceleration
    if 'accelerator' in deployers:
        try:
            accel_result = deployers['accelerator'].benchmark_acceleration(model_path, device)
            results["results"]["acceleration"] = accel_result
        except Exception as e:
            results["results"]["acceleration"] = {"error": str(e)}
    
    # Benchmark battery optimization
    if 'battery' in deployers:
        try:
            battery_result = deployers['battery'].optimize_for_battery(model_path)
            results["results"]["battery"] = battery_result
        except Exception as e:
            results["results"]["battery"] = {"error": str(e)}
    
    logger.info("Edge deployment benchmark completed")
    return results