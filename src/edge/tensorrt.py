"""
TensorRT Optimization and Acceleration Module

Provides comprehensive TensorRT optimization for edge deployment including
engine building, precision optimization, and performance tuning for GPU acceleration.

Features:
- TensorRT engine creation and optimization
- FP32, FP16, and INT8 precision support
- Dynamic shape optimization
- Memory optimization
- Multi-GPU support
- Engine serialization and deserialization
- Performance profiling and benchmarking
- Dynamic batching
"""

import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import time

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    trt = None
    TRT_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    CUDA_AVAILABLE = False

# Removed problematic imports for standalone usage
# from . import EdgeDevice, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class TensorRTConfig:
    """Configuration for TensorRT optimization."""
    precision: str = "fp16"  # fp32, fp16, int8
    max_workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 32
    opt_batch_size: int = 1
    max_dyn_batch_size: int = 16
    use_fp16: bool = True
    use_int8: bool = False
    use_strict_types: bool = False
    allow_gpu_fallback: bool = True
    use_fp16: bool = True
    use_int8: bool = False
    calibration_dataset: Optional[str] = None
    enable_profiling: bool = True
    save_engine: bool = True


@dataclass
class EngineInfo:
    """TensorRT engine information."""
    max_batch_size: int
    max_input_size: int
    num_bindings: int
    num_layers: int
    num_weights: int
    engine_size: int
    precision: str
    optimization_level: int
    build_time: float


class TensorRTOptimizer:
    """TensorRT optimization and acceleration engine."""
    
    def __init__(self, config: Optional[TensorRTConfig] = None):
        """
        Initialize TensorRT optimizer.
        
        Args:
            config: TensorRT configuration
        """
        self.config = config or TensorRTConfig()
        self.engines = {}
        self.performance_cache = {}
        
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available - some features will be disabled")
        else:
            logger.info("TensorRT optimizer initialized")
            self._check_tensorrt_version()
    
    def _check_tensorrt_version(self):
        """Check TensorRT version and capabilities."""
        if TRT_AVAILABLE:
            version = trt.__version__
            major_version = int(version.split('.')[0])
            
            logger.info(f"TensorRT version: {version}")
            logger.info(f"TensorRT major version: {major_version}")
            
            # Check for advanced features
            self.has_dynamic_shapes = major_version >= 8
            self.has_nvinfer_plugin = major_version >= 7
            self.has_int8_support = major_version >= 7
            
            logger.info(f"Dynamic shapes support: {self.has_dynamic_shapes}")
            logger.info(f"Plugin support: {self.has_nvinfer_plugin}")
            logger.info(f"INT8 support: {self.has_int8_support}")
    
    def build_engine(self, onnx_model_path: str, output_path: str,
                    config: Optional[TensorRTConfig] = None) -> Dict[str, Any]:
        """
        Build TensorRT engine from ONNX model.
        
        Args:
            onnx_model_path: Path to ONNX model
            output_path: Output path for TensorRT engine
            config: TensorRT configuration
            
        Returns:
            Build results
        """
        logger.info(f"Building TensorRT engine from {onnx_model_path}")
        
        if not TRT_AVAILABLE:
            return {
                "status": "failed",
                "error": "TensorRT not available"
            }
        
        if config is None:
            config = self.config
        
        start_time = time.time()
        
        try:
            # Create TensorRT logger and builder
            logger_trt = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger_trt)
            config_trt = builder.create_builder_config()
            
            # Set workspace size
            config_trt.max_workspace_size = config.max_workspace_size
            
            # Set precision flags
            if config.precision == "fp16":
                config_trt.set_flag(trt.BuilderFlag.FP16)
            elif config.precision == "int8":
                if not self.has_int8_support:
                    raise ValueError("INT8 not supported in this TensorRT version")
                config_trt.set_flag(trt.BuilderFlag.INT8)
                if config.calibration_dataset:
                    self._set_int8_calibration(config_trt, config)
            elif config.precision == "fp32":
                # No special flags needed for FP32
                pass
            
            # Set optimization level
            if hasattr(trt.BuilderFlag, 'OPTIMIZE_REDUCED_PRECISION'):
                if config.precision != "fp32":
                    config_trt.set_flag(trt.BuilderFlag.OPTIMIZE_REDUCED_PRECISION)
            
            # Create network
            network = builder.create_network()
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, logger_trt)
            
            with open(onnx_model_path, 'rb') as model_file:
                if not parser.parse(model_file.read(), onnx_model_path):
                    raise ValueError("Failed to parse ONNX model")
            
            # Set optimization profiles for dynamic shapes
            if self.has_dynamic_shapes and config.max_dyn_batch_size > 1:
                self._set_dynamic_shapes(network, config, builder)
            
            # Build engine
            logger.info("Building TensorRT engine...")
            engine = builder.build_engine(network, config_trt)
            
            if engine is None:
                raise ValueError("Failed to build TensorRT engine")
            
            # Get engine information
            engine_info = self._get_engine_info(engine, network, config)
            
            # Save engine
            if config.save_engine:
                with open(output_path, 'wb') as f:
                    f.write(engine.serialize())
                logger.info(f"Engine saved to {output_path}")
            
            build_time = time.time() - start_time
            
            result = {
                "status": "success",
                "engine_path": output_path if config.save_engine else None,
                "engine_info": engine_info,
                "build_time": build_time,
                "config": {
                    "precision": config.precision,
                    "max_batch_size": config.max_batch_size,
                    "opt_batch_size": config.opt_batch_size,
                    "workspace_size": config.max_workspace_size
                }
            }
            
            # Cache engine
            self.engines[output_path] = {
                "engine": engine,
                "config": config,
                "info": engine_info
            }
            
            logger.info(f"TensorRT engine built successfully in {build_time:.2f}s")
            logger.info(f"Engine size: {engine_info.engine_size / 1024 / 1024:.2f} MB")
            
            return result
            
        except Exception as e:
            logger.error(f"TensorRT engine building failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "build_time": time.time() - start_time
            }
    
    def _set_dynamic_shapes(self, network, config: TensorRTConfig, builder):
        """Set dynamic shapes for optimization profiles."""
        try:
            # This is a simplified implementation
            # In practice, you would analyze the model to find dynamic dimensions
            for input_idx in range(network.num_inputs):
                input_tensor = network.get_input(input_idx)
                if input_tensor.shape[0] == -1:  # Dynamic batch dimension
                    profile = builder.create_optimization_profile()
                    profile.set_shape(
                        input_tensor.name,
                        (1, config.opt_batch_size, config.max_dyn_batch_size),
                        (1, config.opt_batch_size, config.max_dyn_batch_size),
                        (1, config.opt_batch_size, config.max_dyn_batch_size)
                    )
                    break
        except Exception as e:
            logger.warning(f"Failed to set dynamic shapes: {e}")
    
    def _set_int8_calibration(self, config_trt, config: TensorRTConfig):
        """Set INT8 calibration."""
        try:
            if config.calibration_dataset and os.path.exists(config.calibration_dataset):
                # This would require implementing a calibration algorithm
                # For now, we'll just log that calibration is needed
                logger.info("INT8 calibration dataset provided - would implement calibration here")
        except Exception as e:
            logger.warning(f"Failed to set INT8 calibration: {e}")
    
    def _get_engine_info(self, engine, network, config: TensorRTConfig) -> EngineInfo:
        """Get TensorRT engine information."""
        try:
            # Engine size
            engine_size = engine.serialized_size
            
            # Count layers and weights
            num_layers = network.num_layers
            num_weights = 0
            for i in range(network.num_weights):
                weight = network.get_weight(i)
                if weight:
                    num_weights += weight.size
            
            return EngineInfo(
                max_batch_size=engine.max_batch_size,
                max_input_size=engine.max_input_size,
                num_bindings=engine.num_bindings,
                num_layers=num_layers,
                num_weights=num_weights,
                engine_size=engine_size,
                precision=config.precision,
                optimization_level=1,
                build_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Failed to get engine info: {e}")
            return EngineInfo(
                max_batch_size=0,
                max_input_size=0,
                num_bindings=0,
                num_layers=0,
                num_weights=0,
                engine_size=0,
                precision=config.precision,
                optimization_level=0,
                build_time=0.0
            )
    
    def load_engine(self, engine_path: str) -> Any:
        """
        Load TensorRT engine from file.
        
        Args:
            engine_path: Path to TensorRT engine file
            
        Returns:
            TensorRT runtime engine
        """
        logger.info(f"Loading TensorRT engine: {engine_path}")
        
        if not TRT_AVAILABLE or not CUDA_AVAILABLE:
            logger.error("TensorRT or CUDA not available")
            return None
        
        try:
            # Create runtime
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            if engine is None:
                raise ValueError("Failed to deserialize engine")
            
            # Cache engine
            self.engines[engine_path] = {
                "engine": engine,
                "config": None,
                "info": None
            }
            
            logger.info(f"TensorRT engine loaded successfully")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return None
    
    def benchmark_inference(self, engine_path: str, input_data: np.ndarray,
                          num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, Any]:
        """
        Benchmark TensorRT inference performance.
        
        Args:
            engine_path: Path to TensorRT engine
            input_data: Sample input data
            num_runs: Number of inference runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking TensorRT inference: {engine_path}")
        
        if not TRT_AVAILABLE or not CUDA_AVAILABLE:
            return {"error": "TensorRT or CUDA not available"}
        
        try:
            # Load engine
            engine = self.load_engine(engine_path)
            if engine is None:
                return {"error": "Failed to load engine"}
            
            # Create execution context
            context = engine.create_execution_context()
            
            # Allocate device memory
            device_inputs = []
            device_outputs = []
            host_outputs = []
            
            for binding_idx in range(engine.num_bindings):
                binding = engine[binding_idx]
                if engine.binding_is_input(binding_idx):
                    # Input binding
                    input_size = np.prod(input_data.shape) * input_data.itemsize
                    device_input = cuda.mem_alloc(input_size)
                    device_inputs.append((binding_idx, device_input))
                else:
                    # Output binding
                    output_shape = engine.get_binding_shape(binding_idx)
                    output_size = np.prod(output_shape) * 4  # Assume float32
                    device_output = cuda.mem_alloc(output_size)
                    host_output = np.zeros(output_shape, dtype=np.float32)
                    device_outputs.append((binding_idx, device_output))
                    host_outputs.append(host_output)
            
            # Create stream
            stream = cuda.Stream()
            
            # Warm up
            logger.info(f"Warming up with {warmup_runs} runs...")
            for _ in range(warmup_runs):
                self._run_inference(context, device_inputs, device_outputs, host_outputs, 
                                  input_data, stream)
            
            # Benchmark
            logger.info(f"Running {num_runs} benchmark iterations...")
            latencies = []
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                self._run_inference(context, device_inputs, device_outputs, 
                                  host_outputs, input_data, stream)
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            latencies = np.array(latencies)
            results = {
                "engine_path": engine_path,
                "input_shape": input_data.shape,
                "num_runs": num_runs,
                "warmup_runs": warmup_runs,
                "statistics": {
                    "mean_latency_ms": float(np.mean(latencies)),
                    "median_latency_ms": float(np.median(latencies)),
                    "min_latency_ms": float(np.min(latencies)),
                    "max_latency_ms": float(np.max(latencies)),
                    "std_latency_ms": float(np.std(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "p99_latency_ms": float(np.percentile(latencies, 99))
                },
                "throughput_rps": 1000.0 / np.mean(latencies)
            }
            
            logger.info(f"Mean latency: {results['statistics']['mean_latency_ms']:.2f}ms")
            logger.info(f"Throughput: {results['throughput_rps']:.2f} RPS")
            
            return results
            
        except Exception as e:
            logger.error(f"TensorRT inference benchmark failed: {e}")
            return {
                "error": str(e),
                "engine_path": engine_path
            }
    
    def _run_inference(self, context, device_inputs, device_outputs, 
                     host_outputs, input_data, stream):
        """Run a single inference."""
        try:
            # Copy input to device
            for binding_idx, device_buffer in device_inputs:
                cuda.memcpy_htod_async(device_buffer, input_data.flatten(), stream)
            
            # Execute inference
            context.execute_async_v2(
                bindings=[buffer for _, buffer in device_inputs + device_outputs],
                stream_handle=stream.handle
            )
            
            # Copy output from device
            for i, (binding_idx, device_buffer) in enumerate(device_outputs):
                cuda.memcpy_dtoh_async(host_outputs[i], device_buffer, stream)
            
            # Synchronize
            stream.synchronize()
            
        except Exception as e:
            logger.error(f"Inference execution failed: {e}")
            raise
    
    def optimize_memory(self, config: TensorRTConfig, 
                      model_size_mb: float) -> Dict[str, Any]:
        """
        Optimize TensorRT configuration for memory constraints.
        
        Args:
            config: TensorRT configuration
            model_size_mb: Model size in megabytes
            
        Returns:
            Memory optimization recommendations
        """
        logger.info("Optimizing TensorRT for memory constraints")
        
        memory_optimizations = {
            "workspace_size_recommendation": min(config.max_workspace_size, int(model_size_mb * 1024 * 1024 * 2)),
            "precision_recommendation": "fp16" if model_size_mb > 100 else "int8",
            "batch_size_recommendation": 1 if model_size_mb > 200 else 8,
            "optimizations": []
        }
        
        # Memory-based recommendations
        if model_size_mb > 500:
            memory_optimizations["optimizations"].append("Consider model quantization")
            memory_optimizations["precision_recommendation"] = "int8"
        
        if model_size_mb > 1000:
            memory_optimizations["optimizations"].append("Use gradient checkpointing")
            memory_optimizations["optimizations"].append("Implement dynamic batching")
        
        # Workspace size recommendations
        if config.max_workspace_size > model_size_mb * 1024 * 1024 * 4:
            memory_optimizations["optimizations"].append("Reduce workspace size")
        
        return memory_optimizations
    
    def create_int8_calibrator(self, calibration_data_path: str,
                             num_calibration_samples: int = 100) -> Dict[str, Any]:
        """
        Create INT8 calibrator for quantization.
        
        Args:
            calibration_data_path: Path to calibration data
            num_calibration_samples: Number of calibration samples to use
            
        Returns:
            Calibrator configuration
        """
        logger.info("Creating INT8 calibrator")
        
        calibrator_config = {
            "type": "entropy",  # entropy, min_max, percentile
            "num_samples": num_calibration_samples,
            "data_path": calibration_data_path,
            "precision": "int8"
        }
        
        # In a real implementation, you would:
        # 1. Load calibration dataset
        # 2. Run inference with FP32 model
        # 3. Collect activation statistics
        # 4. Create calibration table
        
        logger.info("INT8 calibrator configuration created")
        return calibrator_config
    
    def compare_precisions(self, onnx_model_path: str, 
                          output_dir: str,
                          precisions: List[str] = None) -> Dict[str, Any]:
        """
        Compare different precision modes (FP32, FP16, INT8).
        
        Args:
            onnx_model_path: Path to ONNX model
            output_dir: Output directory for engines
            precisions: List of precisions to compare
            
        Returns:
            Comparison results
        """
        logger.info("Comparing TensorRT precision modes")
        
        if precisions is None:
            precisions = ["fp32", "fp16", "int8"]
        
        results = {
            "model_path": onnx_model_path,
            "precisions": {},
            "comparison": {}
        }
        
        for precision in precisions:
            logger.info(f"Building {precision.upper()} engine")
            
            # Create config for this precision
            config = TensorRTConfig(
                precision=precision,
                save_engine=True,
                calibration_dataset=onnx_model_path.replace('.onnx', '_calib_data')
            )
            
            # Build engine
            engine_name = f"model_{precision}.engine"
            engine_path = os.path.join(output_dir, engine_name)
            
            build_result = self.build_engine(onnx_model_path, engine_path, config)
            
            if build_result.get("status") == "success":
                results["precisions"][precision] = {
                    "engine_path": engine_path,
                    "engine_info": build_result["engine_info"],
                    "build_time": build_result["build_time"]
                }
            else:
                results["precisions"][precision] = {
                    "status": "failed",
                    "error": build_result.get("error")
                }
        
        # Compare results
        successful_precisions = [p for p in precisions if p in results["precisions"] 
                               and results["precisions"][p].get("status") != "failed"]
        
        if len(successful_precisions) > 1:
            base_precision = successful_precisions[0]
            base_info = results["precisions"][base_precision]["engine_info"]
            
            results["comparison"] = {}
            for precision in successful_precisions[1:]:
                info = results["precisions"][precision]["engine_info"]
                size_reduction = (base_info.engine_size - info.engine_size) / base_info.engine_size * 100
                
                results["comparison"][precision] = {
                    "size_reduction_vs_fp32": size_reduction,
                    "engine_size_mb": info.engine_size / 1024 / 1024
                }
        
        return results
    
    def optimize_for_device(self, device: Dict[str, Any], 
                          onnx_model_path: str) -> Dict[str, Any]:
        """
        Optimize TensorRT configuration for specific device.
        
        Args:
            device: Target edge device
            onnx_model_path: Path to ONNX model
            
        Returns:
            Device-specific optimization configuration
        """
        logger.info(f"Optimizing TensorRT for {device.get('name', 'Unknown Device')}")
        
        # Get device specifications
        gpu_memory = device.get("memory_gb", 8.0) * 1024  # Convert to MB
        max_power = device.get("max_power_watts", 5.0)
        
        # Create device-specific configuration
        if device.get("gpu") and "NVIDIA" in device.get("gpu", ""):
            # NVIDIA GPU device
            config = TensorRTConfig(
                precision="fp16" if gpu_memory > 4 else "int8",
                max_workspace_size=int(gpu_memory * 1024 * 1024 * 0.7),  # 70% of GPU memory
                max_batch_size=min(32, int(gpu_memory * 4)),  # Scale with memory
                opt_batch_size=1,
                use_fp16=True,
                use_int8=gpu_memory <= 4,
                allow_gpu_fallback=True
            )
        elif device.get("gpu") and "AMD" in device.get("gpu", ""):
            # AMD GPU device
            config = TensorRTConfig(
                precision="fp16",
                max_workspace_size=int(gpu_memory * 1024 * 1024 * 0.6),  # 60% of GPU memory
                max_batch_size=16,
                opt_batch_size=1,
                use_fp16=True,
                use_int8=False,  # Limited INT8 support
                allow_gpu_fallback=True
            )
        else:
            # CPU-only device
            config = TensorRTConfig(
                precision="fp32",
                max_workspace_size=256 * 1024 * 1024,  # 256 MB
                max_batch_size=4,
                opt_batch_size=1,
                use_fp16=False,
                use_int8=False,
                allow_gpu_fallback=True
            )
        
        # Apply power optimization
        if max_power < 10:  # Low power device
            config.max_batch_size = min(config.max_batch_size, 4)
            config.max_workspace_size = min(config.max_workspace_size, 256 * 1024 * 1024)
        
        optimization_recommendations = {
            "recommended_config": {
                "precision": config.precision,
                "max_batch_size": config.max_batch_size,
                "workspace_size_mb": config.max_workspace_size // 1024 // 1024
            },
            "device_specs": {
                "memory_gb": device.get("memory_gb", 8.0),
                "max_power_watts": max_power,
                "gpu": device.get("gpu", "None")
            },
            "optimizations": []
        }
        
        # Add optimization recommendations
        if config.precision == "fp16":
            optimization_recommendations["optimizations"].append("Use Tensor Core operations")
        if config.precision == "int8":
            optimization_recommendations["optimizations"].append("Enable INT8 tensor cores")
        if config.max_batch_size > 1:
            optimization_recommendations["optimizations"].append("Enable dynamic batching")
        
        return optimization_recommendations
