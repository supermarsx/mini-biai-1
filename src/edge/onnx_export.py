"""
ONNX Export and Optimization Module

Provides comprehensive ONNX model export and optimization capabilities for
edge deployment including graph optimization, operator fusion, and hardware-specific optimizations.

Features:
- PyTorch/TensorFlow to ONNX export
- ONNX model optimization
- Graph transformation
- Operator fusion
- Hardware-specific optimizations
- ONNX Runtime integration
- Model validation and testing
"""

import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import tempfile

try:
    import torch
    import torch.nn as nn
    from torch.jit import trace
except ImportError:
    torch = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tf2onnx import convert
except ImportError:
    tf = None

try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.tools import optimizer
    from onnxruntime.tools.optimizer import get_optimization_levels
    from onnxruntime.tools.optimizer import optimize_model
except ImportError:
    onnx = None
    ort = None

# from . import EdgeDevice, OptimizationConfig
# Using simplified imports for demo

logger = logging.getLogger(__name__)


@dataclass
class ONNXOptimizationConfig:
    """Configuration for ONNX optimizations."""
    optimize_for: str = "all"  # all, cpu, cuda, tensorrt
    graph_optimization: bool = True
    operator_fusion: bool = True
    constant_folding: bool = True
    dead_code_elimination: bool = True
    shape_inference: bool = True
    validate_model: bool = True


@dataclass
class ExportResult:
    """Results from ONNX export process."""
    original_model_path: str
    onnx_model_path: str
    export_time: float
    model_size_original: int
    model_size_onnx: int
    optimization_time: Optional[float] = None
    optimized_model_path: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None


class ONNXExporter:
    """ONNX model export and optimization engine."""
    
    def __init__(self):
        """Initialize ONNX exporter."""
        self.export_history = []
        self.optimization_cache = {}
        
        # Check available frameworks
        self.torch_available = torch is not None
        self.tf_available = tf is not None
        self.onnx_available = onnx is not None
        self.ort_available = ort is not None
        
        logger.info(f"ONNX Exporter initialized - "
                   f"PyTorch: {self.torch_available}, "
                   f"TensorFlow: {self.tf_available}, "
                   f"ONNX: {self.onnx_available}, "
                   f"ONNX Runtime: {self.ort_available}")
    
    def export_model(self, model_path: str, output_path: Optional[str] = None,
                    input_shape: Optional[Tuple[int, ...]] = None,
                    opset_version: int = 11) -> ExportResult:
        """
        Export model to ONNX format.
        
        Args:
            model_path: Path to the source model
            output_path: Optional output path for ONNX model
            input_shape: Optional input shape specification
            opset_version: ONNX opset version
            
        Returns:
            ExportResult with export details
        """
        logger.info(f"Exporting model {model_path} to ONNX")
        
        import time
        start_time = time.time()
        
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(model_path))[0]
            output_path = f"{base_name}.onnx"
        
        try:
            # Detect model format and export
            if self.torch_available and self._is_pytorch_model(model_path):
                result = self._export_pytorch_model(model_path, output_path, input_shape, opset_version)
            elif self.tf_available and self._is_tensorflow_model(model_path):
                result = self._export_tensorflow_model(model_path, output_path, input_shape, opset_version)
            else:
                raise ValueError(f"Unsupported model format for ONNX export: {model_path}")
            
            result.export_time = time.time() - start_time
            result.model_size_original = os.path.getsize(model_path)
            result.model_size_onnx = os.path.getsize(output_path)
            
            logger.info(f"Model exported successfully to {output_path}")
            logger.info(f"Export time: {result.export_time:.2f}s")
            logger.info(f"Size reduction: {(1 - result.model_size_onnx / result.model_size_original) * 100:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return ExportResult(
                original_model_path=model_path,
                onnx_model_path="",
                export_time=time.time() - start_time,
                model_size_original=os.path.getsize(model_path),
                model_size_onnx=0,
                errors=[str(e)]
            )
    
    def _is_pytorch_model(self, model_path: str) -> bool:
        """Check if model is a PyTorch model."""
        ext = os.path.splitext(model_path)[1].lower()
        if ext in ['.pt', '.pth', '.pkl']:
            return True
        
        try:
            with open(model_path, 'rb') as f:
                return f.read(2) == b'\x80\x01'  # PT file signature
        except:
            return False
    
    def _is_tensorflow_model(self, model_path: str) -> bool:
        """Check if model is a TensorFlow model."""
        ext = os.path.splitext(model_path)[1].lower()
        if ext in ['.h5', '.pb', '.tflite']:
            return True
        
        try:
            if tf:
                tf.saved_model.load(model_path)
                return True
        except:
            pass
        
        return False
    
    def _export_pytorch_model(self, model_path: str, output_path: str,
                            input_shape: Optional[Tuple[int, ...]],
                            opset_version: int) -> ExportResult:
        """Export PyTorch model to ONNX."""
        logger.info("Exporting PyTorch model")
        
        # Load PyTorch model
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        model.eval()
        
        # Determine input shape
        if input_shape is None:
            # Try to infer from model or use default
            input_shape = (1, 3, 224, 224)  # Common image input shape
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
        
        return ExportResult(
            original_model_path=model_path,
            onnx_model_path=output_path,
            export_time=0.0,  # Will be set by caller
            model_size_original=0,  # Will be set by caller
            model_size_onnx=0  # Will be set by caller
        )
    
    def _export_tensorflow_model(self, model_path: str, output_path: str,
                               input_shape: Optional[Tuple[int, ...]],
                               opset_version: int) -> ExportResult:
        """Export TensorFlow model to ONNX."""
        logger.info("Exporting TensorFlow model")
        
        try:
            # Load TensorFlow model
            if model_path.endswith('.h5'):
                model = keras.models.load_model(model_path)
            else:
                model = tf.saved_model.load(model_path)
            
            # Determine input shape
            if input_shape is None:
                # Try to get from model
                if hasattr(model, 'input_shape'):
                    input_shape = model.input_shape
                else:
                    input_shape = (1, 224, 224, 3)  # Default
            
            # Convert to ONNX
            if model_path.endswith('.h5'):
                # Convert Keras model
                onnx_model, _ = convert.from_keras(
                    model,
                    input_signature=[
                        tf.TensorSpec(input_shape, tf.float32, name='input')
                    ],
                    opset=opset_version
                )
            else:
                # Convert SavedModel
                onnx_model, _ = convert.from_saved_model(
                    model_path,
                    input_signature=[
                        tf.TensorSpec(input_shape, tf.float32, name='input')
                    ],
                    opset=opset_version
                )
            
            # Save ONNX model
            onnx.save(onnx_model, output_path)
            
            return ExportResult(
                original_model_path=model_path,
                onnx_model_path=output_path,
                export_time=0.0,  # Will be set by caller
                model_size_original=0,  # Will be set by caller
                model_size_onnx=0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"TensorFlow export failed: {e}")
            raise
    
    def optimize_onnx_model(self, model_path: str, 
                          config: ONNXOptimizationConfig,
                          target_device: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize ONNX model for inference.
        
        Args:
            model_path: Path to ONNX model
            config: Optimization configuration
            target_device: Optional target device (cpu, cuda, tensorrt)
            
        Returns:
            Optimization results
        """
        logger.info(f"Optimizing ONNX model: {model_path}")
        
        if not self.onnx_available:
            raise ImportError("ONNX not available for optimization")
        
        import time
        start_time = time.time()
        
        try:
            original_size = os.path.getsize(model_path)
            
            # Create optimization config
            optimization_config = self._create_ort_optimization_config(config, target_device)
            
            # Optimize model
            optimized_path = model_path.replace('.onnx', '_optimized.onnx')
            optimized_model = optimize_model(
                model_path,
                optimization_config=optimization_config
            )
            
            # Save optimized model
            onnx.save(optimized_model, optimized_path)
            optimized_size = os.path.getsize(optimized_path)
            
            optimization_time = time.time() - start_time
            
            # Validate optimized model
            validation_results = None
            if config.validate_model:
                validation_results = self._validate_onnx_model(optimized_path)
            
            result = {
                "status": "success",
                "original_model": model_path,
                "optimized_model": optimized_path,
                "original_size": original_size,
                "optimized_size": optimized_size,
                "size_reduction": (original_size - optimized_size) / original_size * 100,
                "optimization_time": optimization_time,
                "config": {
                    "optimize_for": config.optimize_for,
                    "graph_optimization": config.graph_optimization,
                    "operator_fusion": config.operator_fusion,
                    "constant_folding": config.constant_folding,
                    "dead_code_elimination": config.dead_code_elimination
                },
                "validation_results": validation_results
            }
            
            logger.info(f"ONNX optimization completed in {optimization_time:.2f}s")
            logger.info(f"Size reduction: {result['size_reduction']:.1f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"ONNX optimization failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model_path": model_path
            }
    
    def _create_ort_optimization_config(self, config: ONNXOptimizationConfig,
                                      target_device: Optional[str]):
        """Create ONNX Runtime optimization configuration."""
        if not self.ort_available:
            return None
        
        try:
            # Get optimization levels
            levels = get_optimization_levels()
            
            # Select optimization level based on config
            if config.optimize_for == "all":
                level = levels.get('all')
            elif config.optimize_for == "cpu":
                level = levels.get('cpu')
            elif config.optimize_for == "cuda":
                level = levels.get('cuda')
            else:
                level = levels.get('all')  # Default
            
            return level
            
        except Exception as e:
            logger.warning(f"Failed to create optimization config: {e}")
            return None
    
    def _validate_onnx_model(self, model_path: str) -> Dict[str, Any]:
        """Validate ONNX model integrity."""
        logger.info("Validating ONNX model")
        
        try:
            # Load model
            model = onnx.load(model_path)
            
            # Check model validity
            onnx.checker.check_model(model)
            
            # Get model info
            graph = model.graph
            node_count = len(graph.node)
            initializer_count = len(graph.initializer)
            input_shapes = []
            output_shapes = []
            
            # Extract input/output shapes
            for input_info in graph.input:
                if input_info.name not in [init.name for init in graph.initializer]:
                    shape = []
                    for dim in input_info.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')
                    input_shapes.append({"name": input_info.name, "shape": shape})
            
            for output_info in graph.output:
                shape = []
                for dim in output_info.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else 'dynamic')
                output_shapes.append({"name": output_info.name, "shape": shape})
            
            return {
                "status": "valid",
                "nodes": node_count,
                "initializers": initializer_count,
                "inputs": input_shapes,
                "outputs": output_shapes,
                "opset_version": model.opset_import[0].version if model.opset_import else "unknown"
            }
            
        except Exception as e:
            logger.error(f"ONNX model validation failed: {e}")
            return {
                "status": "invalid",
                "error": str(e)
            }
    
    def benchmark_onnx_inference(self, model_path: str, 
                               input_data: np.ndarray,
                               providers: List[str] = None) -> Dict[str, Any]:
        """
        Benchmark ONNX model inference performance.
        
        Args:
            model_path: Path to ONNX model
            input_data: Sample input data
            providers: Optional list of execution providers
            
        Returns:
            Benchmark results
        """
        logger.info("Benchmarking ONNX inference")
        
        if not self.ort_available:
            return {"error": "ONNX Runtime not available"}
        
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        try:
            import time
            
            # Create session
            session = ort.InferenceSession(model_path, providers=providers)
            
            # Warm up
            for _ in range(10):
                session.run(None, {'input': input_data})
            
            # Benchmark
            num_runs = 100
            latencies = []
            
            for i in range(num_runs):
                start_time = time.perf_counter()
                outputs = session.run(None, {'input': input_data})
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            latencies = np.array(latencies)
            results = {
                "model_path": model_path,
                "providers": providers,
                "input_shape": input_data.shape,
                "runs": num_runs,
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
            logger.error(f"ONNX inference benchmark failed: {e}")
            return {
                "error": str(e),
                "model_path": model_path
            }
    
    def convert_to_tensorrt(self, model_path: str, 
                          precision: str = "fp16",
                          max_workspace_size: int = 1 << 30) -> Dict[str, Any]:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            model_path: Path to ONNX model
            precision: Precision mode (fp32, fp16, int8)
            max_workspace_size: Maximum workspace size in bytes
            
        Returns:
            Conversion results
        """
        logger.info(f"Converting ONNX to TensorRT: {model_path}")
        
        try:
            import tensorrt as trt
            
            # Create builder
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            config = builder.create_builder_config()
            
            # Set precision
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                # In practice, you would set INT8 calibration here
            
            # Set workspace size
            config.max_workspace_size = max_workspace_size
            
            # Load ONNX model
            network = builder.create_network()
            parser = trt.OnnxParser(network, logger)
            
            with open(model_path, 'rb') as model_file:
                if not parser.parse(model_file.read(), model_path):
                    raise ValueError("Failed to parse ONNX model")
            
            # Build engine
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise ValueError("Failed to build TensorRT engine")
            
            # Save engine
            engine_path = model_path.replace('.onnx', f'_{precision}.engine')
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            # Get engine info
            engine_info = {
                "max_batch_size": engine.max_batch_size,
                "max_input_size": engine.max_input_size,
                "num_bindings": engine.num_bindings,
                "num_layers": network.num_layers
            }
            
            logger.info(f"TensorRT engine created: {engine_path}")
            
            return {
                "status": "success",
                "engine_path": engine_path,
                "precision": precision,
                "engine_info": engine_info,
                "workspace_size": max_workspace_size
            }
            
        except ImportError:
            logger.error("TensorRT not available")
            return {
                "status": "failed",
                "error": "TensorRT not available"
            }
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model_path": model_path
            }
    
    def create_model_archive(self, model_path: str, output_dir: str,
                           include_optimizations: bool = True) -> Dict[str, str]:
        """
        Create model archive with multiple optimized versions.
        
        Args:
            model_path: Path to original model
            output_dir: Output directory for archive
            include_optimizations: Whether to include optimized versions
            
        Returns:
            Dictionary mapping model types to file paths
        """
        logger.info(f"Creating model archive for {model_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to ONNX
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
        
        export_result = self.export_model(model_path, onnx_path)
        
        archive_files = {
            "original": model_path,
            "onnx": onnx_path
        }
        
        # Create optimized versions
        if include_optimizations and self.onnx_available:
            # CPU optimized
            cpu_config = ONNXOptimizationConfig(optimize_for="cpu")
            cpu_result = self.optimize_onnx_model(onnx_path, cpu_config)
            if cpu_result.get("status") == "success":
                archive_files["onnx_cpu_optimized"] = cpu_result["optimized_model"]
            
            # All optimizations
            all_config = ONNXOptimizationConfig(
                graph_optimization=True,
                operator_fusion=True,
                constant_folding=True,
                dead_code_elimination=True
            )
            all_result = self.optimize_onnx_model(onnx_path, all_config)
            if all_result.get("status") == "success":
                archive_files["onnx_fully_optimized"] = all_result["optimized_model"]
        
        # Create manifest
        manifest = {
            "original_model": model_path,
            "base_name": base_name,
            "created": str(np.datetime64('now')),
            "files": archive_files,
            "export_result": {
                "export_time": export_result.export_time,
                "original_size": export_result.model_size_original,
                "onnx_size": export_result.model_size_onnx
            }
        }
        
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        archive_files["manifest"] = manifest_path
        
        logger.info(f"Model archive created in {output_dir}")
        return archive_files
    
    def analyze_model_operations(self, model_path: str) -> Dict[str, Any]:
        """
        Analyze operations in ONNX model.
        
        Args:
            model_path: Path to ONNX model
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing ONNX model operations: {model_path}")
        
        if not self.onnx_available:
            return {"error": "ONNX not available"}
        
        try:
            model = onnx.load(model_path)
            
            # Count operations
            op_counts = {}
            total_nodes = len(model.graph.node)
            
            for node in model.graph.node:
                op_type = node.op_type
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
            
            # Identify supported operations by common execution providers
            cpu_supported = set()
            cuda_supported = set()
            tensorrt_supported = set()
            
            # Common operations supported by each provider
            cpu_ops = ["Add", "Mul", "MatMul", "Conv", "Relu", "MaxPool", "AveragePool", "Gemm", "Reshape", "Transpose", "Concat", "Split", "Pad", "Squeeze", "Unsqueeze"]
            cuda_ops = cpu_ops + ["ConvTranspose", "LSTM", "GRU"]
            tensorrt_ops = cpu_ops + ["ConvTranspose", "LSTM", "GRU", "TopK"]
            
            for op in op_counts.keys():
                if op in cpu_ops:
                    cpu_supported.add(op)
                if op in cuda_ops:
                    cuda_supported.add(op)
                if op in tensorrt_ops:
                    tensorrt_supported.add(op)
            
            analysis = {
                "model_path": model_path,
                "total_operations": total_nodes,
                "operation_counts": op_counts,
                "supported_operations": {
                    "cpu": list(cpu_supported),
                    "cuda": list(cuda_supported),
                    "tensorrt": list(tensorrt_supported)
                },
                "unsupported_operations": {
                    "cpu": [op for op in op_counts.keys() if op not in cpu_ops],
                    "cuda": [op for op in op_counts.keys() if op not in cuda_ops],
                    "tensorrt": [op for op in op_counts.keys() if op not in tensorrt_ops]
                },
                "compatibility": {
                    "cpu": len([op for op in op_counts.keys() if op in cpu_ops]) / total_nodes,
                    "cuda": len([op for op in op_counts.keys() if op in cuda_ops]) / total_nodes,
                    "tensorrt": len([op for op in op_counts.keys() if op in tensorrt_ops]) / total_nodes
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            return {
                "error": str(e),
                "model_path": model_path
            }