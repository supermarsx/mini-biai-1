"""
Progressive Model Loading Module

Provides progressive loading capabilities for edge devices to improve
startup time and reduce memory usage by loading model components on-demand.

Features:
- Chunked model loading
- Priority-based loading
- Dynamic dependency resolution
- Memory-aware loading
- Background loading
- Progressive quantization
- Lazy loading strategies
- Performance monitoring
"""

import os
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    TORCH_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ort = None
    ONNX_AVAILABLE = False

# Removed problematic imports for standalone usage
# from . import EdgeDevice, OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadChunk:
    """Represents a model component chunk for progressive loading."""
    chunk_id: str
    name: str
    chunk_type: str  # "weights", "ops", "layers", "quantization"
    priority: int
    size_bytes: int
    dependencies: List[str]
    content: Optional[bytes] = None
    file_path: Optional[str] = None
    loaded: bool = False
    loading: bool = False


@dataclass
class LoadingStats:
    """Statistics for progressive loading process."""
    total_chunks: int
    loaded_chunks: int
    failed_chunks: int
    current_phase: str
    memory_used_mb: float
    loading_speed_mbps: float
    estimated_time_remaining: float
    start_time: float


@dataclass
class ProgressiveConfig:
    """Configuration for progressive loading."""
    chunk_size_mb: int = 4  # 4MB chunks
    max_memory_mb: int = 256
    loading_threads: int = 4
    enable_priorities: bool = True
    enable_quantization_progressive: bool = True
    enable_background_loading: bool = True
    cache_loaded_chunks: bool = True
    validate_chunks: bool = True
    prewarm_critical_chunks: bool = True


class ProgressiveLoader:
    """Progressive model loading engine for edge devices."""
    
    def __init__(self, config: Optional[ProgressiveConfig] = None):
        """
        Initialize progressive loader.
        
        Args:
            config: Progressive loading configuration
        """
        self.config = config or ProgressiveConfig()
        self.loaded_chunks = {}
        self.loading_queue = deque()
        self.load_stats = {}
        self.model_graph = {}
        self.dependencies = defaultdict(list)
        self.memory_monitor = MemoryMonitor()
        self.loading_executor = None
        
        # Performance tracking
        self.load_performance = []
        self.chunk_cache = {}
        
        # Check framework availability
        self.torch_available = TORCH_AVAILABLE
        self.onnx_available = ONNX_AVAILABLE
        
        logger.info(f"Progressive loader initialized - "
                   f"chunk_size: {self.config.chunk_size_mb}MB, "
                   f"max_memory: {self.config.max_memory_mb}MB, "
                   f"threads: {self.config.loading_threads}")
    
    def prepare_progressive_loading(self, model_path: str, 
                                  output_dir: str) -> Dict[str, Any]:
        """
        Prepare model for progressive loading by creating chunks.
        
        Args:
            model_path: Path to the model
            output_dir: Output directory for chunks
            
        Returns:
            Preparation results with chunk information
        """
        logger.info(f"Preparing progressive loading for {model_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Analyze model structure
            model_analysis = self._analyze_model_structure(model_path)
            
            # Create chunks based on analysis
            chunks = self._create_chunks(model_analysis, model_path, output_dir)
            
            # Create loading plan
            loading_plan = self._create_loading_plan(chunks)
            
            # Save chunk metadata
            self._save_chunk_metadata(output_dir, chunks, loading_plan)
            
            # Create loader configuration
            loader_config = self._create_loader_config(output_dir, chunks, loading_plan)
            
            results = {
                "status": "success",
                "model_path": model_path,
                "output_dir": output_dir,
                "total_chunks": len(chunks),
                "total_size_mb": sum(chunk.size_bytes for chunk in chunks) / 1024 / 1024,
                "chunks": [asdict(chunk) for chunk in chunks],
                "loading_plan": loading_plan,
                "loader_config": loader_config,
                "critical_path": self._identify_critical_path(chunks),
                "estimated_load_time": self._estimate_loading_time(chunks)
            }
            
            logger.info(f"Progressive loading prepared: {len(chunks)} chunks, "
                       f"{results['total_size_mb']:.1f}MB total")
            
            return results
            
        except Exception as e:
            logger.error(f"Progressive loading preparation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model_path": model_path
            }
    
    def _analyze_model_structure(self, model_path: str) -> Dict[str, Any]:
        """Analyze model structure to determine chunking strategy."""
        logger.info("Analyzing model structure")
        
        analysis = {
            "model_path": model_path,
            "model_type": self._detect_model_type(model_path),
            "total_size_bytes": os.path.getsize(model_path),
            "operations": [],
            "layers": [],
            "parameters": [],
            "complexity_score": 0
        }
        
        try:
            if analysis["model_type"] == "pytorch" and self.torch_available:
                analysis = self._analyze_pytorch_model(model_path, analysis)
            elif analysis["model_type"] == "onnx" and self.onnx_available:
                analysis = self._analyze_onnx_model(model_path, analysis)
            else:
                analysis = self._analyze_generic_model(model_path, analysis)
                
        except Exception as e:
            logger.warning(f"Detailed model analysis failed: {e}")
            analysis = self._analyze_generic_model(model_path, analysis)
        
        return analysis
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect model file type."""
        ext = os.path.splitext(model_path)[1].lower()
        
        if ext in ['.pt', '.pth', '.pkl']:
            return "pytorch"
        elif ext in ['.onnx']:
            return "onnx"
        elif ext in ['.h5', '.pb', '.tflite']:
            return "tensorflow"
        else:
            return "unknown"
    
    def _analyze_pytorch_model(self, model_path: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze PyTorch model structure."""
        try:
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            
            # Analyze layers
            layers = []
            parameters = []
            
            for name, module in model.named_modules():
                if len(list(module.children())) == 0:  # Leaf module
                    layer_info = {
                        "name": name,
                        "type": module.__class__.__name__,
                        "parameters": sum(p.numel() for p in module.parameters()),
                        "trainable": any(p.requires_grad for p in module.parameters())
                    }
                    layers.append(layer_info)
            
            # Get parameter information
            for name, param in model.named_parameters():
                param_info = {
                    "name": name,
                    "shape": list(param.shape),
                    "size": param.numel(),
                    "dtype": str(param.dtype)
                }
                parameters.append(param_info)
            
            analysis["layers"] = layers
            analysis["parameters"] = parameters
            analysis["total_parameters"] = sum(p["size"] for p in parameters)
            analysis["complexity_score"] = len(layers) + len(parameters) // 1000
            
        except Exception as e:
            logger.warning(f"PyTorch analysis failed: {e}")
        
        return analysis
    
    def _analyze_onnx_model(self, model_path: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ONNX model structure."""
        try:
            model = onnx.load(model_path)
            
            # Analyze operations
            operations = []
            op_counts = defaultdict(int)
            
            for node in model.graph.node:
                op_info = {
                    "name": node.name,
                    "type": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output)
                }
                operations.append(op_info)
                op_counts[node.op_type] += 1
            
            analysis["operations"] = operations
            analysis["operation_counts"] = dict(op_counts)
            analysis["complexity_score"] = len(operations)
            
        except Exception as e:
            logger.warning(f"ONNX analysis failed: {e}")
        
        return analysis
    
    def _analyze_generic_model(self, model_path: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generic model analysis for unknown formats."""
        file_size = os.path.getsize(model_path)
        
        analysis["total_size_bytes"] = file_size
        analysis["estimated_parameters"] = file_size // 4  # Assume 4 bytes per parameter
        analysis["complexity_score"] = file_size // 100000  # Rough complexity estimate
        
        return analysis
    
    def _create_chunks(self, analysis: Dict[str, Any], model_path: str, 
                      output_dir: str) -> List[LoadChunk]:
        """Create loading chunks based on model analysis."""
        logger.info("Creating loading chunks")
        
        chunks = []
        chunk_size_bytes = self.config.chunk_size_mb * 1024 * 1024
        
        model_type = analysis["model_type"]
        total_size = analysis["total_size_bytes"]
        
        try:
            if model_type == "pytorch" and self.torch_available:
                chunks = self._create_pytorch_chunks(analysis, model_path, output_dir, chunk_size_bytes)
            elif model_type == "onnx" and self.onnx_available:
                chunks = self._create_onnx_chunks(analysis, model_path, output_dir, chunk_size_bytes)
            else:
                chunks = self._create_generic_chunks(analysis, model_path, output_dir, chunk_size_bytes)
                
        except Exception as e:
            logger.error(f"Chunk creation failed: {e}")
            chunks = self._create_generic_chunks(analysis, model_path, output_dir, chunk_size_bytes)
        
        # Add quantization chunks if enabled
        if self.config.enable_quantization_progressive:
            quant_chunks = self._create_quantization_chunks(analysis, output_dir)
            chunks.extend(quant_chunks)
        
        # Sort chunks by priority
        chunks.sort(key=lambda x: x.priority)
        
        return chunks
    
    def _create_pytorch_chunks(self, analysis: Dict[str, Any], model_path: str,
                             output_dir: str, chunk_size_bytes: int) -> List[LoadChunk]:
        """Create chunks for PyTorch models."""
        chunks = []
        
        # Load model parameters
        model = torch.load(model_path, map_location='cpu')
        if isinstance(model, dict) and 'model' in model:
            model = model['model']
        
        # Group parameters by layer
        layer_params = {}
        for name, param in model.named_parameters():
            layer_name = name.split('.')[0]  # Get top-level layer
            if layer_name not in layer_params:
                layer_params[layer_name] = {}
            layer_params[layer_name][name] = param
        
        # Create chunks for parameter groups
        current_chunk_data = {}
        current_size = 0
        chunk_index = 0
        
        for layer_name, params in layer_params.items():
            for param_name, param in params.items():
                param_size = param.numel() * param.element_size()
                
                if current_size + param_size > chunk_size_bytes and current_chunk_data:
                    # Create chunk
                    chunk = self._create_chunk_from_data(
                        f"layer_{chunk_index:03d}",
                        f"Layer {layer_name}",
                        "weights",
                        current_chunk_data,
                        output_dir,
                        priority=self._calculate_layer_priority(layer_name)
                    )
                    chunks.append(chunk)
                    
                    # Reset for next chunk
                    current_chunk_data = {}
                    current_size = 0
                    chunk_index += 1
                
                # Add parameter to current chunk
                current_chunk_data[param_name] = param.detach().numpy()
                current_size += param_size
        
        # Add final chunk if any data remains
        if current_chunk_data:
            chunk = self._create_chunk_from_data(
                f"layer_{chunk_index:03d}",
                f"Final layer group",
                "weights",
                current_chunk_data,
                output_dir,
                priority=len(layer_params)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_onnx_chunks(self, analysis: Dict[str, Any], model_path: str,
                          output_dir: str, chunk_size_bytes: int) -> List[LoadChunk]:
        """Create chunks for ONNX models."""
        chunks = []
        
        try:
            model = onnx.load(model_path)
            model_data = open(model_path, 'rb').read()
            
            # Split model into operation chunks
            operations = analysis.get("operations", [])
            chunk_index = 0
            
            # Create header chunk with model metadata
            header_chunk = LoadChunk(
                chunk_id="header",
                name="Model Header",
                chunk_type="metadata",
                priority=0,
                size_bytes=len(json.dumps({
                    "model_type": "onnx",
                    "operations_count": len(operations),
                    "input_shapes": [node.input for node in operations[:5]],
                    "output_shapes": [node.output for node in operations[-5:]]
                }).encode()),
                dependencies=[]
            )
            chunks.append(header_chunk)
            
            # Create chunks for operation groups
            ops_per_chunk = max(1, chunk_size_bytes // 1000)  # Approximate operations per chunk
            for i in range(0, len(operations), ops_per_chunk):
                chunk_ops = operations[i:i + ops_per_chunk]
                chunk_id = f"ops_{chunk_index:03d}"
                
                chunk = LoadChunk(
                    chunk_id=chunk_id,
                    name=f"Operations {i+1}-{i+len(chunk_ops)}",
                    chunk_type="ops",
                    priority=i // ops_per_chunk + 1,  # Higher priority for earlier chunks
                    size_bytes=len(json.dumps(chunk_ops).encode()),
                    dependencies=["header"] if i == 0 else [f"ops_{chunk_index-1:03d}"]
                )
                chunks.append(chunk)
                chunk_index += 1
                
        except Exception as e:
            logger.warning(f"ONNX chunking failed: {e}")
            return self._create_generic_chunks(analysis, model_path, output_dir, chunk_size_bytes)
        
        return chunks
    
    def _create_generic_chunks(self, analysis: Dict[str, Any], model_path: str,
                             output_dir: str, chunk_size_bytes: int) -> List[LoadChunk]:
        """Create chunks for generic model formats."""
        chunks = []
        
        try:
            # Read model file in chunks
            with open(model_path, 'rb') as f:
                chunk_data = f.read(chunk_size_bytes)
                chunk_index = 0
                
                while chunk_data:
                    chunk_id = f"model_{chunk_index:03d}"
                    chunk = LoadChunk(
                        chunk_id=chunk_id,
                        name=f"Model chunk {chunk_index + 1}",
                        chunk_type="raw",
                        priority=chunk_index,
                        size_bytes=len(chunk_data),
                        dependencies=[] if chunk_index == 0 else [f"model_{chunk_index-1:03d}"],
                        content=chunk_data,
                        file_path=os.path.join(output_dir, f"{chunk_id}.bin")
                    )
                    
                    # Save chunk data
                    with open(chunk.file_path, 'wb') as chunk_file:
                        chunk_file.write(chunk.content)
                    
                    chunks.append(chunk)
                    chunk_data = f.read(chunk_size_bytes)
                    chunk_index += 1
                    
        except Exception as e:
            logger.error(f"Generic chunking failed: {e}")
            raise
        
        return chunks
    
    def _create_quantization_chunks(self, analysis: Dict[str, Any], 
                                  output_dir: str) -> List[LoadChunk]:
        """Create quantization-related chunks for progressive quantization."""
        chunks = []
        
        # Create calibration data chunk
        calibration_chunk = LoadChunk(
            chunk_id="quantization_calib",
            name="Quantization Calibration",
            chunk_type="quantization",
            priority=100,  # Load after core model
            size_bytes=1024 * 1024,  # 1MB calibration data
            dependencies=[]
        )
        chunks.append(calibration_chunk)
        
        # Create INT8 weights chunk
        int8_chunk = LoadChunk(
            chunk_id="quantization_int8",
            name="INT8 Quantized Weights",
            chunk_type="quantization",
            priority=101,
            size_bytes=int(analysis["total_size_bytes"] * 0.7),  # 70% of original size
            dependencies=["quantization_calib"]
        )
        chunks.append(int8_chunk)
        
        return chunks
    
    def _create_chunk_from_data(self, chunk_id: str, name: str, chunk_type: str,
                              data: Dict[str, Any], output_dir: str, priority: int) -> LoadChunk:
        """Create a chunk from data dictionary."""
        # Convert data to bytes (simplified - would use proper serialization)
        chunk_data = json.dumps({k: v.tolist() if hasattr(v, 'tolist') else v 
                               for k, v in data.items()}).encode()
        
        file_path = os.path.join(output_dir, f"{chunk_id}.bin")
        with open(file_path, 'wb') as f:
            f.write(chunk_data)
        
        return LoadChunk(
            chunk_id=chunk_id,
            name=name,
            chunk_type=chunk_type,
            priority=priority,
            size_bytes=len(chunk_data),
            dependencies=[],
            file_path=file_path
        )
    
    def _calculate_layer_priority(self, layer_name: str) -> int:
        """Calculate loading priority for a layer."""
        # Higher priority for early layers (input processing)
        # This is a simplified heuristic
        if "input" in layer_name.lower() or "conv1" in layer_name.lower():
            return 1
        elif "conv" in layer_name.lower() and "2" in layer_name:
            return 2
        elif "conv" in layer_name.lower() and "3" in layer_name:
            return 3
        elif "fc" in layer_name.lower() or "linear" in layer_name.lower():
            return 4
        else:
            return 5
    
    def _create_loading_plan(self, chunks: List[LoadChunk]) -> Dict[str, Any]:
        """Create loading plan for chunks."""
        
        # Group chunks by priority
        priority_groups = defaultdict(list)
        for chunk in chunks:
            priority_groups[chunk.priority].append(chunk.chunk_id)
        
        # Create phases
        phases = []
        for priority in sorted(priority_groups.keys()):
            phases.append({
                "phase": len(phases),
                "priority": priority,
                "chunks": priority_groups[priority],
                "description": f"Load priority {priority} chunks"
            })
        
        return {
            "total_phases": len(phases),
            "phases": phases,
            "dependencies": {chunk.chunk_id: chunk.dependencies for chunk in chunks},
            "critical_path": self._identify_critical_path(chunks)
        }
    
    def _identify_critical_path(self, chunks: List[LoadChunk]) -> List[str]:
        """Identify critical path for model loading."""
        # Find the longest dependency chain
        dependency_graph = {chunk.chunk_id: set(chunk.dependencies) for chunk in chunks}
        
        # Simple topological sort to find critical path
        in_degree = {chunk.chunk_id: len(chunk.dependencies) for chunk in chunks}
        queue = [chunk_id for chunk_id, degree in in_degree.items() if degree == 0]
        critical_path = []
        
        while queue:
            current = queue.pop(0)
            critical_path.append(current)
            
            # Find chunks that depend on current
            for chunk_id, deps in dependency_graph.items():
                if current in deps:
                    in_degree[chunk_id] -= 1
                    if in_degree[chunk_id] == 0:
                        queue.append(chunk_id)
        
        return critical_path
    
    def _estimate_loading_time(self, chunks: List[LoadChunk]) -> float:
        """Estimate total loading time."""
        total_size_mb = sum(chunk.size_bytes for chunk in chunks) / 1024 / 1024
        
        # Assume loading speed based on configuration
        loading_speed_mbps = 10 if self.config.loading_threads > 1 else 5
        
        return total_size_mb / loading_speed_mbps
    
    def _save_chunk_metadata(self, output_dir: str, chunks: List[LoadChunk], 
                           loading_plan: Dict[str, Any]):
        """Save chunk metadata for loading."""
        metadata = {
            "version": "1.0",
            "created": time.time(),
            "config": asdict(self.config),
            "chunks": [asdict(chunk) for chunk in chunks],
            "loading_plan": loading_plan
        }
        
        metadata_path = os.path.join(output_dir, "chunk_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Chunk metadata saved to {metadata_path}")
    
    def _create_loader_config(self, output_dir: str, chunks: List[LoadChunk],
                            loading_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create loader configuration file."""
        config = {
            "model_config": {
                "chunk_size_mb": self.config.chunk_size_mb,
                "max_memory_mb": self.config.max_memory_mb,
                "total_chunks": len(chunks),
                "total_size_mb": sum(chunk.size_bytes for chunk in chunks) / 1024 / 1024
            },
            "loading_config": {
                "loading_threads": self.config.loading_threads,
                "enable_priorities": self.config.enable_priorities,
                "enable_background_loading": self.config.enable_background_loading,
                "cache_loaded_chunks": self.config.cache_loaded_chunks
            },
            "performance_config": {
                "prewarm_critical_chunks": self.config.prewarm_critical_chunks,
                "validate_chunks": self.config.validate_chunks
            }
        }
        
        config_path = os.path.join(output_dir, "loader_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def load_model_progressive(self, chunks_dir: str, 
                             on_demand_handler: Optional[callable] = None) -> Dict[str, Any]:
        """
        Load model progressively with performance monitoring.
        
        Args:
            chunks_dir: Directory containing model chunks
            on_demand_handler: Optional handler for on-demand loading
            
        Returns:
            Loading results and statistics
        """
        logger.info(f"Starting progressive loading from {chunks_dir}")
        
        # Load metadata
        metadata_path = os.path.join(chunks_dir, "chunk_metadata.json")
        if not os.path.exists(metadata_path):
            return {"error": f"Metadata file not found: {metadata_path}"}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        chunks = [LoadChunk(**chunk_info) for chunk_info in metadata["chunks"]]
        loading_plan = metadata["loading_plan"]
        
        # Initialize loading statistics
        stats = LoadingStats(
            total_chunks=len(chunks),
            loaded_chunks=0,
            failed_chunks=0,
            current_phase="initializing",
            memory_used_mb=0,
            loading_speed_mbps=0,
            estimated_time_remaining=0,
            start_time=time.time()
        )
        
        # Start loading
        results = {
            "status": "success",
            "chunks_dir": chunks_dir,
            "statistics": asdict(stats),
            "loaded_chunks": [],
            "failed_chunks": []
        }
        
        try:
            # Load critical path first
            critical_path = loading_plan["critical_path"]
            logger.info(f"Loading critical path: {len(critical_path)} chunks")
            
            critical_chunks = [chunk for chunk in chunks if chunk.chunk_id in critical_path]
            critical_results = self._load_chunk_batch(critical_chunks, chunks_dir, "critical")
            results["loaded_chunks"].extend(critical_results["loaded"])
            results["failed_chunks"].extend(critical_results["failed"])
            
            stats.loaded_chunks = len(results["loaded_chunks"])
            stats.failed_chunks = len(results["failed_chunks"])
            stats.current_phase = "critical_loaded"
            
            # Load remaining chunks in phases
            for phase in loading_plan["phases"]:
                if phase["phase"] == 0:  # Skip critical path phase
                    continue
                
                logger.info(f"Loading phase {phase['phase']}: {len(phase['chunks'])} chunks")
                stats.current_phase = f"phase_{phase['phase']}"
                
                # Get chunks for this phase
                phase_chunks = [chunk for chunk in chunks if chunk.chunk_id in phase["chunks"]]
                
                # Load phase chunks
                phase_results = self._load_chunk_batch(phase_chunks, chunks_dir, f"phase_{phase['phase']}")
                results["loaded_chunks"].extend(phase_results["loaded"])
                results["failed_chunks"].extend(phase_results["failed"])
                
                stats.loaded_chunks = len(results["loaded_chunks"])
                stats.failed_chunks = len(results["failed_chunks"])
            
            # Calculate final statistics
            total_time = time.time() - stats.start_time
            stats.current_phase = "completed"
            results["statistics"] = asdict(stats)
            results["total_time"] = total_time
            results["load_success_rate"] = stats.loaded_chunks / stats.total_chunks
            
            logger.info(f"Progressive loading completed: {stats.loaded_chunks}/{stats.total_chunks} chunks loaded "
                       f"in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Progressive loading failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        return results
    
    def _load_chunk_batch(self, chunks: List[LoadChunk], chunks_dir: str, 
                        batch_name: str) -> Dict[str, List[str]]:
        """Load a batch of chunks."""
        loaded = []
        failed = []
        
        # Create thread pool for parallel loading
        with ThreadPoolExecutor(max_workers=self.config.loading_threads) as executor:
            # Submit loading tasks
            future_to_chunk = {
                executor.submit(self._load_single_chunk, chunk, chunks_dir): chunk 
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    if result["success"]:
                        loaded.append(chunk.chunk_id)
                        if self.config.cache_loaded_chunks:
                            self.chunk_cache[chunk.chunk_id] = result["data"]
                    else:
                        failed.append(chunk.chunk_id)
                        logger.warning(f"Failed to load chunk {chunk.chunk_id}: {result.get('error')}")
                except Exception as e:
                    failed.append(chunk.chunk_id)
                    logger.error(f"Chunk loading error for {chunk.chunk_id}: {e}")
        
        return {"loaded": loaded, "failed": failed}
    
    def _load_single_chunk(self, chunk: LoadChunk, chunks_dir: str) -> Dict[str, Any]:
        """Load a single chunk."""
        start_time = time.time()
        
        try:
            # Check if chunk is already cached
            if chunk.chunk_id in self.chunk_cache:
                return {
                    "success": True,
                    "chunk_id": chunk.chunk_id,
                    "load_time": 0.0,
                    "data": self.chunk_cache[chunk.chunk_id],
                    "cached": True
                }
            
            # Load chunk from file
            if chunk.file_path and os.path.exists(chunk.file_path):
                with open(chunk.file_path, 'rb') as f:
                    chunk_data = f.read()
            else:
                # Handle different chunk types
                chunk_data = self._load_chunk_by_type(chunk, chunks_dir)
            
            # Validate chunk if enabled
            if self.config.validate_chunks:
                if not self._validate_chunk(chunk, chunk_data):
                    return {
                        "success": False,
                        "chunk_id": chunk.chunk_id,
                        "error": "Chunk validation failed"
                    }
            
            # Process chunk data based on type
            processed_data = self._process_chunk_data(chunk, chunk_data)
            
            load_time = time.time() - start_time
            
            return {
                "success": True,
                "chunk_id": chunk.chunk_id,
                "load_time": load_time,
                "data": processed_data,
                "cached": False
            }
            
        except Exception as e:
            return {
                "success": False,
                "chunk_id": chunk.chunk_id,
                "error": str(e)
            }
    
    def _load_chunk_by_type(self, chunk: LoadChunk, chunks_dir: str) -> bytes:
        """Load chunk data based on chunk type."""
        if chunk.chunk_type == "weights":
            # For PyTorch weight chunks
            file_path = os.path.join(chunks_dir, f"{chunk.chunk_id}.pt")
            if os.path.exists(file_path):
                data = torch.load(file_path, map_location='cpu')
                return json.dumps(data).encode()
        elif chunk.chunk_type == "quantization":
            # For quantization chunks
            file_path = os.path.join(chunks_dir, f"{chunk.chunk_id}_quantized.bin")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return f.read()
        
        # Default: try to load from chunk_id.bin
        file_path = os.path.join(chunks_dir, f"{chunk.chunk_id}.bin")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return f.read()
        
        raise FileNotFoundError(f"Chunk file not found: {chunk.chunk_id}")
    
    def _validate_chunk(self, chunk: LoadChunk, data: bytes) -> bool:
        """Validate loaded chunk data."""
        try:
            # Basic size validation
            if len(data) == 0:
                return False
            
            # Type-specific validation
            if chunk.chunk_type == "weights":
                # Validate weight data structure
                if data.startswith(b'{"'):
                    # JSON format
                    parsed = json.loads(data.decode())
                    if not isinstance(parsed, dict):
                        return False
            elif chunk.chunk_type == "quantization":
                # Validate quantization data
                if len(data) < 100:  # Minimum size for calibration/quantization data
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _process_chunk_data(self, chunk: LoadChunk, data: bytes) -> Any:
        """Process chunk data based on type."""
        if chunk.chunk_type == "weights":
            # Parse PyTorch weights
            try:
                if data.startswith(b'{"'):
                    return json.loads(data.decode())
                else:
                    return data  # Raw binary data
            except:
                return data
        elif chunk.chunk_type == "ops":
            # Parse operations
            try:
                return json.loads(data.decode())
            except:
                return data
        else:
            # Return raw data
            return data


class MemoryMonitor:
    """Monitor memory usage during loading."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        self.memory_timeline = []
    
    def update(self, memory_mb: float):
        """Update memory monitoring."""
        self.current_memory = memory_mb
        self.peak_memory = max(self.peak_memory, memory_mb)
        self.memory_timeline.append({
            "time": time.time(),
            "memory_mb": memory_mb
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "peak_memory_mb": self.peak_memory,
            "current_memory_mb": self.current_memory,
            "memory_timeline": self.memory_timeline
        }
