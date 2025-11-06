import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import json
import time
from enum import Enum

logger = logging.getLogger(__name__)

class QuantizationType(Enum):
    """Supported quantization types."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QAT = "qat"  # Quantization Aware Training
    FP16 = "fp16"
    INT8 = "int8"

class CalibrationMethod(Enum):
    """Calibration methods for static quantization."""
    MIN_MAX = "min_max"
    PERCENTILE = "percentile"
    ENTROPY = "entropy"
    MSE = "mse"

class ModelQuantizer:
    """Advanced model quantization for edge deployment."""
    
    def __init__(self, 
                 calibration_samples: int = 100,
                 calibration_method: CalibrationMethod = CalibrationMethod.MIN_MAX,
                 preserve_accuracy: bool = True,
                 verbose: bool = True):
        """
        Initialize the model quantizer.
        
        Args:
            calibration_samples: Number of samples for calibration
            calibration_method: Method for calibration
            preserve_accuracy: Whether to preserve model accuracy
            verbose: Whether to print progress information
        """
        self.calibration_samples = calibration_samples
        self.calibration_method = calibration_method
        self.preserve_accuracy = preserve_accuracy
        self.verbose = verbose
        self.calibration_data = []
        self.quantization_stats = {}
        
    def add_calibration_data(self, data: Union[torch.Tensor, np.ndarray, List]):
        """
        Add calibration data for static quantization.
        
        Args:
            data: Calibration data (images, features, etc.)
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        if isinstance(data, (list, tuple)):
            for item in data:
                if isinstance(item, np.ndarray):
                    item = torch.from_numpy(item)
                self.calibration_data.append(item)
        else:
            self.calibration_data.append(data)
            
        if self.verbose:
            logger.info(f"Added {len(data) if hasattr(data, '__len__') else 1} calibration samples. "
                       f"Total: {len(self.calibration_data)}")
    
    def _get_activation_stats(self, model: nn.Module, 
                            sample_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get activation statistics for quantization.
        
        Args:
            model: PyTorch model
            sample_input: Sample input for activation capture
            
        Returns:
            Dictionary of activation statistics
        """
        # Hook to capture activations
        activation_stats = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = {
                        'min': output.detach().min().item(),
                        'max': output.detach().max().item(),
                        'mean': output.detach().mean().item(),
                        'std': output.detach().std().item()
                    }
            return hook
        
        # Register hooks on relevant layers
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        # Forward pass to collect statistics
        model.eval()
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activation_stats
    
    def _calibrate_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Calibrate model using collected data.
        
        Args:
            model: PyTorch model to calibrate
            
        Returns:
            Dictionary of calibration statistics
        """
        if not self.calibration_data:
            raise ValueError("No calibration data available. Add data first using add_calibration_data().")
        
        model.eval()
        all_stats = []
        
        if self.verbose:
            logger.info(f"Calibrating model with {len(self.calibration_data)} samples...")
        
        with torch.no_grad():
            for i, data in enumerate(self.calibration_data[:self.calibration_samples]):
                if isinstance(data, (list, tuple)):
                    data = data[0]  # Take first element if batch
                
                # Ensure data is on correct device
                if hasattr(model, 'device'):
                    data = data.to(model.device)
                
                # Get activation statistics
                stats = self._get_activation_stats(model, data)
                all_stats.append(stats)
                
                if self.verbose and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{min(len(self.calibration_data), self.calibration_samples)} samples")
        
        # Aggregate statistics
        aggregated_stats = {}
        layer_names = set()
        for stats in all_stats:
            layer_names.update(stats.keys())
        
        for layer_name in layer_names:
            layer_stats = [s[layer_name] for s in all_stats if layer_name in s]
            
            if self.calibration_method == CalibrationMethod.MIN_MAX:
                # Use global min/max across all samples
                min_vals = [s['min'] for s in layer_stats]
                max_vals = [s['max'] for s in layer_stats]
                aggregated_stats[layer_name] = {
                    'min': min(min_vals),
                    'max': max(max_vals),
                    'method': 'min_max'
                }
            
            elif self.calibration_method == CalibrationMethod.PERCENTILE:
                # Use percentiles for more robust quantization
                all_values = []
                for s in layer_stats:
                    # Approximate percentile using mean and std
                    mean, std = s['mean'], s['std']
                    all_values.extend([
                        mean - 2 * std,  # 2.5th percentile approx
                        mean + 2 * std  # 97.5th percentile approx
                    ])
                
                p_low, p_high = np.percentile(all_values, [1, 99])
                aggregated_stats[layer_name] = {
                    'min': p_low,
                    'max': p_high,
                    'method': 'percentile'
                }
        
        return aggregated_stats
    
    def quantize_model(self, 
                      model: nn.Module, 
                      quantization_type: QuantizationType = QuantizationType.STATIC,
                      precision: str = 'int8') -> nn.Module:
        """
        Quantize a PyTorch model for edge deployment.
        
        Args:
            model: PyTorch model to quantize
            quantization_type: Type of quantization to apply
            precision: Target precision ('int8', 'fp16')
            
        Returns:
            Quantized model
        """
        if precision not in ['int8', 'fp16']:
            raise ValueError(f"Unsupported precision: {precision}. Use 'int8' or 'fp16'.")
        
        if self.verbose:
            logger.info(f"Starting {quantization_type.value} quantization to {precision}")
        
        start_time = time.time()
        
        if quantization_type == QuantizationType.FP16:
            return self._quantize_fp16(model)
        elif quantization_type == QuantizationType.DYNAMIC:
            return self._quantize_dynamic(model)
        elif quantization_type == QuantizationType.STATIC:
            return self._quantize_static(model, precision)
        elif quantization_type == QuantizationType.QAT:
            return self._quantize_qat(model, precision)
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
    
    def _quantize_fp16(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16 precision."""
        if self.verbose:
            logger.info("Converting model to FP16...")
        
        # Convert to half precision
        model = model.half()
        
        # Convert back specific layers that need FP32
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.float()
        
        self.quantization_stats['type'] = 'fp16'
        self.quantization_stats['original_size_mb'] = self._get_model_size_mb(model)
        
        if self.verbose:
            logger.info(f"FP16 conversion completed in {time.time() - start_time:.2f}s")
        
        return model
    
    def _quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        if self.verbose:
            logger.info("Applying dynamic quantization...")
        
        # Dynamic quantization works best on Linear and Conv layers
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d}, 
            dtype=torch.qint8
        )
        
        self.quantization_stats['type'] = 'dynamic_int8'
        self.quantization_stats['original_size_mb'] = self._get_model_size_mb(model)
        
        if self.verbose:
            logger.info(f"Dynamic quantization completed in {time.time() - start_time:.2f}s")
        
        return quantized_model
    
    def _quantize_static(self, model: nn.Module, precision: str) -> nn.Module:
        """Apply static quantization with calibration."""
        if self.verbose:
            logger.info(f"Applying static {precision} quantization with calibration...")
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm' if 'cpu' in str(model.device) else 'qnnpack')
        
        # Insert observers
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate using collected data
        if not self.calibration_data:
            warnings.warn("No calibration data provided. Using default calibration.")
        else:
            calibration_stats = self._calibrate_model(model)
            
            # Apply calibration statistics to observers
            for name, module in model.named_modules():
                if hasattr(module, 'qconfig') and module.qconfig:
                    if name in calibration_stats:
                        stats = calibration_stats[name]
                        # Update observer ranges if possible
                        if hasattr(module, 'weight_fake_quant') and hasattr(module.weight_fake_quant, 'scale'):
                            # This is a simplified approach - real implementation would need more sophisticated observer updates
                            pass
        
        # Convert to quantized model
        if precision == 'int8':
            quantized_model = torch.quantization.convert(model, inplace=False)
        elif precision == 'fp16':
            quantized_model = model.half()
        
        self.quantization_stats.update({
            'type': f'static_{precision}',
            'original_size_mb': self._get_model_size_mb(model),
            'calibration_samples': len(self.calibration_data),
            'calibration_method': self.calibration_method.value
        })
        
        if self.verbose:
            logger.info(f"Static {precision} quantization completed in {time.time() - start_time:.2f}s")
        
        return quantized_model
    
    def _quantize_qat(self, model: nn.Module, precision: str) -> nn.Module:
        """Apply Quantization Aware Training."""
        if self.verbose:
            logger.info("Setting up Quantization Aware Training...")
        
        # This is a simplified QAT setup - real implementation would require training
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare model for QAT
        torch.quantization.prepare_qat(model, inplace=True)
        
        self.quantization_stats['type'] = f'qat_{precision}'
        self.quantization_stats['note'] = 'Model prepared for QAT - requires training'
        
        if self.verbose:
            logger.info("QAT setup completed - model ready for training")
        
        return model
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in MB."""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get detailed information about the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = self._get_model_size_mb(model)
        
        # Count different layer types
        layer_counts = {}
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'layer_counts': layer_counts,
            'quantization_stats': self.quantization_stats
        }
    
    def save_quantization_config(self, filepath: Union[str, Path]):
        """
        Save quantization configuration and statistics.
        
        Args:
            filepath: Path to save the configuration
        """
        config = {
            'calibration_samples': self.calibration_samples,
            'calibration_method': self.calibration_method.value,
            'preserve_accuracy': self.preserve_accuracy,
            'quantization_stats': self.quantization_stats,
            'calibration_data_count': len(self.calibration_data)
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        
        if self.verbose:
            logger.info(f"Quantization configuration saved to {filepath}")
    
    def load_quantization_config(self, filepath: Union[str, Path]):
        """
        Load quantization configuration.
        
        Args:
            filepath: Path to load the configuration from
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.calibration_samples = config.get('calibration_samples', 100)
        self.calibration_method = CalibrationMethod(config.get('calibration_method', 'min_max'))
        self.preserve_accuracy = config.get('preserve_accuracy', True)
        self.quantization_stats = config.get('quantization_stats', {})
        
        if self.verbose:
            logger.info(f"Quantization configuration loaded from {filepath}")
    
    def benchmark_quantization(self, 
                             model: nn.Module, 
                             test_input: torch.Tensor,
                             original_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Benchmark the impact of quantization on model performance.
        
        Args:
            model: Original model
            test_input: Test input data
            original_metrics: Original model metrics (latency, accuracy, etc.)
            
        Returns:
            Dictionary containing benchmark results
        """
        if self.verbose:
            logger.info("Starting quantization benchmark...")
        
        # Original model performance
        model.eval()
        with torch.no_grad():
            start_time = time.time()
            original_output = model(test_input)
            original_time = time.time() - start_time
        
        # Quantized model performance
        quantized_model = self.quantize_model(model, QuantizationType.STATIC, 'int8')
        quantized_model.eval()
        
        with torch.no_grad():
            start_time = time.time()
            quantized_output = quantized_model(test_input)
            quantized_time = time.time() - start_time
        
        # Calculate metrics
        original_size = self._get_model_size_mb(model)
        quantized_size = self._get_model_size_mb(quantized_model)
        
        # Accuracy impact (if outputs are comparable)
        try:
            if isinstance(original_output, (torch.Tensor, np.ndarray)) and isinstance(quantized_output, (torch.Tensor, np.ndarray)):
                if isinstance(original_output, torch.Tensor):
                    original_output = original_output.numpy()
                if isinstance(quantized_output, torch.Tensor):
                    quantized_output = quantized_output.numpy()
                
                # Calculate relative error
                relative_error = np.mean(np.abs(original_output - quantized_output) / (np.abs(original_output) + 1e-8))
            else:
                relative_error = None
        except:
            relative_error = None
        
        results = {
            'original_latency_ms': original_time * 1000,
            'quantized_latency_ms': quantized_time * 1000,
            'speedup_factor': original_time / quantized_time if quantized_time > 0 else 0,
            'original_size_mb': original_size,
            'quantized_size_mb': quantized_size,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100 if original_size > 0 else 0,
            'relative_error': relative_error,
            'quantization_type': self.quantization_stats.get('type', 'unknown'),
            'model_info': self.get_model_info(model)
        }
        
        if self.verbose:
            logger.info(f"Quantization benchmark completed:")
            logger.info(f"  Speedup: {results['speedup_factor']:.2f}x")
            logger.info(f"  Size reduction: {results['size_reduction_percent']:.1f}%")
            if relative_error is not None:
                logger.info(f"  Relative error: {relative_error:.6f}")
        
        return results