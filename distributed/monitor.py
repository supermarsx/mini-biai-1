import torch
import torch.distributed as dist
import logging
import time
import json
import psutil
import GPUtil
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
from datetime import datetime
import threading
import os
import sys
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Real-time training monitoring and performance tracking system.
    
    This class provides comprehensive monitoring capabilities for distributed training:
    - Real-time metrics tracking
    - System resource monitoring (CPU, GPU, memory)
    - Training performance analytics
    - Anomaly detection and alerts
    - Custom metric registration
    - Performance profiling and optimization suggestions
    - Logging and visualization support
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        save_logs: bool = True,
        log_dir: str = './logs',
        track_gpu_memory: bool = True,
        track_cpu_memory: bool = True,
        track_network: bool = True,
        max_history: int = 1000,
        enable_alerts: bool = True
    ):
        """Initialize training monitor.
        
        Args:
            log_interval: Interval for logging metrics
            save_logs: Whether to save logs to file
            log_dir: Directory for log files
            track_gpu_memory: Track GPU memory usage
            track_cpu_memory: Track CPU memory usage
            track_network: Track network I/O
            max_history: Maximum number of history entries to keep
            enable_alerts: Enable anomaly detection and alerts
        """
        self.log_interval = log_interval
        self.save_logs = save_logs
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.track_gpu_memory = track_gpu_memory
        self.track_cpu_memory = track_cpu_memory
        self.track_network = track_network
        self.max_history = max_history
        self.enable_alerts = enable_alerts
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.last_log_step = 0
        self.last_log_time = time.time()
        
        # Metrics tracking
        self.metrics = {}
        self.metric_history = defaultdict(lambda: deque(maxlen=max_history))
        self.custom_metrics = {}
        self.metric_functions = {}
        
        # System monitoring
        self.system_stats = {
            'cpu_usage': deque(maxlen=max_history),
            'cpu_memory': deque(maxlen=max_history),
            'gpu_memory': deque(maxlen=max_history),
            'gpu_utilization': deque(maxlen=max_history),
            'network_io': deque(maxlen=max_history)
        }
        
        # Callbacks
        self.callbacks = defaultdict(list)
        self.alert_callbacks = []
        
        # Performance analysis
        self.performance_stats = {}
        self.anomalies = []
        
        # Threading for async monitoring
        self.monitor_thread = None
        self.running = False
        self.monitor_interval = 1.0  # seconds
        
        # Initialize monitoring
        self._setup_logging()
        self._initialize_system_monitoring()
        
        logger.info(f"Initialized TrainingMonitor with log_interval={log_interval}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        if self.save_logs:
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_file = self.log_dir / f"training_log_{timestamp}.json"
            self.metric_log_file = self.log_dir / f"metrics_{timestamp}.json"
            
            logger.info(f"Logs will be saved to: {self.log_dir}")
    
    def _initialize_system_monitoring(self):
        """Initialize system monitoring components."""
        # Check GPU availability
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            try:
                self.gpus = GPUtil.getGPUs()
                self.gpu_count = len(self.gpus)
                logger.info(f"Found {self.gpu_count} GPU(s)")
            except Exception as e:
                logger.warning(f"Could not initialize GPU monitoring: {e}")
                self.gpu_count = 0
        else:
            self.gpu_count = 0
            logger.info("No GPU detected")
        
        # Check network interfaces
        if self.track_network:
            try:
                self.net_io_before = psutil.net_io_counters()
            except Exception as e:
                logger.warning(f"Could not initialize network monitoring: {e}")
                self.track_network = False
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started training monitoring")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.running:
            return
        
        self.running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("Stopped training monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._collect_system_stats()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_system_stats(self):
        """Collect system statistics."""
        current_time = time.time()
        
        # CPU statistics
        if self.track_cpu_memory:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                self.system_stats['cpu_usage'].append((current_time, cpu_percent))
                self.system_stats['cpu_memory'].append((current_time, memory.percent))
            except Exception as e:
                logger.debug(f"Could not collect CPU stats: {e}")
        
        # GPU statistics
        if self.track_gpu_memory and self.has_gpu:
            try:
                if self.gpus:
                    for i, gpu in enumerate(self.gpus):
                        gpu_key = f'gpu_{i}'
                        if gpu_key not in self.system_stats:
                            self.system_stats[gpu_key] = deque(maxlen=self.max_history)
                        
                        self.system_stats[gpu_key].append((
                            current_time,
                            {
                                'memory_used': gpu.memoryUsed,
                                'memory_total': gpu.memoryTotal,
                                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                                'temperature': gpu.temperature,
                                'utilization': gpu.load * 100
                            }
                        ))
            except Exception as e:
                logger.debug(f"Could not collect GPU stats: {e}")
        
        # Network statistics
        if self.track_network:
            try:
                net_io = psutil.net_io_counters()
                if hasattr(self, 'net_io_before'):
                    bytes_sent = net_io.bytes_sent - self.net_io_before.bytes_sent
                    bytes_recv = net_io.bytes_recv - self.net_io_before.bytes_recv
                    
                    self.system_stats['network_io'].append((
                        current_time,
                        {
                            'bytes_sent': bytes_sent,
                            'bytes_recv': bytes_recv
                        }
                    ))
                    
                    self.net_io_before = net_io
            except Exception as e:
                logger.debug(f"Could not collect network stats: {e}")
    
    def track_metric(self, name: str, value: float, step: Optional[int] = None):
        """Track a training metric.
        
        Args:
            name: Name of the metric
            value: Metric value
            step: Training step (uses current step if None)
        """
        if step is None:
            step = self.step
        
        # Store current value
        if name not in self.metrics:
            self.metrics[name] = value
        else:
            # Update with exponential moving average
            alpha = 0.1
            self.metrics[name] = alpha * value + (1 - alpha) * self.metrics[name]
        
        # Add to history
        current_time = time.time()
        self.metric_history[name].append((current_time, value, step))
        
        # Check for anomalies
        if self.enable_alerts:
            self._check_anomalies(name, value)
        
        # Trigger callbacks
        self._trigger_callbacks(name, {'value': value, 'step': step})
    
    def add_custom_metric(
        self,
        name: str,
        function: Callable,
        interval: int = 1
    ):
        """Add a custom metric that gets computed periodically.
        
        Args:
            name: Name of the custom metric
            function: Function to compute the metric
            interval: Compute every N steps
        """
        self.custom_metrics[name] = {
            'function': function,
            'interval': interval,
            'last_computed': -1
        }
    
    def compute_custom_metrics(self):
        """Compute all custom metrics that are due."""
        current_time = time.time()
        
        for name, config in self.custom_metrics.items():
            if self.step - config['last_computed'] >= config['interval']:
                try:
                    value = config['function']()
                    self.track_metric(name, value)
                    config['last_computed'] = self.step
                except Exception as e:
                    logger.error(f"Error computing custom metric {name}: {e}")
    
    def _check_anomalies(self, name: str, value: float):
        """Check for anomalies in metrics."""
        history = list(self.metric_history[name])
        
        if len(history) < 10:  # Need at least 10 points for anomaly detection
            return
        
        # Extract values (excluding current)
        values = [h[1] for h in history[:-1]]
        
        # Check for NaN or infinite values
        if not (torch.isfinite(torch.tensor(value)).item()):
            anomaly = {
                'metric': name,
                'value': value,
                'step': self.step,
                'type': 'invalid_value',
                'timestamp': datetime.now().isoformat()
            }
            self.anomalies.append(anomaly)
            logger.warning(f"Anomaly detected in {name}: invalid value {value}")
        
        # Check for sudden jumps (simple threshold-based)
        if len(values) > 0:
            prev_value = values[-1]
            if abs(value - prev_value) / max(abs(prev_value), 1e-6) > 10.0:  # 10x change
                anomaly = {
                    'metric': name,
                    'value': value,
                    'previous_value': prev_value,
                    'step': self.step,
                    'type': 'sudden_change',
                    'timestamp': datetime.now().isoformat()
                }
                self.anomalies.append(anomaly)
                logger.warning(f"Anomaly detected in {name}: sudden change from {prev_value} to {value}")
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for specific events.
        
        Args:
            event: Event name (e.g., 'step', 'epoch', 'metric_update')
            callback: Callback function
        """
        self.callbacks[event].append(callback)
    
    def add_alert_callback(self, callback: Callable):
        """Add a callback for anomaly alerts.
        
        Args:
            callback: Callback function that takes anomaly dict
        """
        self.alert_callbacks.append(callback)
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]):
        """Trigger callbacks for an event.
        
        Args:
            event: Event name
            data: Event data
        """
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error in callback for {event}: {e}")
    
    def log_metrics(self, force: bool = False):
        """Log current metrics.
        
        Args:
            force: Force logging regardless of interval
        """
        if not force and self.step - self.last_log_step < self.log_interval:
            return
        
        current_time = time.time()
        elapsed_time = current_time - self.last_log_time
        
        # Calculate performance metrics
        steps_per_second = (self.step - self.last_log_step) / elapsed_time if elapsed_time > 0 else 0
        samples_per_second = steps_per_second * self._get_batch_size()
        
        # Prepare log data
        log_data = {
            'step': self.step,
            'epoch': self.epoch,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'steps_per_second': steps_per_second,
            'samples_per_second': samples_per_second,
            'metrics': dict(self.metrics),
            'system_stats': self._get_latest_system_stats()
        }
        
        # Log to console
        self._console_log(log_data)
        
        # Save to file
        if self.save_logs:
            self._save_log_data(log_data)
        
        # Update last log time
        self.last_log_step = self.step
        self.last_log_time = current_time
    
    def _console_log(self, log_data: Dict[str, Any]):
        """Log data to console."""
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in log_data['metrics'].items()])
        system_str = self._format_system_stats(log_data['system_stats'])
        
        logger.info(
            f"Step {log_data['step']} | "
            f"Epoch {log_data['epoch']} | "
            f"Metrics: {metrics_str} | "
            f"Speed: {log_data['samples_per_second']:.1f} samples/s | "
            f"System: {system_str}"
        )
    
    def _format_system_stats(self, stats: Dict[str, Any]) -> str:
        """Format system stats for logging."""
        parts = []
        
        if 'cpu_memory' in stats:
            parts.append(f"CPU: {stats['cpu_memory']:.1f}%")
        
        if 'gpu_memory' in stats:
            parts.append(f"GPU: {stats['gpu_memory']:.1f}%")
        
        return ' '.join(parts)
    
    def _save_log_data(self, log_data: Dict[str, Any]):
        """Save log data to file."""
        try:
            # Append to metric log
            with open(self.metric_log_file, 'a') as f:
                f.write(json.dumps(log_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to save log data: {e}")
    
    def _get_latest_system_stats(self) -> Dict[str, Any]:
        """Get latest system statistics."""
        latest_stats = {}
        
        for key, history in self.system_stats.items():
            if history:
                latest_stats[key] = history[-1][1]
        
        return latest_stats
    
    def _get_batch_size(self) -> int:
        """Get current batch size (estimated)."""
        # This is a simple estimate - in practice, this might need to be passed in
        return 32  # Default assumption
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics.
        
        Returns:
            Dictionary containing performance statistics
        """
        current_time = time.time()
        total_time = current_time - self.start_time
        
        # Calculate training speed
        steps_per_second = self.step / total_time if total_time > 0 else 0
        
        # Analyze metric trends
        trends = self._analyze_trends()
        
        # System utilization
        utilization = self._calculate_utilization()
        
        # Performance score
        performance_score = self._calculate_performance_score()
        
        stats = {
            'training_stats': {
                'total_steps': self.step,
                'total_epochs': self.epoch,
                'total_time': total_time,
                'steps_per_second': steps_per_second,
                'estimated_time_to_completion': self._estimate_time_to_completion(steps_per_second)
            },
            'metric_trends': trends,
            'system_utilization': utilization,
            'performance_score': performance_score,
            'anomaly_count': len(self.anomalies),
            'recent_anomalies': self.anomalies[-5:] if self.anomalies else []
        }
        
        return stats
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in metrics."""
        trends = {}
        
        for name, history in self.metric_history.items():
            if len(history) < 10:
                continue
            
            values = [h[1] for h in history]
            steps = [h[2] for h in history]
            
            # Calculate trend (simple linear regression slope)
            if len(values) > 1:
                x = torch.tensor(steps, dtype=torch.float32)
                y = torch.tensor(values, dtype=torch.float32)
                
                # Simple slope calculation
                slope = torch.mean((x - torch.mean(x)) * (y - torch.mean(y))) / torch.var(x)
                
                # Determine trend direction
                if abs(slope.item()) < 1e-6:
                    direction = 'stable'
                elif slope.item() > 0:
                    direction = 'increasing'
                else:
                    direction = 'decreasing'
                
                trends[name] = {
                    'direction': direction,
                    'slope': slope.item(),
                    'current_value': values[-1],
                    'initial_value': values[0],
                    'change_percent': ((values[-1] - values[0]) / max(abs(values[0]), 1e-6)) * 100
                }
        
        return trends
    
    def _calculate_utilization(self) -> Dict[str, Any]:
        """Calculate system utilization statistics."""
        utilization = {}
        
        # CPU utilization
        if self.system_stats['cpu_usage']:
            cpu_values = [s[1] for s in self.system_stats['cpu_usage'][-60:]]  # Last minute
            utilization['cpu'] = {
                'current': cpu_values[-1] if cpu_values else 0,
                'average': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0
            }
        
        # GPU utilization
        gpu_util = []
        for key, history in self.system_stats.items():
            if key.startswith('gpu_') and 'utilization' in history[-1][1]:
                gpu_util.append(history[-1][1]['utilization'])
        
        if gpu_util:
            utilization['gpu'] = {
                'average': sum(gpu_util) / len(gpu_util),
                'max': max(gpu_util)
            }
        
        return utilization
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0.0
        
        # System utilization contribution (40%)
        if 'cpu' in self.system_stats['cpu_usage']:
            cpu_util = self.system_stats['cpu_usage'][-1][1] if self.system_stats['cpu_usage'] else 0
            cpu_score = min(cpu_util / 100.0, 1.0) * 40
            score += cpu_score
        
        # Training stability (30%) - based on anomaly count
        anomaly_penalty = min(len(self.anomalies) * 2, 30)
        stability_score = 30 - anomaly_penalty
        score += max(stability_score, 0)
        
        # Performance consistency (30%) - based on metric variance
        consistency_score = self._calculate_consistency_score()
        score += consistency_score
        
        return min(score, 100.0)
    
    def _calculate_consistency_score(self) -> float:
        """Calculate training consistency score."""
        total_score = 0.0
        metric_count = 0
        
        for name, history in self.metric_history.items():
            if len(history) < 10:
                continue
            
            values = [h[1] for h in history[-50:]]  # Last 50 points
            if len(values) > 1:
                variance = torch.var(torch.tensor(values)).item()
                mean_abs = torch.mean(torch.abs(torch.tensor(values))).item()
                
                # Lower variance relative to mean = higher consistency
                if mean_abs > 1e-6:
                    consistency = 1.0 / (1.0 + variance / (mean_abs ** 2))
                    total_score += consistency
                    metric_count += 1
        
        return (total_score / max(metric_count, 1)) * 30 if metric_count > 0 else 0
    
    def _estimate_time_to_completion(self, steps_per_second: float) -> Optional[float]:
        """Estimate time to completion based on current speed."""
        if steps_per_second <= 0:
            return None
        
        # This is a simple estimate - could be improved with more sophisticated models
        # Assume 100 epochs total
        estimated_total_steps = 100 * len(next(iter(self.metric_history.values())))
        
        if self.step >= estimated_total_steps:
            return 0.0
        
        remaining_steps = estimated_total_steps - self.step
        return remaining_steps / steps_per_second
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics.
        
        Returns:
            Current system statistics
        """
        return self._get_latest_system_stats()
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export all metrics to a file.
        
        Args:
            filename: Output filename (auto-generated if None)
        
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"exported_metrics_{timestamp}.json"
        else:
            filename = Path(filename)
        
        export_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_steps': self.step,
                'total_epochs': self.epoch,
                'total_time': time.time() - self.start_time
            },
            'metrics': dict(self.metrics),
            'metric_history': {k: list(v) for k, v in self.metric_history.items()},
            'system_stats': {k: list(v) for k, v in self.system_stats.items() if v},
            'anomalies': self.anomalies,
            'performance_stats': self.get_performance_stats()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Metrics exported to: {filename}")
            return str(filename)
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return ""
    
    def reset(self):
        """Reset all monitoring data."""
        self.step = 0
        self.epoch = 0
        self.metrics.clear()
        self.metric_history.clear()
        self.anomalies.clear()
        
        for key in self.system_stats:
            self.system_stats[key].clear()
        
        self.start_time = time.time()
        self.last_log_step = 0
        self.last_log_time = time.time()
        
        logger.info("Reset training monitor")
    
    def set_step(self, step: int):
        """Set current training step."""
        self.step = step
    
    def set_epoch(self, epoch: int):
        """Set current training epoch."""
        self.epoch = epoch
    
    def increment_step(self):
        """Increment training step."""
        self.step += 1
        self.compute_custom_metrics()
        self.log_metrics()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()

class MemoryProfiler:
    """Memory usage profiler for training optimization."""
    
    def __init__(self):
        self.memory_history = []
        self.peak_memory = 0
        self.oom_events = []
    
    def profile_training(self):
        """Profile memory usage during training."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before profiling
            
            # Get initial memory usage
            initial_memory = torch.cuda.memory_allocated()
            
            # Profile memory over time
            start_time = time.time()
            
            # This would typically be called during training
            # to monitor memory usage patterns
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            
            self.memory_history.append({
                'timestamp': time.time(),
                'memory_allocated': current_memory,
                'peak_memory': peak_memory,
                'initial_memory': initial_memory
            })
            
            self.peak_memory = max(self.peak_memory, peak_memory)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}
        
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
        cached_memory = torch.cuda.memory_reserved() - current_memory
        
        return {
            'current_memory_mb': current_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'reserved_memory_mb': reserved_memory / 1024 / 1024,
            'cached_memory_mb': cached_memory / 1024 / 1024,
            'memory_history_length': len(self.memory_history),
            'oom_events': len(self.oom_events)
        }
    
    def clear_memory(self):
        """Clear CUDA memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.memory_history.clear()
            self.peak_memory = 0

# Export main classes
__all__ = [
    'TrainingMonitor',
    'MemoryProfiler'
]