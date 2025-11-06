"""
Learning Metrics and Monitoring System.

This module provides comprehensive metrics collection, analysis, and monitoring
for spiking neural network learning processes.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import statistics
from concurrent.futures import ThreadPoolExecutor


class MetricType(Enum):
    """Types of metrics."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    LEARNING = "learning"
    STABILITY = "stability"
    CONVERGENCE = "convergence"


class AlertLevel(Enum):
    """Alert levels for monitoring."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """Configuration for metric alerts."""
    metric_name: str
    threshold: float
    comparison: str  # "gt", "lt", "eq", "ne"
    level: AlertLevel
    cooldown: float = 60.0  # seconds
    callback: Optional[Callable] = None


@dataclass
class MetricWindow:
    """Sliding window for metric calculation."""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add(self, value: float, timestamp: float):
        """Add value to window."""
        self.values.append(value)
        self.timestamps.append(timestamp)
        
    def get_windowed_stats(self, window_size: Optional[int] = None) -> Dict[str, float]:
        """Get statistics for recent values."""
        if not self.values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
            
        values = list(self.values)[-window_size:] if window_size else list(self.values)
        
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }


class AlertManager:
    """Manages alerts for metric thresholds."""
    
    def __init__(self):
        self.alerts: Dict[str, AlertConfig] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_time: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def add_alert(self, alert_config: AlertConfig):
        """Add alert configuration."""
        with self._lock:
            self.alerts[alert_config.metric_name] = alert_config
            
    def remove_alert(self, metric_name: str):
        """Remove alert configuration."""
        with self._lock:
            self.alerts.pop(metric_name, None)
            
    def check_alerts(self, metrics: Dict[str, float], timestamp: float) -> List[Dict[str, Any]]:
        """Check if any alerts should be triggered."""
        triggered_alerts = []
        
        with self._lock:
            for metric_name, alert_config in self.alerts.items():
                if metric_name not in metrics:
                    continue
                    
                value = metrics[metric_name]
                
                # Check if alert should trigger
                should_trigger = False
                if alert_config.comparison == "gt" and value > alert_config.threshold:
                    should_trigger = True
                elif alert_config.comparison == "lt" and value < alert_config.threshold:
                    should_trigger = True
                elif alert_config.comparison == "eq" and value == alert_config.threshold:
                    should_trigger = True
                elif alert_config.comparison == "ne" and value != alert_config.threshold:
                    should_trigger = True
                    
                if should_trigger:
                    # Check cooldown
                    last_time = self.last_alert_time.get(metric_name, 0)
                    if timestamp - last_time >= alert_config.cooldown:
                        self._trigger_alert(alert_config, value, timestamp)
                        self.last_alert_time[metric_name] = timestamp
                        triggered_alerts.append({
                            "metric": metric_name,
                            "value": value,
                            "threshold": alert_config.threshold,
                            "level": alert_config.level.value,
                            "timestamp": timestamp
                        })
                        
        return triggered_alerts
        
    def _trigger_alert(self, alert_config: AlertConfig, value: float, timestamp: float):
        """Trigger an alert."""
        alert_data = {
            "metric": alert_config.metric_name,
            "value": value,
            "threshold": alert_config.threshold,
            "level": alert_config.level.value,
            "timestamp": timestamp
        }
        
        self.alert_history.append(alert_data)
        
        # Execute callback if provided
        if alert_config.callback:
            try:
                alert_config.callback(alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
                
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            recent_alerts = [a for a in self.alert_history 
                           if time.time() - a["timestamp"] < 3600.0]
                           
            return {
                "total_alerts": len(self.alert_history),
                "recent_alerts": len(recent_alerts),
                "active_alerts": len(self.alerts),
                "alert_counts_by_level": {
                    level.value: sum(1 for a in recent_alerts if a["level"] == level.value)
                    for level in AlertLevel
                }
            }


class LearningMetrics:
    """
    Comprehensive learning metrics system.
    
    Collects, analyzes, and monitors various aspects of learning performance
    including convergence, stability, resource usage, and quality metrics.
    """
    
    def __init__(self, history_size: int = 10000):
        """
        Initialize learning metrics system.
        
        Args:
            history_size: Maximum number of metric points to store
        """
        self.history_size = history_size
        
        # Metric storage
        self.metric_windows: Dict[str, MetricWindow] = {}
        self.metric_points: Dict[str, deque] = {
            metric_type.value: deque(maxlen=history_size) 
            for metric_type in MetricType
        }
        
        # Alert management
        self.alert_manager = AlertManager()
        self._setup_default_alerts()
        
        # Analysis
        self.analysis_window = 1000  # Points for trend analysis
        self.trend_cache: Dict[str, float] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.last_update_time = time.time()
        self.update_frequency = 1.0  # seconds
        
    def _setup_default_alerts(self):
        """Setup default alert configurations."""
        # Performance alerts
        self.alert_manager.add_alert(AlertConfig(
            metric_name="learning_rate",
            threshold=0.01,
            comparison="lt",
            level=AlertLevel.WARNING,
            callback=self._handle_learning_rate_alert
        ))
        
        # Quality alerts
        self.alert_manager.add_alert(AlertConfig(
            metric_name="quality_score",
            threshold=0.3,
            comparison="lt",
            level=AlertLevel.ERROR,
            callback=self._handle_quality_alert
        ))
        
        # Resource alerts
        self.alert_manager.add_alert(AlertConfig(
            metric_name="memory_usage",
            threshold=0.9,
            comparison="gt",
            level=AlertLevel.WARNING,
            callback=self._handle_resource_alert
        ))
        
        # Stability alerts
        self.alert_manager.add_alert(AlertConfig(
            metric_name="weight_stability",
            threshold=0.5,
            comparison="lt",
            level=AlertLevel.WARNING,
            callback=self._handle_stability_alert
        ))
        
    def record_metric(self, 
                     name: str,
                     value: float,
                     metric_type: MetricType,
                     timestamp: Optional[float] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            timestamp: Timestamp (defaults to current time)
            metadata: Optional metadata
        """
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            # Add to metric window
            if name not in self.metric_windows:
                self.metric_windows[name] = MetricWindow()
                
            self.metric_windows[name].add(value, timestamp)
            
            # Add to type-specific storage
            metric_point = MetricPoint(
                timestamp=timestamp,
                value=value,
                metric_type=metric_type,
                metadata=metadata or {}
            )
            
            self.metric_points[metric_type.value].append(metric_point)
            
    def get_metric_statistics(self, 
                            metric_name: str,
                            window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            window_size: Number of recent values to analyze
            
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            if metric_name not in self.metric_windows:
                return {"error": f"Metric '{metric_name}' not found"}
                
            window = self.metric_windows[metric_name]
            stats = window.get_windowed_stats(window_size)
            
            # Calculate additional statistics
            values = list(window.values)[-window_size:] if window_size else list(window.values)
            
            if values:
                # Trend analysis
                trend = self._calculate_trend(values)
                
                # Percentiles
                percentiles = np.percentile(values, [25, 50, 75, 90, 95, 99])
                
                # Rate of change
                if len(values) >= 2:
                    rate_of_change = (values[-1] - values[0]) / len(values)
                else:
                    rate_of_change = 0.0
                    
                # Volatility (coefficient of variation)
                volatility = stats["std"] / max(1e-10, abs(stats["mean"]))
                
                stats.update({
                    "trend": trend,
                    "percentiles": {
                        "p25": percentiles[0],
                        "p50": percentiles[1],
                        "p75": percentiles[2],
                        "p90": percentiles[3],
                        "p95": percentiles[4],
                        "p99": percentiles[5]
                    },
                    "rate_of_change": rate_of_change,
                    "volatility": volatility,
                    "latest_value": values[-1],
                    "oldest_value": values[0] if values else 0.0
                })
                
            return stats
            
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of recent values."""
        if len(values) < 3:
            return 0.0
            
        # Use simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        return float(slope)
        
    def get_learning_quality_score(self) -> Dict[str, Any]:
        """
        Calculate overall learning quality score.
        
        Returns:
            Dictionary with quality assessment
        """
        with self._lock:
            # Get key metrics
            quality_metrics = {}
            
            for metric_name in ["quality_score", "learning_accuracy", "convergence_rate"]:
                if metric_name in self.metric_windows:
                    stats = self.metric_windows[metric_name].get_windowed_stats(50)
                    quality_metrics[metric_name] = stats["mean"]
                    
            # Calculate stability metrics
            stability_metrics = {}
            for metric_name in ["weight_stability", "performance_stability"]:
                if metric_name in self.metric_windows:
                    stats = self.metric_windows[metric_name].get_windowed_stats(50)
                    stability_metrics[metric_name] = stats["mean"]
                    
            # Calculate overall score
            quality_score = np.mean(list(quality_metrics.values())) if quality_metrics else 0.5
            stability_score = np.mean(list(stability_metrics.values())) if stability_metrics else 0.5
            
            overall_score = (quality_score * 0.6 + stability_score * 0.4)
            
            return {
                "overall_score": overall_score,
                "quality_component": quality_score,
                "stability_component": stability_score,
                "quality_metrics": quality_metrics,
                "stability_metrics": stability_metrics,
                "score_breakdown": {
                    "quality_weight": 0.6,
                    "stability_weight": 0.4
                }
            }
            
    def get_convergence_analysis(self) -> Dict[str, Any]:
        """Analyze learning convergence."""
        with self._lock:
            # Look for convergence in key metrics
            convergence_metrics = {}
            
            for metric_name in ["learning_error", "weight_variance", "performance_variance"]:
                if metric_name in self.metric_windows:
                    values = list(self.metric_windows[metric_name].values)
                    if len(values) >= 20:
                        # Check if metric is stabilizing
                        recent_std = np.std(values[-10:])
                        overall_std = np.std(values)
                        
                        # Converging if recent variance is much lower
                        convergence_ratio = recent_std / max(1e-10, overall_std)
                        
                        convergence_metrics[metric_name] = {
                            "converging": convergence_ratio < 0.5,
                            "convergence_ratio": convergence_ratio,
                            "recent_variance": recent_std,
                            "overall_variance": overall_std
                        }
                        
            # Overall convergence assessment
            converging_count = sum(1 for m in convergence_metrics.values() if m["converging"])
            total_metrics = len(convergence_metrics)
            
            convergence_status = "unknown"
            if total_metrics > 0:
                convergence_ratio = converging_count / total_metrics
                if convergence_ratio >= 0.7:
                    convergence_status = "converging"
                elif convergence_ratio >= 0.4:
                    convergence_status = "partially_converging"
                else:
                    convergence_status = "not_converging"
                    
            return {
                "status": convergence_status,
                "convergence_ratio": converging_count / max(1, total_metrics),
                "metrics": convergence_metrics,
                "total_assessed": total_metrics
            }
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        with self._lock:
            current_time = time.time()
            
            # Get recent metrics (last hour)
            recent_threshold = current_time - 3600.0
            
            # Performance metrics
            performance_stats = {}
            for metric_name in self.metric_windows:
                window = self.metric_windows[metric_name]
                recent_values = [v for v, t in zip(window.values, window.timestamps) 
                               if t >= recent_threshold]
                
                if recent_values:
                    performance_stats[metric_name] = {
                        "recent_mean": np.mean(recent_values),
                        "recent_count": len(recent_values),
                        "trend": self._calculate_trend(recent_values[-20:]) if len(recent_values) >= 5 else 0.0
                    }
                    
            # Alert statistics
            alert_stats = self.alert_manager.get_alert_statistics()
            
            # Quality assessment
            quality_assessment = self.get_learning_quality_score()
            
            # Convergence analysis
            convergence_analysis = self.get_convergence_analysis()
            
            # Overall health score
            health_components = []
            
            if quality_assessment["overall_score"] > 0:
                health_components.append(quality_assessment["overall_score"])
                
            if convergence_analysis["convergence_ratio"] > 0:
                health_components.append(convergence_analysis["convergence_ratio"])
                
            # Resource health (invert resource usage)
            resource_metrics = ["memory_usage", "cpu_usage"]
            resource_health = 1.0
            for metric_name in resource_metrics:
                if metric_name in self.metric_windows:
                    recent_usage = np.mean(list(self.metric_windows[metric_name].values)[-10:])
                    resource_health *= (1.0 - recent_usage)
                    
            if resource_metrics:
                health_components.append(resource_health)
                
            overall_health = np.mean(health_components) if health_components else 0.5
            
            return {
                "timestamp": current_time,
                "overall_health": overall_health,
                "quality_assessment": quality_assessment,
                "convergence_analysis": convergence_analysis,
                "performance_metrics": performance_stats,
                "alert_statistics": alert_stats,
                "resource_health": resource_health,
                "metrics_tracked": len(self.metric_windows),
                "report_period_hours": 1.0
            }
            
    def export_metrics(self, 
                      format: str = "json",
                      timeframe: Optional[Tuple[float, float]] = None) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format: Export format ("json", "csv")
            timeframe: Optional (start_time, end_time) tuple
            
        Returns:
            Exported metrics as string
        """
        with self._lock:
            export_data = {}
            
            for metric_name, window in self.metric_windows.items():
                values = list(window.values)
                timestamps = list(window.timestamps)
                
                # Apply timeframe filter
                if timeframe:
                    start_time, end_time = timeframe
                    filtered_data = [(v, t) for v, t in zip(values, timestamps)
                                   if start_time <= t <= end_time]
                    values, timestamps = zip(*filtered_data) if filtered_data else ([], [])
                    
                export_data[metric_name] = {
                    "values": values,
                    "timestamps": timestamps,
                    "statistics": self.get_metric_statistics(metric_name)
                }
                
            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            else:
                # Simple CSV export
                csv_lines = ["metric,timestamp,value"]
                for metric_name, data in export_data.items():
                    for value, timestamp in zip(data["values"], data["timestamps"]):
                        csv_lines.append(f"{metric_name},{timestamp},{value}")
                        
                return "\n".join(csv_lines)
                
    def check_alerts(self, timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Check for alert conditions."""
        if timestamp is None:
            timestamp = time.time()
            
        with self._lock:
            # Get current metric values
            current_metrics = {}
            for metric_name, window in self.metric_windows.items():
                if window.values:
                    current_metrics[metric_name] = window.values[-1]
                    
            return self.alert_manager.check_alerts(current_metrics, timestamp)
            
    def _handle_learning_rate_alert(self, alert_data: Dict[str, Any]):
        """Handle learning rate alert."""
        self.logger.warning(f"Low learning rate detected: {alert_data['value']:.6f}")
        
    def _handle_quality_alert(self, alert_data: Dict[str, Any]):
        """Handle quality alert."""
        self.logger.error(f"Low quality score detected: {alert_data['value']:.3f}")
        
    def _handle_resource_alert(self, alert_data: Dict[str, Any]):
        """Handle resource usage alert."""
        self.logger.warning(f"High resource usage detected: {alert_data['value']:.2%}")
        
    def _handle_stability_alert(self, alert_data: Dict[str, Any]):
        """Handle stability alert."""
        self.logger.warning(f"Low weight stability detected: {alert_data['value']:.3f}")
        
    def clear_metrics(self, metric_name: Optional[str] = None):
        """Clear metrics data."""
        with self._lock:
            if metric_name:
                self.metric_windows.pop(metric_name, None)
            else:
                self.metric_windows.clear()
                for metric_type in self.metric_points:
                    self.metric_points[metric_type].clear()
                    
        self.logger.info(f"Metrics cleared for {'all metrics' if metric_name is None else metric_name}")
        
    def get_metric_names(self) -> List[str]:
        """Get list of available metric names."""
        with self._lock:
            return list(self.metric_windows.keys())
            
    def add_custom_alert(self, alert_config: AlertConfig):
        """Add custom alert configuration."""
        self.alert_manager.add_alert(alert_config)
        
    def remove_alert(self, metric_name: str):
        """Remove alert configuration."""
        self.alert_manager.remove_alert(metric_name)