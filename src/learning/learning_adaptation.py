"""
Learning Rate Adaptation and Quality Monitoring.

This module provides adaptive learning mechanisms and quality assessment
for online learning systems in spiking neural networks.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
import threading


class AdaptationStrategy(Enum):
    """Learning rate adaptation strategies."""
    FIXED = "fixed"                # Fixed learning rate
    DECREASING = "decreasing"      # Linearly decreasing
    EXPONENTIAL = "exponential"    # Exponentially decreasing
    ADAPTIVE = "adaptive"          # Performance-based adaptation
    OSCILLATING = "oscillating"    # Cyclic oscillation
    MOMENTUM = "momentum"          # Momentum-based adaptation


class QualityMetric(Enum):
    """Types of quality metrics."""
    LEARNING_RATE = "learning_rate"
    PREDICTION_ACCURACY = "prediction_accuracy"
    WEIGHT_STABILITY = "weight_stability"
    REPLAY_PERFORMANCE = "replay_performance"
    ERROR_REDUCTION = "error_reduction"
    CONVERGENCE_RATE = "convergence_rate"


@dataclass
class AdaptationConfig:
    """Configuration for learning rate adaptation."""
    strategy: AdaptationStrategy = AdaptationStrategy.ADAPTIVE
    initial_rate: float = 0.1
    min_rate: float = 0.001
    max_rate: float = 1.0
    
    # Adaptation parameters
    adaptation_threshold: float = 0.05
    adaptation_patience: int = 10
    adaptation_factor: float = 0.9
    
    # Momentum parameters
    momentum: float = 0.9
    momentum_decay: float = 0.95
    
    # Oscillation parameters
    oscillation_amplitude: float = 0.1
    oscillation_period: int = 1000
    
    # Quality thresholds
    target_quality: float = 0.8
    quality_window: int = 50


@dataclass 
class QualityThresholds:
    """Quality assessment thresholds."""
    excellent: float = 0.9
    good: float = 0.7
    acceptable: float = 0.5
    poor: float = 0.3
    unacceptable: float = 0.1


class PerformanceTracker:
    """Track and analyze learning performance metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: Dict[QualityMetric, deque] = {
            metric: deque(maxlen=window_size) for metric in QualityMetric
        }
        self._lock = threading.RLock()
        
    def add_metric(self, metric: QualityMetric, value: float):
        """Add a metric value."""
        with self._lock:
            self.metrics_history[metric].append(value)
            
    def get_recent_values(self, metric: QualityMetric, n: int = 10) -> List[float]:
        """Get recent values for a metric."""
        with self._lock:
            return list(self.metrics_history[metric])[-n:]
            
    def get_average(self, metric: QualityMetric, window: Optional[int] = None) -> float:
        """Get average value for a metric."""
        with self._lock:
            values = self.get_recent_values(metric, window or self.window_size)
            return np.mean(values) if values else 0.0
            
    def get_trend(self, metric: QualityMetric, window: int = 20) -> float:
        """Calculate trend (slope) for a metric."""
        with self._lock:
            values = self.get_recent_values(metric, window)
            if len(values) < 2:
                return 0.0
                
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            return slope
            
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive statistics for all metrics."""
        with self._lock:
            stats = {}
            
            for metric in QualityMetric:
                values = list(self.metrics_history[metric])
                if values:
                    stats[metric.value] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "trend": self.get_trend(metric),
                        "count": len(values)
                    }
                else:
                    stats[metric.value] = {
                        "mean": 0.0, "std": 0.0, "min": 0.0, 
                        "max": 0.0, "trend": 0.0, "count": 0
                    }
                    
            return stats


class LearningRateAdapter:
    """
    Adaptive learning rate controller.
    
    Provides multiple strategies for learning rate adaptation based on
    performance monitoring and quality assessment.
    """
    
    def __init__(self, 
                 config: Optional[AdaptationConfig] = None,
                 performance_tracker: Optional[PerformanceTracker] = None):
        """
        Initialize learning rate adapter.
        
        Args:
            config: Adaptation configuration
            performance_tracker: Performance tracking instance
        """
        self.config = config or AdaptationConfig()
        self.performance_tracker = performance_tracker or PerformanceTracker()
        
        self.current_rate = self.config.initial_rate
        self.momentum_velocity = 0.0
        self.adaptation_count = 0
        self.stagnation_count = 0
        self.last_adaptation_time = time.time()
        
        # State tracking
        self.rate_history = deque(maxlen=self.config.quality_window)
        self.quality_history = deque(maxlen=self.config.quality_window)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def adapt_learning_rate(self,
                          current_rate: float,
                          quality_score: float,
                          performance_history: Optional[List[float]] = None,
                          weight_statistics: Optional[Dict[str, float]] = None) -> float:
        """
        Adapt learning rate based on current performance.
        
        Args:
            current_rate: Current learning rate
            quality_score: Current quality score (0-1)
            performance_history: Recent performance values
            weight_statistics: Current weight statistics
            
        Returns:
            Adapted learning rate
        """
        # Update internal state
        self.current_rate = current_rate
        self.quality_history.append(quality_score)
        
        if performance_history:
            avg_performance = np.mean(performance_history[-10:])
            self.performance_tracker.add_metric(QualityMetric.REPLAY_PERFORMANCE, avg_performance)
            
        if weight_statistics:
            stability = 1.0 / (1.0 + weight_statistics.get("std_weight", 1.0))
            self.performance_tracker.add_metric(QualityMetric.WEIGHT_STABILITY, stability)
            
        self.performance_tracker.add_metric(QualityMetric.LEARNING_RATE, current_rate)
        self.performance_tracker.add_metric(QualityMetric.PREDICTION_ACCURACY, quality_score)
        
        # Apply adaptation strategy
        if self.config.strategy == AdaptationStrategy.FIXED:
            new_rate = current_rate
            
        elif self.config.strategy == AdaptationStrategy.DECREASING:
            new_rate = self._decreasing_adaptation(quality_score)
            
        elif self.config.strategy == AdaptationStrategy.EXPONENTIAL:
            new_rate = self._exponential_adaptation()
            
        elif self.config.strategy == AdaptationStrategy.ADAPTIVE:
            new_rate = self._adaptive_adaptation(quality_score, performance_history or [])
            
        elif self.config.strategy == AdaptationStrategy.OSCILLATING:
            new_rate = self._oscillating_adaptation()
            
        elif self.config.strategy == AdaptationStrategy.MOMENTUM:
            new_rate = self._momentum_adaptation(quality_score)
            
        else:
            new_rate = current_rate
            
        # Clamp to valid range
        new_rate = np.clip(new_rate, self.config.min_rate, self.config.max_rate)
        
        # Update tracking
        self.rate_history.append(new_rate)
        self.adaptation_count += 1 if new_rate != current_rate else 0
        self.last_adaptation_time = time.time()
        
        if abs(new_rate - current_rate) < self.config.adaptation_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
            
        # Log significant changes
        if abs(new_rate - current_rate) > 0.01:
            self.logger.info(f"Learning rate adapted: {current_rate:.6f} -> {new_rate:.6f}, "
                           f"quality: {quality_score:.3f}")
                           
        return new_rate
        
    def _decreasing_adaptation(self, quality_score: float) -> float:
        """Linear decreasing adaptation."""
        decay_factor = 0.9995  # Very slow decay
        return self.current_rate * decay_factor
        
    def _exponential_adaptation(self) -> float:
        """Exponential decay adaptation."""
        decay_rate = 0.001
        return self.current_rate * np.exp(-decay_rate)
        
    def _adaptive_adaptation(self, quality_score: float, 
                           performance_history: List[float]) -> float:
        """Performance-based adaptive adaptation."""
        if len(self.quality_history) < 5:
            return self.current_rate
            
        # Analyze quality trend
        recent_quality = list(self.quality_history)[-5:]
        quality_trend = recent_quality[-1] - recent_quality[0]
        
        # Analyze performance trend
        if performance_history:
            recent_perf = performance_history[-5:]
            perf_trend = np.mean(recent_perf[-2:]) - np.mean(recent_perf[:2])
        else:
            perf_trend = 0.0
            
        # Adaptation logic
        if quality_trend < -0.1 or perf_trend < -0.1:
            # Performance declining - increase learning rate
            adaptation_factor = 1.1 + abs(quality_trend)
            new_rate = self.current_rate * min(adaptation_factor, 2.0)
            
        elif quality_trend > 0.1 or perf_trend > 0.1:
            # Performance improving - cautiously decrease learning rate
            adaptation_factor = 0.95 - abs(quality_trend) * 0.1
            new_rate = self.current_rate * max(adaptation_factor, 0.8)
            
        elif self.stagnation_count > self.config.adaptation_patience:
            # Learning stagnated - apply momentum approach
            new_rate = self._momentum_adaptation(quality_score)
            
        else:
            # Stable performance - gentle decay
            new_rate = self.current_rate * 0.999
            
        return new_rate
        
    def _oscillating_adaptation(self) -> float:
        """Cyclical oscillation adaptation."""
        cycle_position = self.adaptation_count % self.config.oscillation_period
        oscillation = np.sin(2 * np.pi * cycle_position / self.config.oscillation_period)
        
        # Oscillate around current rate
        amplitude = self.config.oscillation_amplitude
        new_rate = self.current_rate * (1.0 + amplitude * oscillation)
        
        return new_rate
        
    def _momentum_adaptation(self, quality_score: float) -> float:
        """Momentum-based adaptation with velocity."""
        # Calculate gradient approximation
        if len(self.quality_history) >= 2:
            quality_diff = quality_score - self.quality_history[-2]
            gradient = -quality_diff  # Negative because we want to increase rate when quality decreases
        else:
            gradient = 0.0
            
        # Update velocity with momentum
        self.momentum_velocity = (self.config.momentum * self.momentum_velocity + 
                                gradient * (1 - self.config.momentum))
        
        # Apply velocity to learning rate
        adaptation_factor = 1.0 + self.momentum_velocity * 0.1
        new_rate = self.current_rate * adaptation_factor
        
        # Decay momentum
        self.momentum_velocity *= self.config.momentum_decay
        
        return new_rate
        
    def should_reset_momentum(self) -> bool:
        """Check if momentum should be reset."""
        return self.stagnation_count > self.config.adaptation_patience * 2
        
    def reset(self):
        """Reset adaptation state."""
        self.current_rate = self.config.initial_rate
        self.momentum_velocity = 0.0
        self.adaptation_count = 0
        self.stagnation_count = 0
        self.rate_history.clear()
        self.quality_history.clear()
        
        self.logger.info("Learning rate adapter reset")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            "current_rate": self.current_rate,
            "adaptation_count": self.adaptation_count,
            "stagnation_count": self.stagnation_count,
            "strategy": self.config.strategy.value,
            "rate_history_size": len(self.rate_history),
            "quality_history_size": len(self.quality_history),
            "performance_stats": self.performance_tracker.get_statistics(),
            "recent_rate_change": (list(self.rate_history)[-1] - list(self.rate_history)[-2]) 
                                 if len(self.rate_history) >= 2 else 0.0
        }


class QualityMonitor:
    """
    Monitor and assess learning quality.
    
    Provides comprehensive quality metrics and thresholds for
    assessing learning performance and triggering adaptations.
    """
    
    def __init__(self,
                 window_size: int = 50,
                 target_score: float = 0.8,
                 thresholds: Optional[QualityThresholds] = None):
        """
        Initialize quality monitor.
        
        Args:
            window_size: Window size for quality calculations
            target_score: Target quality score
            thresholds: Quality assessment thresholds
        """
        self.window_size = window_size
        self.target_score = target_score
        self.thresholds = thresholds or QualityThresholds()
        
        self.quality_history = deque(maxlen=window_size)
        self.performance_history = deque(maxlen=window_size)
        self.error_history = deque(maxlen=window_size)
        
        self.last_assessment_time = time.time()
        self.consecutive_good_assessments = 0
        self.consecutive_poor_assessments = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def update_quality(self, 
                      updates: int,
                      total_attempts: int,
                      error_count: int = 0,
                      weight_changes: float = 0.0) -> float:
        """
        Update quality assessment based on recent performance.
        
        Args:
            updates: Number of successful updates
            total_attempts: Total number of attempts
            error_count: Number of errors
            weight_changes: Magnitude of weight changes
            
        Returns:
            Quality score (0-1)
        """
        if total_attempts == 0:
            quality_score = 0.0
        else:
            # Calculate success rate
            success_rate = updates / total_attempts
            
            # Calculate error rate
            error_rate = error_count / max(1, total_attempts)
            
            # Calculate weight change factor (prefer moderate changes)
            weight_factor = 1.0 / (1.0 + abs(weight_changes))
            
            # Combine factors
            quality_score = (success_rate * 0.5 + 
                           (1.0 - error_rate) * 0.3 + 
                           weight_factor * 0.2)
                           
        # Update history
        self.quality_history.append(quality_score)
        self.performance_history.append(success_rate)
        self.error_history.append(error_rate)
        
        # Update assessment counters
        if quality_score >= self.thresholds.good:
            self.consecutive_good_assessments += 1
            self.consecutive_poor_assessments = 0
        elif quality_score < self.thresholds.poor:
            self.consecutive_poor_assessments += 1
            self.consecutive_good_assessments = 0
        else:
            self.consecutive_good_assessments = 0
            self.consecutive_poor_assessments = 0
            
        self.last_assessment_time = time.time()
        
        return np.clip(quality_score, 0.0, 1.0)
        
    def assess_quality(self) -> Dict[str, Any]:
        """Perform comprehensive quality assessment."""
        if not self.quality_history:
            return {"overall_quality": 0.0, "assessment": "no_data"}
            
        recent_quality = list(self.quality_history)[-10:]
        avg_quality = np.mean(recent_quality)
        
        # Calculate trend
        if len(recent_quality) >= 3:
            trend = recent_quality[-1] - recent_quality[0]
        else:
            trend = 0.0
            
        # Quality level assessment
        if avg_quality >= self.thresholds.excellent:
            level = "excellent"
            status = "learning_very_well"
        elif avg_quality >= self.thresholds.good:
            level = "good"
            status = "learning_well"
        elif avg_quality >= self.thresholds.acceptable:
            level = "acceptable"
            status = "learning_stably"
        elif avg_quality >= self.thresholds.poor:
            level = "poor"
            status = "learning_poorly"
        else:
            level = "unacceptable"
            status = "learning_failed"
            
        return {
            "overall_quality": avg_quality,
            "quality_level": level,
            "status": status,
            "trend": trend,
            "target_score": self.target_score,
            "good_consecutive": getattr(self, 'consecutive_good_assessments', 0),
            "poor_consecutive": getattr(self, 'consecutive_poor_assessments', 0),
            "history_size": len(self.quality_history)
        }
        
    def should_adapt_rate(self) -> bool:
        """Determine if learning rate should be adapted."""
        assessment = self.assess_quality()
        
        # Adapt if quality is consistently poor or excellent
        if (assessment["poor_consecutive"] >= 5 or 
            assessment["good_consecutive"] >= 10):
            return True
            
        # Adapt if quality deviates significantly from target
        quality_deviation = abs(assessment["overall_quality"] - self.target_score)
        if quality_deviation > 0.2:
            return True
            
        # Adapt if trend is strongly negative
        if assessment["trend"] < -0.05:
            return True
            
        return False
        
    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get detailed quality metrics."""
        return {
            "current_quality": list(self.quality_history)[-1] if self.quality_history else 0.0,
            "average_quality": np.mean(list(self.quality_history)) if self.quality_history else 0.0,
            "quality_stability": 1.0 - np.std(list(self.quality_history)) if self.quality_history else 0.0,
            "recent_performance": np.mean(list(self.performance_history)) if self.performance_history else 0.0,
            "recent_error_rate": np.mean(list(self.error_history)) if self.error_history else 0.0,
            "target_achievement": (np.mean(list(self.quality_history)) >= self.target_score 
                                 if self.quality_history else False),
            "assessment": self.assess_quality()
        }
        
    def reset(self):
        """Reset quality monitoring state."""
        self.quality_history.clear()
        self.performance_history.clear()
        self.error_history.clear()
        self.consecutive_good_assessments = 0
        self.consecutive_poor_assessments = 0
        self.last_assessment_time = time.time()
        
        self.logger.info("Quality monitor reset")