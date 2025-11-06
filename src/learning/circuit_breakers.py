"""
Learning Circuit Breakers for Performance Protection.

This module provides circuit breaker mechanisms to protect network performance
during learning processes by detecting failures and implementing safety measures.
"""

import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import statistics
import math


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped
    HALF_OPEN = "half_open"  # Testing if condition improved


class FailureType(Enum):
    """Types of failures detected by circuit breakers."""
    LEARNING_FAILURE = "learning_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TIMEOUT = "timeout"
    MEMORY_OVERFLOW = "memory_overflow"
    QUALITY_DROP = "quality_drop"
    STABILITY_LOSS = "stability_loss"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    # Threshold settings
    failure_threshold: float = 0.1
    success_threshold: float = 0.8
    quality_threshold: float = 0.3
    
    # Time settings
    timeout: float = 300.0  # seconds
    half_open_test_duration: float = 30.0  # seconds
    monitoring_window: int = 100
    
    # Recovery settings
    gradual_recovery: bool = True
    recovery_rate: float = 0.1
    
    # Alert settings
    alert_on_trip: bool = True
    alert_callback: Optional[Callable] = None


@dataclass
class FailureEvent:
    """Record of a failure event."""
    failure_type: FailureType
    timestamp: float
    severity: float  # 0-1
    details: Dict[str, Any]
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp


class FailureDetector:
    """Detects various types of learning failures."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_history: deque = deque(maxlen=config.monitoring_window)
        self.performance_history: deque = deque(maxlen=config.monitoring_window)
        self.quality_history: deque = deque(maxlen=config.monitoring_window)
        
        self._lock = threading.RLock()
        
    def detect_failures(self,
                       performance_metrics: Dict[str, Any],
                       quality_metrics: Dict[str, Any],
                       resource_metrics: Optional[Dict[str, Any]] = None) -> List[FailureEvent]:
        """
        Detect failures based on current metrics.
        
        Args:
            performance_metrics: Performance-related metrics
            quality_metrics: Quality-related metrics
            resource_metrics: Resource usage metrics
            
        Returns:
            List of detected failures
        """
        with self._lock:
            failures = []
            current_time = time.time()
            
            # Check for learning failures
            if self._detect_learning_failure(performance_metrics, current_time):
                failures.append(FailureEvent(
                    failure_type=FailureType.LEARNING_FAILURE,
                    timestamp=current_time,
                    severity=0.8,
                    details={"metrics": performance_metrics}
                ))
                
            # Check for performance degradation
            if self._detect_performance_degradation(performance_metrics, current_time):
                failures.append(FailureEvent(
                    failure_type=FailureType.PERFORMANCE_DEGRADATION,
                    timestamp=current_time,
                    severity=0.6,
                    details={"current": performance_metrics, "history": list(self.performance_history)}
                ))
                
            # Check for quality drops
            if self._detect_quality_drop(quality_metrics, current_time):
                failures.append(FailureEvent(
                    failure_type=FailureType.QUALITY_DROP,
                    timestamp=current_time,
                    severity=0.7,
                    details={"quality": quality_metrics, "threshold": self.config.quality_threshold}
                ))
                
            # Check for stability loss
            if self._detect_stability_loss(performance_metrics, current_time):
                failures.append(FailureEvent(
                    failure_type=FailureType.STABILITY_LOSS,
                    timestamp=current_time,
                    severity=0.5,
                    details={"stability_metrics": performance_metrics}
                ))
                
            # Check for resource exhaustion
            if resource_metrics and self._detect_resource_exhaustion(resource_metrics, current_time):
                failures.append(FailureEvent(
                    failure_type=FailureType.RESOURCE_EXHAUSTION,
                    timestamp=current_time,
                    severity=0.9,
                    details={"resources": resource_metrics}
                ))
                
            # Check for timeouts
            if self._detect_timeout(current_time):
                failures.append(FailureEvent(
                    failure_type=FailureType.TIMEOUT,
                    timestamp=current_time,
                    severity=0.4,
                    details={"timeout_threshold": self.config.timeout}
                ))
                
            # Update histories
            self.failure_history.extend(failures)
            self.performance_history.append(performance_metrics.copy())
            self.quality_history.append(quality_metrics.copy())
            
            return failures
            
    def _detect_learning_failure(self, metrics: Dict[str, Any], current_time: float) -> bool:
        """Detect learning failures."""
        # Check update success rate
        updates = metrics.get("updates", 0)
        total_attempts = metrics.get("total_attempts", 1)
        
        if total_attempts == 0:
            return True
            
        success_rate = updates / total_attempts
        
        # Failure if success rate is too low
        if success_rate < (1.0 - self.config.failure_threshold):
            return True
            
        # Check for consecutive failures in recent history
        recent_failures = [f for f in self.failure_history 
                          if current_time - f.timestamp < 60.0 and 
                          f.failure_type == FailureType.LEARNING_FAILURE]
                          
        if len(recent_failures) >= 5:
            return True
            
        return False
        
    def _detect_performance_degradation(self, metrics: Dict[str, Any], current_time: float) -> bool:
        """Detect performance degradation trends."""
        if len(self.performance_history) < 10:
            return False
            
        # Get recent performance metrics
        recent_performance = [m.get("updates", 0) for m in list(self.performance_history)[-10:]]
        
        if not recent_performance:
            return False
            
        # Check if performance is consistently declining
        baseline = np.mean(recent_performance[:5])
        recent_avg = np.mean(recent_performance[-5:])
        
        performance_drop = (baseline - recent_avg) / max(1e-10, baseline)
        
        return performance_drop > self.config.failure_threshold
        
    def _detect_quality_drop(self, metrics: Dict[str, Any], current_time: float) -> bool:
        """Detect significant quality drops."""
        current_quality = metrics.get("overall_quality", 1.0)
        
        # Immediate drop below threshold
        if current_quality < self.config.quality_threshold:
            return True
            
        # Check trend in quality history
        if len(self.quality_history) >= 10:
            recent_quality = [q.get("overall_quality", 0.0) for q in list(self.quality_history)[-10:]]
            
            if recent_quality:
                quality_trend = recent_quality[-1] - recent_quality[0]
                
                # Significant downward trend
                if quality_trend < -0.2:
                    return True
                    
        return False
        
    def _detect_stability_loss(self, metrics: Dict[str, Any], current_time: float) -> bool:
        """Detect loss of weight stability."""
        weight_std = metrics.get("weight_std", 0.0)
        weight_mean = metrics.get("weight_mean", 0.5)
        
        if weight_mean == 0:
            return False
            
        # Coefficient of variation as stability measure
        cv = weight_std / abs(weight_mean)
        
        # High coefficient of variation indicates instability
        if cv > 1.0:  # Very high variability
            return True
            
        # Check if variability is increasing
        if len(self.performance_history) >= 5:
            recent_cvs = []
            for m in list(self.performance_history)[-5:]:
                w_std = m.get("weight_std", 0.0)
                w_mean = m.get("weight_mean", 0.5)
                if w_mean != 0:
                    recent_cvs.append(w_std / abs(w_mean))
                    
            if len(recent_cvs) >= 2 and recent_cvs[-1] > recent_cvs[0] * 1.5:
                return True
                
        return False
        
    def _detect_resource_exhaustion(self, metrics: Dict[str, Any], current_time: float) -> bool:
        """Detect resource exhaustion."""
        # Check memory usage
        memory_usage = metrics.get("memory_usage", 0.0)
        if memory_usage > 0.95:  # 95% memory usage
            return True
            
        # Check CPU usage
        cpu_usage = metrics.get("cpu_usage", 0.0)
        if cpu_usage > 0.90:  # 90% CPU usage
            return True
            
        # Check GPU usage if available
        gpu_usage = metrics.get("gpu_usage", 0.0)
        if gpu_usage and gpu_usage > 0.90:
            return True
            
        return False
        
    def _detect_timeout(self, current_time: float) -> bool:
        """Detect timeout conditions."""
        if not self.failure_history:
            return False
            
        last_failure = max(self.failure_history, key=lambda f: f.timestamp)
        time_since_last_failure = current_time - last_failure.timestamp
        
        return time_since_last_failure > self.config.timeout
        
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure statistics."""
        with self._lock:
            now = time.time()
            
            # Count failures by type
            failure_counts = defaultdict(int)
            recent_failures = []
            
            for failure in self.failure_history:
                failure_counts[failure.failure_type] += 1
                if now - failure.timestamp < 300.0:  # Last 5 minutes
                    recent_failures.append(failure)
                    
            return {
                "total_failures": len(self.failure_history),
                "failure_counts": dict(failure_counts),
                "recent_failures": len(recent_failures),
                "failure_rate": len(self.failure_history) / max(1, len(self.performance_history)),
                "current_performance": list(self.performance_history)[-1] if self.performance_history else {},
                "current_quality": list(self.quality_history)[-1] if self.quality_history else {}
            }


class LearningCircuitBreaker:
    """
    Circuit breaker for learning processes.
    
    Implements a three-state circuit breaker (Closed/Open/Half-Open) to
    protect network performance during learning by detecting failures
    and implementing protective measures.
    """
    
    def __init__(self, 
                 config: Optional[CircuitBreakerConfig] = None,
                 failure_detector: Optional[FailureDetector] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            failure_detector: Failure detection instance
        """
        self.config = config or CircuitBreakerConfig()
        self.failure_detector = failure_detector or FailureDetector(self.config)
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.half_open_start_time = 0.0
        self.success_count = 0
        self.failure_count = 0
        
        # Recovery tracking
        self.recovery_attempts = 0
        self.last_recovery_attempt = 0.0
        
        # Statistics
        self.total_trips = 0
        self.total_recoveries = 0
        self.circuit_events: deque = deque(maxlen=1000)
        
        # Callbacks
        self.on_trip_callback: Optional[Callable] = None
        self.on_recovery_callback: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def check_operation(self, 
                       performance_metrics: Dict[str, Any],
                       quality_metrics: Dict[str, Any],
                       resource_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if operation should proceed based on circuit breaker state.
        
        Args:
            performance_metrics: Current performance metrics
            quality_metrics: Current quality metrics
            resource_metrics: Current resource usage metrics
            
        Returns:
            True if operation should proceed, False if blocked
        """
        with self._lock:
            current_time = time.time()
            
            # Check current state
            if self.state == CircuitState.CLOSED:
                return self._check_closed_state(current_time, performance_metrics, 
                                              quality_metrics, resource_metrics)
            elif self.state == CircuitState.OPEN:
                return self._check_open_state(current_time)
            elif self.state == CircuitState.HALF_OPEN:
                return self._check_half_open_state(current_time, performance_metrics,
                                                 quality_metrics)
                
        return False
        
    def _check_closed_state(self, 
                          current_time: float,
                          performance_metrics: Dict[str, Any],
                          quality_metrics: Dict[str, Any],
                          resource_metrics: Optional[Dict[str, Any]]) -> bool:
        """Check operation in closed state."""
        # Detect failures
        failures = self.failure_detector.detect_failures(
            performance_metrics, quality_metrics, resource_metrics
        )
        
        if failures:
            # Record failures and potentially trip
            self._record_failures(failures)
            
            # Check if circuit should trip
            if self._should_trip_circuit(current_time):
                self._trip_circuit(current_time)
                
                if self.config.alert_on_trip and self.on_trip_callback:
                    self.on_trip_callback(failures)
                    
                return False
                
        # Update success tracking
        self.success_count += 1
        self._record_event("success", current_time, performance_metrics, quality_metrics)
        
        return True
        
    def _check_open_state(self, current_time: float) -> bool:
        """Check operation in open state."""
        # Check if timeout has elapsed
        time_since_failure = current_time - self.last_failure_time
        
        if time_since_failure >= self.config.timeout:
            # Move to half-open state
            self._move_to_half_open(current_time)
            self.logger.info("Circuit breaker moved to half-open state")
            
            # Allow operation in half-open state
            return True
            
        return False
        
    def _check_half_open_state(self, 
                             current_time: float,
                             performance_metrics: Dict[str, Any],
                             quality_metrics: Dict[str, Any]) -> bool:
        """Check operation in half-open state."""
        elapsed_time = current_time - self.half_open_start_time
        
        if elapsed_time > self.config.half_open_test_duration:
            # Test period over, evaluate results
            success_rate = self.success_count / max(1, self.success_count + self.failure_count)
            
            if success_rate >= self.config.success_threshold:
                # Recovery successful
                self._recover_circuit(current_time)
                self.logger.info("Circuit breaker recovered successfully")
                return True
            else:
                # Test failed, return to open state
                self._return_to_open(current_time)
                self.logger.info("Circuit breaker half-open test failed, returning to open")
                return False
                
        # Continue testing
        return True
        
    def _should_trip_circuit(self, current_time: float) -> bool:
        """Determine if circuit should trip based on failure rate."""
        # Check recent failure rate
        recent_failures = [e for e in self.circuit_events 
                          if current_time - e["timestamp"] < 60.0 and e["type"] == "failure"]
                          
        if len(recent_failures) >= 5:
            return True
            
        # Check failure rate percentage
        recent_events = [e for e in self.circuit_events 
                        if current_time - e["timestamp"] < 60.0]
                        
        if len(recent_events) >= 10:
            failure_rate = sum(1 for e in recent_events if e["type"] == "failure") / len(recent_events)
            if failure_rate > self.config.failure_threshold:
                return True
                
        return False
        
    def _record_failures(self, failures: List[FailureEvent]):
        """Record failure events."""
        current_time = time.time()
        
        for failure in failures:
            self.failure_count += 1
            self.last_failure_time = current_time
            
            self.circuit_events.append({
                "type": "failure",
                "timestamp": current_time,
                "failure_type": failure.failure_type.value,
                "severity": failure.severity,
                "details": failure.details
            })
            
        self.logger.warning(f"Recorded {len(failures)} learning failures")
        
    def _record_event(self, event_type: str, timestamp: float,
                     performance_metrics: Dict[str, Any],
                     quality_metrics: Dict[str, Any]):
        """Record circuit breaker event."""
        self.circuit_events.append({
            "type": event_type,
            "timestamp": timestamp,
            "performance": performance_metrics,
            "quality": quality_metrics
        })
        
    def _trip_circuit(self, current_time: float):
        """Trip the circuit breaker to open state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = current_time
        self.total_trips += 1
        
        self.circuit_events.append({
            "type": "trip",
            "timestamp": current_time,
            "state": self.state.value
        })
        
        self.logger.error(f"Circuit breaker tripped to OPEN state")
        
    def _move_to_half_open(self, current_time: float):
        """Move circuit breaker to half-open state for testing."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_start_time = current_time
        self.success_count = 0
        self.failure_count = 0
        self.recovery_attempts += 1
        
        self.circuit_events.append({
            "type": "half_open",
            "timestamp": current_time,
            "state": self.state.value
        })
        
    def _recover_circuit(self, current_time: float):
        """Recover circuit breaker to closed state."""
        self.state = CircuitState.CLOSED
        self.success_count = 0
        self.failure_count = 0
        self.total_recoveries += 1
        
        if self.on_recovery_callback:
            self.on_recovery_callback()
            
        self.circuit_events.append({
            "type": "recovery",
            "timestamp": current_time,
            "state": self.state.value
        })
        
        self.logger.info(f"Circuit breaker recovered to CLOSED state")
        
    def _return_to_open(self, current_time: float):
        """Return to open state after failed half-open test."""
        self.state = CircuitState.OPEN
        self.last_failure_time = current_time
        self.success_count = 0
        self.failure_count = 0
        
        self.circuit_events.append({
            "type": "return_to_open",
            "timestamp": current_time,
            "state": self.state.value
        })
        
    def record_success(self):
        """Record successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                
    def record_failure(self):
        """Record failed operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.failure_count += 1
                
    def is_tripped(self) -> bool:
        """Check if circuit breaker is tripped (open state)."""
        return self.state == CircuitState.OPEN
        
    def force_open(self):
        """Force circuit breaker to open state."""
        with self._lock:
            self._trip_circuit(time.time())
            
    def force_close(self):
        """Force circuit breaker to closed state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.success_count = 0
            self.failure_count = 0
            
            self.circuit_events.append({
                "type": "force_close",
                "timestamp": time.time()
            })
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker statistics."""
        with self._lock:
            current_time = time.time()
            
            recent_events = [e for e in self.circuit_events 
                           if current_time - e["timestamp"] < 300.0]
                           
            return {
                "state": self.state.value,
                "total_trips": self.total_trips,
                "total_recoveries": self.total_recoveries,
                "recovery_attempts": self.recovery_attempts,
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "time_since_last_failure": current_time - self.last_failure_time,
                "half_open_elapsed": (current_time - self.half_open_start_time 
                                    if self.state == CircuitState.HALF_OPEN else 0),
                "recent_events": len(recent_events),
                "failure_detector_stats": self.failure_detector.get_failure_statistics()
            }
            
    def set_callbacks(self, 
                     on_trip: Optional[Callable] = None,
                     on_recovery: Optional[Callable] = None):
        """Set event callbacks."""
        self.on_trip_callback = on_trip
        self.on_recovery_callback = on_recovery
        
    def reset(self):
        """Reset circuit breaker state."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.success_count = 0
            self.failure_count = 0
            self.last_failure_time = 0.0
            self.half_open_start_time = 0.0
            self.recovery_attempts = 0
            self.last_recovery_attempt = 0.0
            self.circuit_events.clear()
            
            self.logger.info("Circuit breaker reset")