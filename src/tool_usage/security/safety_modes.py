#!/usr/bin/env python3
"""
Safety Modes and System Protection Module

This module provides system safety controls including circuit breakers,
rate limiting, emergency procedures, and system health monitoring.

Author: Mini-Biai Framework Team
Version: 1.0.0
Date: 2024
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..core.exceptions import SafetyError, CircuitBreakerError


class SafetyMode(Enum):
    """System safety operational modes"""
    NORMAL = "normal"           # Normal operations
    SAFE = "safe"               # Restricted operations
    RESTRICTED = "restricted"   # Minimal operations
    EMERGENCY = "emergency"     # Emergency procedures only
    LOCKDOWN = "lockdown"       # Complete system lockdown


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"           # Normal operation
    OPEN = "open"               # Blocking requests
    HALF_OPEN = "half_open"     # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5          # Number of failures to open circuit
    recovery_timeout: int = 60          # Seconds to wait before trying recovery
    success_threshold: int = 3          # Successes needed to close circuit
    timeout: float = 30.0              # Request timeout in seconds
    expected_exception: type = Exception  # Exception type to monitor
    

@dataclass
class SystemHealthMetrics:
    """System health monitoring metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_active: bool = False
    active_processes: int = 0
    error_rate: float = 0.0
    response_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "network_active": self.network_active,
            "active_processes": self.active_processes,
            "error_rate": self.error_rate,
            "response_time": self.response_time
        }


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rejected_requests": 0,
            "state_changes": 0
        }
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.stats["total_requests"] += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    self._log_state_change("Circuit moved to HALF_OPEN for testing")
                else:
                    self.stats["rejected_requests"] += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                    )
            
            try:
                # Execute function with timeout
                result = self._execute_with_timeout(func, args, kwargs)
                
                # Update success counters
                self._on_success()
                return result
                
            except self.config.expected_exception as e:
                self._on_failure()
                raise
    
    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict) -> Any:
        """Execute function with timeout protection"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {self.config.timeout} seconds")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.config.timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        except:
            signal.alarm(0)  # Cancel alarm
            raise
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _on_success(self) -> None:
        """Handle successful function execution"""
        self.stats["successful_requests"] += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self._reset_counters()
                self._log_state_change("Circuit breaker recovered and CLOSED")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0  # Reset on success
    
    def _on_failure(self) -> None:
        """Handle failed function execution"""
        self.stats["failed_requests"] += 1
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self._log_state_change("Circuit breaker opened due to failures")
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            self._log_state_change("Circuit breaker reopened during testing")
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return False
        
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.config.recovery_timeout
    
    def _reset_counters(self) -> None:
        """Reset failure and success counters"""
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    def _log_state_change(self, message: str) -> None:
        """Log circuit breaker state change"""
        self.stats["state_changes"] += 1
        # In a real implementation, this would use the framework's logging
        print(f"[CIRCUIT_BREAKER] {message}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state and statistics"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "stats": self.stats.copy()
        }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self._reset_counters()
            self._log_state_change("Circuit breaker manually reset to CLOSED")


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        """
        Initialize rate limiter
        
        Args:
            max_tokens: Maximum number of tokens in bucket
            refill_rate: Tokens to add per second
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "requests_allowed": 0,
            "requests_denied": 0,
            "total_tokens_consumed": 0,
            "peak_tokens_used": max_tokens
        }
    
    def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from bucket
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens were acquired, False otherwise
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.stats["requests_allowed"] += 1
                self.stats["total_tokens_consumed"] += tokens
                
                # Update peak usage
                tokens_used = self.max_tokens - self.tokens
                if tokens_used > self.stats["peak_tokens_used"]:
                    self.stats["peak_tokens_used"] = tokens_used
                
                return True
            else:
                self.stats["requests_denied"] += 1
                return False
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            tokens_to_add = elapsed * self.refill_rate
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "current_tokens": self.tokens,
            "max_tokens": self.max_tokens,
            "refill_rate": self.refill_rate,
            "stats": self.stats.copy()
        }
    
    def reset(self) -> None:
        """Reset rate limiter to full capacity"""
        with self.lock:
            self.tokens = self.max_tokens
            self.last_refill = time.time()


class SystemHealthMonitor:
    """System health and performance monitoring"""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=1000)
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        # Health thresholds
        self.thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "error_rate": 0.05,  # 5%
            "response_time": 2.0  # seconds
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[SystemHealthMetrics], None]] = []
    
    def start_monitoring(self, interval_seconds: int = 30) -> None:
        """Start system health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._shutdown_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop system health monitoring"""
        self.monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for health issues
                self._check_health_thresholds(metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                print(f"Error in health monitoring: {e}")
    
    def _collect_metrics(self) -> SystemHealthMetrics:
        """Collect current system metrics"""
        metrics = SystemHealthMetrics()
        
        try:
            import psutil
            
            # CPU usage
            metrics.cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.disk_percent = (disk.used / disk.total) * 100
            
            # Active processes
            metrics.active_processes = len(psutil.pids())
            
        except ImportError:
            # psutil not available, use mock values
            pass
        except Exception as e:
            print(f"Error collecting metrics: {e}")
        
        return metrics
    
    def _check_health_thresholds(self, metrics: SystemHealthMetrics) -> None:
        """Check metrics against health thresholds"""
        issues = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            issues.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            issues.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > self.thresholds["disk_percent"]:
            issues.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        if metrics.error_rate > self.thresholds["error_rate"]:
            issues.append(f"High error rate: {metrics.error_rate:.2%}")
        
        if metrics.response_time > self.thresholds["response_time"]:
            issues.append(f"Slow response time: {metrics.response_time:.2f}s")
        
        # Trigger alerts
        if issues:
            for callback in self.alert_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    print(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[SystemHealthMetrics], None]) -> None:
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    def get_latest_metrics(self) -> Optional[SystemHealthMetrics]:
        """Get latest collected metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, hours: int = 1) -> List[SystemHealthMetrics]:
        """Get metrics history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        if not self.metrics_history:
            return {"status": "no_data", "message": "No metrics collected yet"}
        
        latest = self.get_latest_metrics()
        recent_metrics = self.get_metrics_history(hours=1)
        
        # Calculate averages
        if recent_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
            avg_disk = sum(m.disk_percent for m in recent_metrics) / len(recent_metrics)
        else:
            avg_cpu = avg_memory = avg_disk = 0.0
        
        # Determine overall health status
        status = "healthy"
        if latest:
            if (latest.cpu_percent > self.thresholds["cpu_percent"] or 
                latest.memory_percent > self.thresholds["memory_percent"] or
                latest.disk_percent > self.thresholds["disk_percent"]):
                status = "warning"
            
            if (latest.cpu_percent > self.thresholds["cpu_percent"] + 10 or 
                latest.memory_percent > self.thresholds["memory_percent"] + 10 or
                latest.disk_percent > self.thresholds["disk_percent"] + 5):
                status = "critical"
        
        return {
            "status": status,
            "latest_metrics": latest.to_dict() if latest else None,
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "disk_percent": avg_disk
            },
            "thresholds": self.thresholds.copy(),
            "samples_collected": len(self.metrics_history)
        }


class SafetyManager:
    """Central safety management system"""
    
    def __init__(self, initial_mode: SafetyMode = SafetyMode.NORMAL):
        self.current_mode = initial_mode
        self.mode_history: deque = deque(maxlen=100)
        self.emergency_procedures: Dict[str, Callable] = {}
        
        # Initialize safety components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.health_monitor = SystemHealthMonitor()
        
        # Mode-specific configurations
        self.mode_configs = {
            SafetyMode.NORMAL: {
                "circuit_breaker_failures": 5,
                "rate_limit_requests_per_minute": 1000,
                "allow_network_access": True,
                "allow_file_writes": True
            },
            SafetyMode.SAFE: {
                "circuit_breaker_failures": 3,
                "rate_limit_requests_per_minute": 500,
                "allow_network_access": True,
                "allow_file_writes": False
            },
            SafetyMode.RESTRICTED: {
                "circuit_breaker_failures": 2,
                "rate_limit_requests_per_minute": 100,
                "allow_network_access": False,
                "allow_file_writes": False
            },
            SafetyMode.EMERGENCY: {
                "circuit_breaker_failures": 1,
                "rate_limit_requests_per_minute": 10,
                "allow_network_access": False,
                "allow_file_writes": False
            },
            SafetyMode.LOCKDOWN: {
                "circuit_breaker_failures": 1,
                "rate_limit_requests_per_minute": 0,
                "allow_network_access": False,
                "allow_file_writes": False
            }
        }
        
        self.logger = None  # Will be set by parent logger
        
        # Initialize default protections
        self._initialize_default_protections()
    
    def set_logger(self, logger) -> None:
        """Set logger for safety operations"""
        self.logger = logger
    
    def _initialize_default_protections(self) -> None:
        """Initialize default safety protections"""
        # Default circuit breaker for external services
        self.add_circuit_breaker(
            "external_services",
            CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30)
        )
        
        # Default rate limiter for API calls
        self.add_rate_limiter(
            "api_calls",
            max_tokens=100,  # 100 requests
            refill_rate=1.67  # ~100 per minute
        )
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
    
    def change_safety_mode(self, new_mode: SafetyMode, reason: str = "") -> None:
        """Change system safety mode"""
        if new_mode == self.current_mode:
            return
        
        old_mode = self.current_mode
        self.current_mode = new_mode
        
        # Record mode change
        self.mode_history.append({
            "timestamp": datetime.now(),
            "from_mode": old_mode.value,
            "to_mode": new_mode.value,
            "reason": reason
        })
        
        # Update component configurations
        self._apply_mode_configuration(new_mode)
        
        # Execute mode change procedures
        self._execute_mode_change_procedures(old_mode, new_mode)
        
        if self.logger:
            self.logger.info(
                f"Safety mode changed from {old_mode.value} to {new_mode.value}. Reason: {reason}"
            )
    
    def _apply_mode_configuration(self, mode: SafetyMode) -> None:
        """Apply configuration changes for safety mode"""
        config = self.mode_configs[mode]
        
        # Update circuit breaker thresholds
        for cb_name, cb in self.circuit_breakers.items():
            cb.config.failure_threshold = config["circuit_breaker_failures"]
        
        # Update rate limiter
        for rl_name, rl in self.rate_limiters.items():
            if "api" in rl_name.lower():
                requests_per_minute = config["rate_limit_requests_per_minute"]
                rl.max_tokens = requests_per_minute
                rl.refill_rate = requests_per_minute / 60.0
    
    def _execute_mode_change_procedures(self, old_mode: SafetyMode, new_mode: SafetyMode) -> None:
        """Execute procedures when changing safety modes"""
        if new_mode == SafetyMode.LOCKDOWN:
            self._execute_emergency_shutdown()
        elif old_mode == SafetyMode.LOCKDOWN and new_mode != SafetyMode.LOCKDOWN:
            self._execute_recovery_procedures()
    
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> None:
        """Add circuit breaker protection"""
        self.circuit_breakers[name] = CircuitBreaker(config)
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def add_rate_limiter(self, name: str, max_tokens: int, refill_rate: float) -> None:
        """Add rate limiter protection"""
        self.rate_limiters[name] = RateLimiter(max_tokens, refill_rate)
    
    def get_rate_limiter(self, name: str) -> Optional[RateLimiter]:
        """Get rate limiter by name"""
        return self.rate_limiters.get(name)
    
    def execute_with_protection(self, operation_name: str, func: Callable, 
                               *args, **kwargs) -> Any:
        """Execute function with all relevant protections"""
        # Check rate limiting
        for limiter_name, limiter in self.rate_limiters.items():
            if "api" in limiter_name.lower() or operation_name in limiter_name:
                if not limiter.acquire():
                    raise SafetyError(f"Rate limit exceeded for {operation_name}")
        
        # Check circuit breakers
        for cb_name, circuit_breaker in self.circuit_breakers.items():
            if cb_name in operation_name.lower() or "general" in cb_name:
                try:
                    return circuit_breaker.call(func, *args, **kwargs)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Operation {operation_name} failed: {e}")
                    
                    # Check if this failure should trigger mode change
                    if self._should_escalate_safety_mode():
                        self.change_safety_mode(
                            SafetyMode.SAFE, 
                            f"Operation {operation_name} failed with protection active"
                        )
                    raise
        
        # Execute without circuit breaker if none matched
        return func(*args, **kwargs)
    
    def _should_escalate_safety_mode(self) -> bool:
        """Check if recent failures warrant safety mode escalation"""
        # Check recent circuit breaker failures
        recent_failures = 0
        for cb in self.circuit_breakers.values():
            if cb.state == CircuitState.OPEN:
                recent_failures += 1
        
        # Escalate if multiple circuit breakers are open
        return recent_failures >= 2
    
    def trigger_emergency_shutdown(self, reason: str = "") -> None:
        """Trigger emergency shutdown procedures"""
        self.change_safety_mode(SafetyMode.LOCKDOWN, reason or "Emergency shutdown triggered")
    
    def _execute_emergency_shutdown(self) -> None:
        """Execute emergency shutdown procedures"""
        # Execute registered emergency procedures
        for name, procedure in self.emergency_procedures.items():
            try:
                procedure()
                if self.logger:
                    self.logger.info(f"Emergency procedure '{name}' executed")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Emergency procedure '{name}' failed: {e}")
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        if self.logger:
            self.logger.critical("Emergency shutdown procedures completed")
    
    def _execute_recovery_procedures(self) -> None:
        """Execute recovery procedures after lockdown"""
        # Restart health monitoring
        self.health_monitor.start_monitoring()
        
        # Reset all circuit breakers
        for cb in self.circuit_breakers.values():
            cb.reset()
        
        # Reset all rate limiters
        for rl in self.rate_limiters.values():
            rl.reset()
        
        if self.logger:
            self.logger.info("Recovery procedures completed")
    
    def register_emergency_procedure(self, name: str, procedure: Callable) -> None:
        """Register emergency procedure"""
        self.emergency_procedures[name] = procedure
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status"""
        return {
            "current_mode": self.current_mode.value,
            "mode_history": [
                {
                    "timestamp": entry["timestamp"].isoformat(),
                    "from_mode": entry["from_mode"],
                    "to_mode": entry["to_mode"],
                    "reason": entry["reason"]
                }
                for entry in self.mode_history
            ],
            "circuit_breakers": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
            "rate_limiters": {
                name: rl.get_stats() for name, rl in self.rate_limiters.items()
            },
            "health_status": self.health_monitor.get_health_summary(),
            "emergency_procedures": list(self.emergency_procedures.keys())
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = self.health_monitor.get_health_summary()
        
        # Determine if safety mode should be adjusted
        if health_status["status"] == "critical":
            if self.current_mode == SafetyMode.NORMAL:
                self.change_safety_mode(
                    SafetyMode.SAFE, 
                    "System health critical detected"
                )
        elif health_status["status"] == "warning":
            if self.current_mode == SafetyMode.NORMAL:
                self.change_safety_mode(
                    SafetyMode.RESTRICTED, 
                    "System health warning detected"
                )
        
        return health_status
    
    def shutdown(self) -> None:
        """Shutdown safety manager"""
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        # Record shutdown
        self.mode_history.append({
            "timestamp": datetime.now(),
            "from_mode": self.current_mode.value,
            "to_mode": "shutdown",
            "reason": "Safety manager shutdown"
        })
        
        if self.logger:
            self.logger.info("Safety manager shutdown completed")


# Utility functions

def create_safety_manager(mode: SafetyMode = SafetyMode.NORMAL) -> SafetyManager:
    """Create and configure safety manager"""
    return SafetyManager(mode)


def create_circuit_breaker(failure_threshold: int = 5, 
                          recovery_timeout: int = 60) -> CircuitBreaker:
    """Create circuit breaker with custom configuration"""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout
    )
    return CircuitBreaker(config)


def create_rate_limiter(requests_per_minute: int = 100) -> RateLimiter:
    """Create rate limiter with specified requests per minute"""
    return RateLimiter(
        max_tokens=requests_per_minute,
        refill_rate=requests_per_minute / 60.0
    )


# Export main classes and functions
__all__ = [
    "SafetyManager",
    "CircuitBreaker",
    "RateLimiter",
    "SystemHealthMonitor",
    "SafetyMode",
    "CircuitState",
    "CircuitBreakerConfig",
    "SystemHealthMetrics",
    "create_safety_manager",
    "create_circuit_breaker",
    "create_rate_limiter"
]