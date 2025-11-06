#!/usr/bin/env python3
"""
Audit Logging and Security Monitoring Module

This module provides comprehensive audit logging with structured logging,
event monitoring, compliance reporting, and real-time security alerts.

Author: Mini-Biai Framework Team
Version: 1.0.0
Date: 2024
"""

import gzip
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable

from ..core.exceptions import AuditError


class EventType(Enum):
    """Types of audit events"""
    TOOL_EXECUTION = "tool_execution"
    PERMISSION_CHECK = "permission_check"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    USER_ACTION = "user_action"
    COMMAND_VALIDATION = "command_validation"
    SANDBOX_EXECUTION = "sandbox_execution"
    CIRCUIT_BREAKER = "circuit_breaker"
    RATE_LIMIT = "rate_limit"
    CONFIGURATION_CHANGE = "configuration_change"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    API_CALL = "api_call"
    SYSTEM_SHUTDOWN = "system_shutdown"


class AuditLevel(Enum):
    """Audit log severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


@dataclass
class AuditEntry:
    """Individual audit log entry"""
    event_id: str
    timestamp: datetime
    event_type: EventType
    level: AuditLevel
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    severity_score: float = 0.0  # 0.0 to 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit entry to dictionary"""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "level": self.level.value,
            "message": self.message,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "source": self.source,
            "details": self.details,
            "tags": list(self.tags),
            "severity_score": self.severity_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEntry':
        """Create audit entry from dictionary"""
        return cls(
            event_id=data["event_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=EventType(data["event_type"]),
            level=AuditLevel(data["level"]),
            message=data["message"],
            user_id=data.get("user_id"),
            session_id=data.get("session_id"),
            source=data.get("source"),
            details=data.get("details", {}),
            tags=set(data.get("tags", [])),
            severity_score=data.get("severity_score", 0.0)
        )


@dataclass
class ComplianceReport:
    """Security compliance report"""
    report_id: str
    generated_at: datetime
    time_period: timedelta
    total_events: int
    security_violations: int
    critical_events: int
    compliance_score: float
    top_violations: List[Dict[str, Any]]
    recommendations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


class AuditLogger:
    """Centralized audit logging system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Log configuration
        self.log_dir = self.config.get("log_dir", "logs/audit")
        self.max_file_size = self.config.get("max_file_size_mb", 100)
        self.backup_count = self.config.get("backup_count", 5)
        self.retention_days = self.config.get("retention_days", 90)
        
        # Event queue and processing
        self.event_queue: deque = deque(maxlen=10000)
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.processing = False
        
        # Statistics
        self.stats = {
            "events_logged": 0,
            "events_processed": 0,
            "queue_overflows": 0,
            "processing_errors": 0
        }
        
        # Security monitoring
        self.violation_counters = defaultdict(int)
        self.alert_thresholds = self.config.get("alert_thresholds", {
            "security_violations_per_hour": 10,
            "failed_authentications_per_hour": 50,
            "critical_errors_per_hour": 5
        })
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[AuditEntry], None]] = []
        
        # Initialize logging
        self._setup_logging()
        self._start_processing()
    
    def _setup_logging(self) -> None:
        """Setup logging infrastructure"""
        # Create log directory
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup file handler
        log_file = os.path.join(self.log_dir, "audit.log")
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size * 1024 * 1024,  # Convert MB to bytes
            backupCount=self.backup_count
        )
        file_handler.setFormatter(formatter)
        
        # Console handler for critical events
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.CRITICAL)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Create audit logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.DEBUG)
    
    def _start_processing(self) -> None:
        """Start background event processing"""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.processing = True
        self.shutdown_event.clear()
        self.processing_thread = threading.Thread(
            target=self._process_events,
            daemon=True
        )
        self.processing_thread.start()
    
    def _process_events(self) -> None:
        """Background thread for processing audit events"""
        while not self.shutdown_event.is_set():
            try:
                # Process events in batch
                events_to_process = []
                
                # Collect events from queue
                while self.event_queue and len(events_to_process) < 100:
                    events_to_process.append(self.event_queue.popleft())
                
                if events_to_process:
                    self._write_events(events_to_process)
                    self.stats["events_processed"] += len(events_to_process)
                
                # Sleep briefly
                time.sleep(0.1)
                
            except Exception as e:
                self.stats["processing_errors"] += 1
                self.logger.error(f"Error processing audit events: {e}")
    
    def _write_events(self, events: List[AuditEntry]) -> None:
        """Write events to log files"""
        try:
            # Write to JSON log file
            json_log_file = os.path.join(self.log_dir, "audit.jsonl")
            
            with open(json_log_file, "a", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            
            # Handle compression and cleanup
            self._rotate_logs()
            
        except Exception as e:
            self.logger.error(f"Failed to write audit events: {e}")
    
    def _rotate_logs(self) -> None:
        """Rotate and cleanup log files"""
        try:
            current_time = datetime.now()
            
            # Compress old log files
            for file_path in Path(self.log_dir).glob("audit*.log.*"):
                if not file_path.name.endswith(".gz"):
                    with open(file_path, "rb") as f_in:
                        with gzip.open(str(file_path) + ".gz", "wb") as f_out:
                            f_out.writelines(f_in)
                    file_path.unlink()
            
            # Remove old files
            cutoff_time = current_time - timedelta(days=self.retention_days)
            for file_path in Path(self.log_dir).glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        
        except Exception as e:
            self.logger.error(f"Error during log rotation: {e}")
    
    def log_event(self, event_type: EventType, level: AuditLevel, message: str,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 source: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 tags: Optional[Set[str]] = None,
                 severity_score: Optional[float] = None) -> str:
        """Log an audit event"""
        import uuid
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Calculate severity score if not provided
        if severity_score is None:
            severity_score = self._calculate_severity(event_type, level, message)
        
        # Create audit entry
        entry = AuditEntry(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            level=level,
            message=message,
            user_id=user_id,
            session_id=session_id,
            source=source,
            details=details or {},
            tags=tags or set(),
            severity_score=severity_score
        )
        
        # Add to queue
        try:
            self.event_queue.append(entry)
            self.stats["events_logged"] += 1
        except IndexError:
            # Queue overflow
            self.stats["queue_overflows"] += 1
            if self.event_queue:
                self.event_queue.popleft()  # Remove oldest
                self.event_queue.append(entry)
        
        # Update violation counters
        if event_type == EventType.SECURITY_VIOLATION:
            self.violation_counters["security_violations"] += 1
        elif event_type == EventType.AUTHENTICATION and level == AuditLevel.WARNING:
            self.violation_counters["failed_authentications"] += 1
        elif level == AuditLevel.CRITICAL:
            self.violation_counters["critical_errors"] += 1
        
        # Check for alerts
        self._check_alerts(entry)
        
        # Log to standard logger for immediate visibility
        log_level = self._map_audit_level(level)
        self.logger.log(log_level, f"[{event_type.value}] {message}")
        
        return event_id
    
    def _calculate_severity(self, event_type: EventType, level: AuditLevel, message: str) -> float:
        """Calculate severity score for event"""
        base_score = {
            AuditLevel.DEBUG: 1.0,
            AuditLevel.INFO: 2.0,
            AuditLevel.WARNING: 4.0,
            AuditLevel.ERROR: 6.0,
            AuditLevel.CRITICAL: 8.0,
            AuditLevel.SECURITY: 9.0
        }.get(level, 1.0)
        
        # Adjust based on event type
        if event_type == EventType.SECURITY_VIOLATION:
            base_score = max(base_score, 7.0)
        elif event_type in [EventType.SYSTEM_SHUTDOWN, EventType.AUTHENTICATION]:
            base_score += 1.0
        
        # Check for critical keywords
        critical_keywords = ["critical", "emergency", "breach", "attack", "exploit", "vulnerability"]
        if any(keyword in message.lower() for keyword in critical_keywords):
            base_score = min(10.0, base_score + 2.0)
        
        return min(10.0, base_score)
    
    def _map_audit_level(self, level: AuditLevel) -> int:
        """Map audit level to logging level"""
        mapping = {
            AuditLevel.DEBUG: logging.DEBUG,
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.ERROR: logging.ERROR,
            AuditLevel.CRITICAL: logging.CRITICAL,
            AuditLevel.SECURITY: logging.CRITICAL  # Treat security as critical
        }
        return mapping.get(level, logging.INFO)
    
    def _check_alerts(self, entry: AuditEntry) -> None:
        """Check if event triggers alerts"""
        current_time = datetime.now()
        hour_ago = current_time - timedelta(hours=1)
        
        # Check violation thresholds
        for violation_type, count in self.violation_counters.items():
            threshold_key = f"{violation_type}_per_hour"
            if threshold_key in self.alert_thresholds:
                threshold = self.alert_thresholds[threshold_key]
                if count >= threshold:
                    self._trigger_alert(
                        f"{violation_type.replace('_', ' ').title()} threshold exceeded: {count} >= {threshold}",
                        entry
                    )
        
        # Check individual event severity
        if entry.severity_score >= 8.0:
            self._trigger_alert(f"High severity event: {entry.message}", entry)
        
        # Trigger callback alerts
        for callback in self.alert_callbacks:
            try:
                callback(entry)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _trigger_alert(self, alert_message: str, related_entry: AuditEntry) -> None:
        """Trigger security alert"""
        self.log_event(
            EventType.SECURITY_VIOLATION,
            AuditLevel.CRITICAL,
            f"SECURITY ALERT: {alert_message}",
            user_id=related_entry.user_id,
            session_id=related_entry.session_id,
            details={
                "alert_type": "threshold_exceeded",
                "related_event_id": related_entry.event_id,
                "related_event_type": related_entry.event_type.value
            },
            severity_score=10.0
        )
    
    def add_alert_callback(self, callback: Callable[[AuditEntry], None]) -> None:
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def get_events(self, start_time: Optional[datetime] = None, 
                  end_time: Optional[datetime] = None,
                  event_type: Optional[EventType] = None,
                  level: Optional[AuditLevel] = None,
                  user_id: Optional[str] = None,
                  limit: int = 1000) -> List[AuditEntry]:
        """Retrieve audit events with filtering"""
        events = []
        
        # Read from JSON log file
        json_log_file = os.path.join(self.log_dir, "audit.jsonl")
        
        if not os.path.exists(json_log_file):
            return events
        
        try:
            with open(json_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = AuditEntry.from_dict(data)
                        
                        # Apply filters
                        if start_time and entry.timestamp < start_time:
                            continue
                        if end_time and entry.timestamp > end_time:
                            continue
                        if event_type and entry.event_type != event_type:
                            continue
                        if level and entry.level != level:
                            continue
                        if user_id and entry.user_id != user_id:
                            continue
                        
                        events.append(entry)
                        
                        if len(events) >= limit:
                            break
                            
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading audit log: {e}")
        
        return events
    
    def get_security_events(self, hours: int = 24) -> List[AuditEntry]:
        """Get security-related events from recent time period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        return self.get_events(
            start_time=start_time,
            end_time=end_time,
            event_type=EventType.SECURITY_VIOLATION
        )
    
    def generate_compliance_report(self, hours: int = 24) -> ComplianceReport:
        """Generate compliance report for time period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get all events in period
        events = self.get_events(start_time=start_time, end_time=end_time, limit=100000)
        
        # Calculate statistics
        total_events = len(events)
        security_violations = len([e for e in events if e.event_type == EventType.SECURITY_VIOLATION])
        critical_events = len([e for e in events if e.level in [AuditLevel.CRITICAL, AuditLevel.SECURITY]])
        
        # Calculate compliance score (0.0 to 1.0)
        if total_events == 0:
            compliance_score = 1.0
        else:
            # Deduct points for violations and critical events
            violation_penalty = (security_violations / total_events) * 0.5
            critical_penalty = (critical_events / total_events) * 0.3
            compliance_score = max(0.0, 1.0 - violation_penalty - critical_penalty)
        
        # Top violations
        violation_counts = defaultdict(int)
        for event in events:
            if event.event_type == EventType.SECURITY_VIOLATION:
                violation_counts[event.message] += 1
        
        top_violations = [
            {"message": msg, "count": count}
            for msg, count in sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        # Generate recommendations
        recommendations = []
        if security_violations > 0:
            recommendations.append("Review and address security violations")
        if critical_events > 0:
            recommendations.append("Investigate and resolve critical errors")
        if compliance_score < 0.8:
            recommendations.append("Improve system security and monitoring")
        recommendations.extend([
            "Implement additional security controls",
            "Regular security audits and penetration testing",
            "Update security policies and procedures"
        ])
        
        return ComplianceReport(
            report_id=str(uuid.uuid4()),
            generated_at=end_time,
            time_period=timedelta(hours=hours),
            total_events=total_events,
            security_violations=security_violations,
            critical_events=critical_events,
            compliance_score=compliance_score,
            top_violations=top_violations,
            recommendations=recommendations,
            details={
                "events_by_type": {et.value: len([e for e in events if e.event_type == et]) 
                                 for et in EventType},
                "events_by_level": {al.value: len([e for e in events if e.level == al]) 
                                   for al in AuditLevel}
            }
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit system statistics"""
        return {
            **self.stats,
            "current_queue_size": len(self.event_queue),
            "violation_counters": dict(self.violation_counters),
            "log_directory": self.log_dir,
            "config": self.config
        }
    
    def reset_statistics(self) -> None:
        """Reset statistics counters"""
        self.stats = {
            "events_logged": 0,
            "events_processed": 0,
            "queue_overflows": 0,
            "processing_errors": 0
        }
        self.violation_counters.clear()
    
    def shutdown(self) -> None:
        """Shutdown audit logger"""
        self.shutdown_event.set()
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Process remaining events
        if self.event_queue:
            remaining_events = list(self.event_queue)
            self.event_queue.clear()
            self._write_events(remaining_events)
        
        self.processing = False
        
        # Log shutdown
        self.log_event(
            EventType.SYSTEM_SHUTDOWN,
            AuditLevel.INFO,
            "Audit logger shutdown completed"
        )


class MonitoringSystem:
    """Real-time monitoring and alerting system"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.monitoring_active = False
        self.monitored_metrics = set()
        self.alert_rules = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._shutdown_event.clear()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        
        self.audit_logger.log_event(
            EventType.SYSTEM_SHUTDOWN,  # Using existing event type
            AuditLevel.INFO,
            "Real-time monitoring started"
        )
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring"""
        self.monitoring_active = False
        self._shutdown_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active and not self._shutdown_event.is_set():
            try:
                # Check alert rules
                self._check_alert_rules()
                
                # Monitor system health
                self._check_system_health()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.audit_logger.log_event(
                    EventType.SYSTEM_ERROR,
                    AuditLevel.ERROR,
                    f"Monitoring system error: {e}"
                )
    
    def _check_alert_rules(self) -> None:
        """Check configured alert rules"""
        # Implementation would check various metrics against thresholds
        pass
    
    def _check_system_health(self) -> None:
        """Check overall system health"""
        # Get audit system statistics
        stats = self.audit_logger.get_statistics()
        
        # Check for issues
        if stats["processing_errors"] > 10:
            self.audit_logger.log_event(
                EventType.SYSTEM_ERROR,
                AuditLevel.WARNING,
                f"High number of processing errors: {stats['processing_errors']}"
            )
        
        if stats["queue_overflows"] > 5:
            self.audit_logger.log_event(
                EventType.SYSTEM_ERROR,
                AuditLevel.WARNING,
                f"Frequent queue overflows: {stats['queue_overflows']}"
            )


# Utility functions

def log_security_event(audit_logger: AuditLogger, event_type: EventType, message: str,
                      user_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      **kwargs) -> str:
    """Convenience function to log security events"""
    return audit_logger.log_event(
        event_type=event_type,
        level=AuditLevel.SECURITY,
        message=message,
        user_id=user_id,
        session_id=session_id,
        **kwargs
    )


def create_audit_logger(config: Optional[Dict[str, Any]] = None) -> AuditLogger:
    """Create and configure audit logger"""
    default_config = {
        "log_dir": "logs/audit",
        "max_file_size_mb": 100,
        "backup_count": 5,
        "retention_days": 90,
        "alert_thresholds": {
            "security_violations_per_hour": 10,
            "failed_authentications_per_hour": 50,
            "critical_errors_per_hour": 5
        }
    }
    
    if config:
        default_config.update(config)
    
    return AuditLogger(default_config)


# Export main classes and functions
__all__ = [
    "AuditLogger",
    "AuditEntry",
    "ComplianceReport",
    "MonitoringSystem",
    "EventType",
    "AuditLevel",
    "log_security_event",
    "create_audit_logger"
]