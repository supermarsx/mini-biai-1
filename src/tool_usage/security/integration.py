#!/usr/bin/env python3
"""
Security Framework Integration Module

This module provides unified integration of all security components including
validator, sandbox, permissions, audit, and safety systems.

Author: Mini-Biai Framework Team
Version: 1.0.0
Date: 2024
"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

from ..core.exceptions import SecurityError, ValidationError, PermissionError

# Import all security components
from .validator import CommandValidator, ValidationResult, ValidationLevel, ThreatLevel
from .sandbox import SandboxManager, SandboxConfig, ExecutionResult
from .permissions import PermissionChecker, OperationType, PermissionLevel
from .audit import AuditLogger, EventType, AuditLevel, AuditEntry
from .safety_modes import SafetyManager, SafetyMode, CircuitBreaker


class IntegrationLevel(Enum):
    """Integration security levels"""
    BASIC = "basic"         # Minimal integration
    STANDARD = "standard"   # Balanced security
    STRICT = "strict"       # High security
    MAXIMUM = "maximum"     # Maximum security


@dataclass
class SecurityConfiguration:
    """Integrated security configuration"""
    integration_level: IntegrationLevel = IntegrationLevel.STANDARD
    
    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    enable_threat_detection: bool = True
    enable_input_sanitization: bool = True
    
    # Sandbox settings
    enable_sandbox: bool = True
    sandbox_isolation: str = "strict"
    max_memory_mb: int = 256
    max_execution_time_sec: int = 30
    
    # Permission settings
    enable_permissions: bool = True
    require_authentication: bool = True
    session_timeout_minutes: int = 30
    
    # Audit settings
    enable_audit: bool = True
    log_all_operations: bool = True
    alert_on_security_violations: bool = True
    
    # Safety settings
    enable_safety_modes: bool = True
    auto_escalate_on_failures: bool = True
    health_monitoring: bool = True
    
    # Integration settings
    enable_cross_component_validation: bool = True
    shared_session_management: bool = True
    emergency_shutdown_enabled: bool = True


class SecurityFramework:
    """Unified security framework integrating all components"""
    
    def __init__(self, config: Optional[SecurityConfiguration] = None):
        self.config = config or SecurityConfiguration()
        
        # Initialize components
        self.validator = CommandValidator(self.config.validation_level)
        self.sandbox = SandboxManager(self._create_sandbox_config())
        self.permissions = PermissionChecker()
        self.audit = AuditLogger(self._create_audit_config())
        self.safety = SafetyManager(self._get_initial_safety_mode())
        
        # Integration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.component_status = {
            "validator": True,
            "sandbox": True,
            "permissions": True,
            "audit": True,
            "safety": True
        }
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = {}
        self._setup_event_handlers()
        
        # Set up cross-component logging
        self._setup_logging()
        
        # Start background monitoring
        if self.config.health_monitoring:
            self._start_monitoring()
    
    def _create_sandbox_config(self) -> SandboxConfig:
        """Create sandbox configuration from security config"""
        from .sandbox import IsolationLevel
        
        isolation_map = {
            "basic": IsolationLevel.BASIC,
            "strict": IsolationLevel.STRICT,
            "container": IsolationLevel.CONTAINER
        }
        
        return SandboxConfig(
            isolation_level=isolation_map.get(self.config.sandbox_isolation, IsolationLevel.STRICT),
            max_memory_mb=self.config.max_memory_mb,
            max_wall_time_sec=self.config.max_execution_time_sec,
            max_cpu_time_sec=int(self.config.max_execution_time_sec * 0.8)
        )
    
    def _create_audit_config(self) -> Dict[str, Any]:
        """Create audit configuration from security config"""
        return {
            "log_dir": "logs/security",
            "max_file_size_mb": 100,
            "backup_count": 10,
            "retention_days": 180,
            "alert_thresholds": {
                "security_violations_per_hour": 5,
                "failed_authentications_per_hour": 20,
                "critical_errors_per_hour": 3
            } if self.config.alert_on_security_violations else {}
        }
    
    def _get_initial_safety_mode(self) -> SafetyMode:
        """Get initial safety mode based on integration level"""
        mode_map = {
            IntegrationLevel.BASIC: SafetyMode.NORMAL,
            IntegrationLevel.STANDARD: SafetyMode.NORMAL,
            IntegrationLevel.STRICT: SafetyMode.SAFE,
            IntegrationLevel.MAXIMUM: SafetyMode.RESTRICTED
        }
        return mode_map.get(self.config.integration_level, SafetyMode.NORMAL)
    
    def _setup_logging(self) -> None:
        """Setup logging across all components"""
        # Set logger for all components
        logger = self.audit.logger
        
        self.sandbox.set_logger(logger)
        self.permissions.set_logger(logger)
        self.safety.set_logger(logger)
        self.validator.logger = logger
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for cross-component communication"""
        # Security violation handler
        self.add_event_handler("security_violation", self._handle_security_violation)
        
        # Component failure handler
        self.add_event_handler("component_failure", self._handle_component_failure)
        
        # Emergency shutdown handler
        self.add_event_handler("emergency_shutdown", self._handle_emergency_shutdown)
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for framework events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Trigger framework event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    self.audit.log_event(
                        EventType.SYSTEM_ERROR,
                        AuditLevel.ERROR,
                        f"Event handler failed for {event_type}: {e}",
                        details={"error": str(e), "event_data": event_data}
                    )
    
    def validate_and_execute(self, command: str, session_id: Optional[str] = None,
                           user_id: Optional[str] = None,
                           require_permission: bool = True,
                           force_sandbox: bool = False) -> Dict[str, Any]:
        """
        Complete security validation and execution pipeline
        
        Args:
            command: Command to validate and execute
            session_id: User session ID
            user_id: User ID
            require_permission: Whether to check permissions
            force_sandbox: Whether to force sandbox execution
            
        Returns:
            Dictionary with execution results and security information
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"
        
        # Log start of operation
        self.audit.log_event(
            EventType.TOOL_EXECUTION,
            AuditLevel.INFO,
            f"Starting secure command execution: {command[:100]}...",
            user_id=user_id,
            session_id=session_id,
            details={
                "execution_id": execution_id,
                "require_permission": require_permission,
                "force_sandbox": force_sandbox,
                "integration_level": self.config.integration_level.value
            }
        )
        
        try:
            # Step 1: Command validation
            if not self.component_status["validator"]:
                raise SecurityError("Validator component is not available")
            
            validation_result = self.validator.validate_command(command)
            
            # Log validation results
            self.audit.log_event(
                EventType.COMMAND_VALIDATION,
                AuditLevel.WARNING if not validation_result.is_valid else AuditLevel.INFO,
                f"Command validation: {validation_result.threat_level.value} - {validation_result.confidence:.2f}",
                user_id=user_id,
                session_id=session_id,
                details={
                    "execution_id": execution_id,
                    "is_valid": validation_result.is_valid,
                    "threat_level": validation_result.threat_level.value,
                    "confidence": validation_result.confidence,
                    "issues": validation_result.issues,
                    "suggestions": validation_result.suggestions
                }
            )
            
            # Step 2: Permission check
            if require_permission and self.component_status["permissions"]:
                if not session_id:
                    raise PermissionError("Session ID required for permission check")
                
                self.permissions.require_permission(
                    session_id, OperationType.SYSTEM_EXECUTE, PermissionLevel.EXECUTE
                )
                
                self.audit.log_event(
                    EventType.AUTHORIZATION,
                    AuditLevel.INFO,
                    "Permission check passed",
                    user_id=user_id,
                    session_id=session_id,
                    details={"execution_id": execution_id}
                )
            
            # Step 3: Safety mode check
            if self.component_status["safety"]:
                if self.safety.current_mode == SafetyMode.LOCKDOWN:
                    raise SecurityError("System is in lockdown mode - operations not allowed")
                
                # Check if operation should be blocked by current safety mode
                if self.safety.current_mode == SafetyMode.EMERGENCY:
                    if not any(keyword in command.lower() for keyword in ["emergency", "shutdown", "kill"]):
                        self.audit.log_event(
                            EventType.SECURITY_VIOLATION,
                            AuditLevel.CRITICAL,
                            f"Operation blocked in emergency mode: {command[:100]}...",
                            user_id=user_id,
                            session_id=session_id,
                            details={"execution_id": execution_id},
                            severity_score=10.0
                        )
                        raise SecurityError("Operation not allowed in emergency mode")
            
            # Step 4: Execution
            execution_result = None
            execution_error = None
            
            # Determine execution method
            should_sandbox = (
                force_sandbox or 
                self.config.enable_sandbox or
                (not validation_result.is_valid and validation_result.threat_level != ThreatLevel.LOW)
            )
            
            if should_sandbox and self.component_status["sandbox"]:
                # Execute in sandbox
                try:
                    execution_result = self.safety.execute_with_protection(
                        "sandbox_execution",
                        self.sandbox.execute_command,
                        command,
                        timeout=self.config.max_execution_time_sec
                    )
                    
                    self.audit.log_event(
                        EventType.SANDBOX_EXECUTION,
                        AuditLevel.INFO,
                        "Command executed in sandbox",
                        user_id=user_id,
                        session_id=session_id,
                        details={
                            "execution_id": execution_id,
                            "success": execution_result.success,
                            "execution_time": execution_result.execution_time,
                            "exit_code": execution_result.exit_code
                        }
                    )
                    
                except Exception as e:
                    execution_error = str(e)
                    self.audit.log_event(
                        EventType.SANDBOX_EXECUTION,
                        AuditLevel.ERROR,
                        f"Sandbox execution failed: {e}",
                        user_id=user_id,
                        session_id=session_id,
                        details={"execution_id": execution_id, "error": str(e)}
                    )
            else:
                # Direct execution (not recommended)
                self.audit.log_event(
                    EventType.TOOL_EXECUTION,
                    AuditLevel.WARNING,
                    "Command executed without sandbox protection",
                    user_id=user_id,
                    session_id=session_id,
                    details={"execution_id": execution_id}
                )
                
                # Note: In production, you would implement direct execution here
                # For security reasons, we'll raise an error
                raise SecurityError("Direct execution not implemented for security reasons")
            
            # Step 5: Final security audit
            execution_time = time.time() - start_time
            
            if execution_result and execution_result.success:
                self.audit.log_event(
                    EventType.TOOL_EXECUTION,
                    AuditLevel.INFO,
                    "Command execution completed successfully",
                    user_id=user_id,
                    session_id=session_id,
                    details={
                        "execution_id": execution_id,
                        "execution_time": execution_time,
                        "exit_code": execution_result.exit_code,
                        "output_length": len(execution_result.stdout) + len(execution_result.stderr)
                    }
                )
            else:
                self.audit.log_event(
                    EventType.SYSTEM_ERROR,
                    AuditLevel.ERROR,
                    f"Command execution failed: {execution_error or 'Unknown error'}",
                    user_id=user_id,
                    session_id=session_id,
                    details={
                        "execution_id": execution_id,
                        "execution_time": execution_time,
                        "error": execution_error
                    }
                )
            
            # Return comprehensive result
            return {
                "execution_id": execution_id,
                "success": execution_result.success if execution_result else False,
                "exit_code": execution_result.exit_code if execution_result else None,
                "stdout": execution_result.stdout if execution_result else "",
                "stderr": execution_result.stderr if execution_result else execution_error or "",
                "execution_time": execution_time,
                "validation": {
                    "is_valid": validation_result.is_valid,
                    "threat_level": validation_result.threat_level.value,
                    "confidence": validation_result.confidence,
                    "issues": validation_result.issues,
                    "suggestions": validation_result.suggestions
                },
                "execution_method": "sandbox" if should_sandbox else "direct",
                "security_score": self._calculate_security_score(validation_result, execution_result),
                "component_status": self.component_status.copy()
            }
            
        except Exception as e:
            # Log the error
            execution_time = time.time() - start_time
            
            self.audit.log_event(
                EventType.SYSTEM_ERROR,
                AuditLevel.ERROR,
                f"Secure execution failed: {e}",
                user_id=user_id,
                session_id=session_id,
                details={
                    "execution_id": execution_id,
                    "execution_time": execution_time,
                    "error_type": type(e).__name__
                }
            )
            
            # Trigger security violation event if it's a security-related error
            if isinstance(e, (SecurityError, ValidationError, PermissionError)):
                self._trigger_event("security_violation", {
                    "execution_id": execution_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "command": command[:100],
                    "user_id": user_id,
                    "session_id": session_id
                })
            
            raise
    
    def _calculate_security_score(self, validation_result: ValidationResult, 
                                 execution_result: Optional[ExecutionResult]) -> float:
        """Calculate overall security score for execution"""
        score = 1.0
        
        # Deduct for validation issues
        if not validation_result.is_valid:
            score -= 0.3
        elif validation_result.threat_level != ThreatLevel.NONE:
            threat_penalty = {
                ThreatLevel.LOW: 0.1,
                ThreatLevel.MEDIUM: 0.2,
                ThreatLevel.HIGH: 0.4,
                ThreatLevel.CRITICAL: 0.6
            }.get(validation_result.threat_level, 0.0)
            score -= threat_penalty
        
        # Deduct for low confidence
        score -= (1.0 - validation_result.confidence) * 0.2
        
        # Deduct for execution failures
        if execution_result and not execution_result.success:
            score -= 0.4
        
        # Deduct for sandbox violations
        if execution_result and execution_result.errors:
            score -= len(execution_result.errors) * 0.1
        
        return max(0.0, score)
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and create session"""
        try:
            session_id = self.permissions.authenticate_user(username, password)
            
            if session_id:
                # Create integrated session
                self.active_sessions[session_id] = {
                    "user_id": None,  # Will be set after session creation
                    "username": username,
                    "created_at": datetime.now(),
                    "last_activity": datetime.now(),
                    "security_context": None
                }
                
                # Find user ID
                for user in self.permissions.users.values():
                    if user.username == username:
                        self.active_sessions[session_id]["user_id"] = user.user_id
                        break
                
                self.audit.log_event(
                    EventType.AUTHENTICATION,
                    AuditLevel.INFO,
                    f"User authenticated: {username}",
                    user_id=self.active_sessions[session_id]["user_id"],
                    session_id=session_id
                )
            
            return session_id
            
        except Exception as e:
            self.audit.log_event(
                EventType.AUTHENTICATION,
                AuditLevel.WARNING,
                f"Authentication failed for {username}: {e}",
                details={"username": username}
            )
            return None
    
    def check_permission(self, session_id: str, operation: OperationType,
                        min_level: PermissionLevel = PermissionLevel.READ,
                        resource: Optional[str] = None) -> bool:
        """Check user permission with integrated audit"""
        try:
            has_permission = self.permissions.check_permission(session_id, operation, min_level, resource)
            
            # Log permission check
            self.audit.log_event(
                EventType.PERMISSION_CHECK,
                AuditLevel.INFO if has_permission else AuditLevel.WARNING,
                f"Permission check: {operation.value} ({min_level.name}) - {'GRANTED' if has_permission else 'DENIED'}",
                user_id=self.active_sessions.get(session_id, {}).get("user_id"),
                session_id=session_id,
                details={
                    "operation": operation.value,
                    "min_level": min_level.name,
                    "resource": resource,
                    "granted": has_permission
                }
            )
            
            return has_permission
            
        except Exception as e:
            self.audit.log_event(
                EventType.SYSTEM_ERROR,
                AuditLevel.ERROR,
                f"Permission check error: {e}",
                session_id=session_id,
                details={"operation": operation.value, "error": str(e)}
            )
            return False
    
    def log_security_event(self, event_type: EventType, message: str,
                          user_id: Optional[str] = None,
                          session_id: Optional[str] = None,
                          **kwargs) -> str:
        """Log security event with framework context"""
        return self.audit.log_event(
            event_type=event_type,
            level=AuditLevel.SECURITY,
            message=message,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security framework status"""
        return {
            "framework_version": "1.0.0",
            "integration_level": self.config.integration_level.value,
            "component_status": self.component_status.copy(),
            "active_sessions": len(self.active_sessions),
            "current_safety_mode": self.safety.current_mode.value,
            "validator_config": {
                "validation_level": self.validator.validation_level.value,
                "active_rules": len(self.validator.validation_rules)
            },
            "sandbox_config": self.sandbox.get_sandbox_info(),
            "permissions_config": {
                "active_users": len(self.permissions.users),
                "active_sessions": len(self.permissions.active_sessions),
                "pending_escalations": len(self.permissions.pending_escalations)
            },
            "audit_config": self.audit.get_statistics(),
            "safety_config": self.safety.get_safety_status()
        }
    
    def _handle_security_violation(self, event_data: Dict[str, Any]) -> None:
        """Handle security violation events"""
        # Check if escalation is needed
        violation_count = len(self.audit.get_security_events(hours=1))
        
        if violation_count >= 5:  # Threshold for escalation
            self.safety.change_safety_mode(
                SafetyMode.RESTRICTED,
                f"Multiple security violations detected: {violation_count} in last hour"
            )
    
    def _handle_component_failure(self, event_data: Dict[str, Any]) -> None:
        """Handle component failure events"""
        failed_component = event_data.get("component")
        if failed_component in self.component_status:
            self.component_status[failed_component] = False
            
            # Escalate safety mode
            self.safety.change_safety_mode(
                SafetyMode.SAFE,
                f"Component failure: {failed_component}"
            )
    
    def _handle_emergency_shutdown(self, event_data: Dict[str, Any]) -> None:
        """Handle emergency shutdown events"""
        reason = event_data.get("reason", "Emergency shutdown triggered")
        self.safety.trigger_emergency_shutdown(reason)
    
    def _start_monitoring(self) -> None:
        """Start background monitoring"""
        def monitoring_loop():
            while True:
                try:
                    # Check system health
                    health_status = self.safety.check_system_health()
                    
                    # Cleanup expired sessions
                    expired_count = self.permissions.cleanup_expired_sessions()
                    
                    # Clean up framework sessions
                    current_time = datetime.now()
                    expired_framework_sessions = [
                        sid for sid, session in self.active_sessions.items()
                        if (current_time - session["last_activity"]).total_seconds() > (self.config.session_timeout_minutes * 60)
                    ]
                    
                    for sid in expired_framework_sessions:
                        del self.active_sessions[sid]
                    
                    if expired_count > 0 or expired_framework_sessions:
                        self.audit.log_event(
                            EventType.SYSTEM_SHUTDOWN,
                            AuditLevel.INFO,
                            f"Cleaned up expired sessions: {expired_count + len(expired_framework_sessions)}"
                        )
                    
                    time.sleep(60)  # Monitor every minute
                    
                except Exception as e:
                    self.audit.log_event(
                        EventType.SYSTEM_ERROR,
                        AuditLevel.ERROR,
                        f"Monitoring loop error: {e}"
                    )
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()
    
    def update_configuration(self, **kwargs) -> None:
        """Update security configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                
                # Update relevant components
                if key == "validation_level":
                    self.validator.validation_level = value
                elif key in ["max_memory_mb", "max_execution_time_sec", "sandbox_isolation"]:
                    self.sandbox.config = self._create_sandbox_config()
                elif key == "enable_audit":
                    if not value:
                        self.component_status["audit"] = False
                
                self.audit.log_event(
                    EventType.CONFIGURATION_CHANGE,
                    AuditLevel.INFO,
                    f"Configuration updated: {key} = {value}"
                )
    
    def emergency_shutdown(self, reason: str = "") -> None:
        """Trigger emergency shutdown"""
        self._trigger_event("emergency_shutdown", {"reason": reason})
        
        self.audit.log_event(
            EventType.SYSTEM_SHUTDOWN,
            AuditLevel.CRITICAL,
            f"EMERGENCY SHUTDOWN: {reason}",
            severity_score=10.0
        )
    
    def shutdown(self) -> None:
        """Graceful shutdown of security framework"""
        # Shutdown all components
        self.safety.shutdown()
        self.audit.shutdown()
        
        # Clear sessions
        self.active_sessions.clear()
        
        self.audit.log_event(
            EventType.SYSTEM_SHUTDOWN,
            AuditLevel.INFO,
            "Security framework shutdown completed"
        )


# Export main classes and functions
__all__ = [
    "SecurityFramework",
    "SecurityConfiguration",
    "IntegrationLevel"
]