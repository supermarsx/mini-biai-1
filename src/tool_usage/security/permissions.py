#!/usr/bin/env python3
"""
Permission Management and Role-Based Access Control Module

This module provides comprehensive permission management with role-based access
control, session management, and security context tracking.

Author: Mini-Biai Framework Team
Version: 1.0.0
Date: 2024
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core.exceptions import PermissionError, SecurityError


class PermissionLevel(Enum):
    """Permission levels (hierarchical)"""
    NONE = 0      # No permissions
    READ = 1      # Read-only access
    WRITE = 2     # Read and write access
    EXECUTE = 3   # Execute commands
    ADMIN = 4     # Administrative access
    SUPERUSER = 5 # Full system access


class RoleType(Enum):
    """User role types"""
    GUEST = "guest"         # Limited read access
    USER = "user"           # Standard user access
    POWER_USER = "power_user" # Extended privileges
    ADMIN = "admin"         # Administrative access
    SYSTEM = "system"       # System-level access


class OperationType(Enum):
    """Types of operations requiring permission checks"""
    # File system operations
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    FILE_EXECUTE = "file_execute"
    DIRECTORY_CREATE = "directory_create"
    DIRECTORY_DELETE = "directory_delete"
    
    # Network operations
    NETWORK_CONNECT = "network_connect"
    NETWORK_LISTEN = "network_listen"
    NETWORK_SEND = "network_send"
    NETWORK_RECEIVE = "network_receive"
    
    # System operations
    SYSTEM_EXECUTE = "system_execute"
    SYSTEM_ADMIN = "system_admin"
    PROCESS_CREATE = "process_create"
    PROCESS_KILL = "process_kill"
    
    # Data operations
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # Configuration operations
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    CONFIG_ADMIN = "config_admin"
    
    # Security operations
    SECURITY_AUDIT = "security_audit"
    SECURITY_MANAGE = "security_manage"
    USER_MANAGE = "user_manage"
    PERMISSION_ESCALATE = "permission_escalate"


@dataclass
class Permission:
    """Individual permission definition"""
    operation: OperationType
    level: PermissionLevel
    resource_pattern: Optional[str] = None  # Regex pattern for resource matching
    conditions: Optional[Dict[str, Any]] = None  # Additional conditions
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
    
    def matches_operation(self, operation: OperationType, resource: Optional[str] = None) -> bool:
        """Check if permission matches operation and resource"""
        if operation != self.operation:
            return False
        
        if self.resource_pattern and resource:
            import re
            return bool(re.match(self.resource_pattern, resource))
        
        return True


@dataclass
class Role:
    """User role with associated permissions"""
    name: str
    role_type: RoleType
    permissions: List[Permission] = field(default_factory=list)
    description: str = ""
    is_system_role: bool = False  # System roles cannot be modified
    
    def add_permission(self, operation: OperationType, level: PermissionLevel, 
                      resource_pattern: Optional[str] = None,
                      conditions: Optional[Dict[str, Any]] = None) -> None:
        """Add permission to role"""
        permission = Permission(operation, level, resource_pattern, conditions)
        self.permissions.append(permission)
    
    def remove_permission(self, operation: OperationType, 
                         resource_pattern: Optional[str] = None) -> bool:
        """Remove permission from role"""
        original_count = len(self.permissions)
        self.permissions = [
            p for p in self.permissions 
            if not (p.operation == operation and 
                   (resource_pattern is None or p.resource_pattern == resource_pattern))
        ]
        return len(self.permissions) < original_count
    
    def has_permission(self, operation: OperationType, 
                      min_level: PermissionLevel = PermissionLevel.READ,
                      resource: Optional[str] = None) -> bool:
        """Check if role has specific permission"""
        for permission in self.permissions:
            if permission.matches_operation(operation, resource):
                if permission.level.value >= min_level.value:
                    return True
        return False
    
    def get_permissions_for_operation(self, operation: OperationType) -> List[Permission]:
        """Get all permissions for specific operation"""
        return [p for p in self.permissions if p.operation == operation]


@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    password_hash: str
    role: Role
    is_active: bool = True
    is_locked: bool = False
    failed_attempts: int = 0
    last_login: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def verify_password(self, password: str) -> bool:
        """Verify user password"""
        return hashlib.sha256(password.encode()).hexdigest() == self.password_hash
    
    def lock_account(self) -> None:
        """Lock user account"""
        self.is_locked = True
    
    def unlock_account(self) -> None:
        """Unlock user account"""
        self.is_locked = False
        self.failed_attempts = 0
    
    def increment_failed_attempts(self) -> None:
        """Increment failed login attempts"""
        self.failed_attempts += 1
    
    def reset_failed_attempts(self) -> None:
        """Reset failed login attempts"""
        self.failed_attempts = 0
    
    def update_last_login(self) -> None:
        """Update last login timestamp"""
        self.last_login = datetime.now()


@dataclass
class SecurityContext:
    """Current security context for a session"""
    session_id: str
    user_id: str
    role: Role
    permissions: Set[str] = field(default_factory=set)
    escalations: List['PermissionEscalation'] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Initialize permissions from role
        self._update_permissions_from_role()
    
    def _update_permissions_from_role(self) -> None:
        """Update permission set from role permissions"""
        self.permissions.clear()
        for permission in self.role.permissions:
            perm_key = f"{permission.operation.value}:{permission.level.value}"
            if permission.resource_pattern:
                perm_key += f":{permission.resource_pattern}"
            self.permissions.add(perm_key)
    
    def has_permission(self, operation: OperationType, 
                      min_level: PermissionLevel = PermissionLevel.READ,
                      resource: Optional[str] = None) -> bool:
        """Check if context has specific permission"""
        # Check base permissions from role
        if self.role.has_permission(operation, min_level, resource):
            return True
        
        # Check temporary escalations
        for escalation in self.escalations:
            if escalation.is_active() and escalation.allows_operation(operation, min_level):
                return True
        
        return False
    
    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Check if session is expired"""
        return (datetime.now() - self.last_activity).total_seconds() > (timeout_minutes * 60)
    
    def get_session_duration(self) -> timedelta:
        """Get current session duration"""
        return datetime.now() - self.start_time


@dataclass
class PermissionEscalation:
    """Temporary permission escalation request"""
    escalation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: OperationType = OperationType.PERMISSION_ESCALATE
    requested_level: PermissionLevel = PermissionLevel.ADMIN
    resource_pattern: Optional[str] = None
    reason: str = ""
    requested_by: str = ""  # User ID who requested escalation
    approved_by: Optional[str] = None  # User ID who approved
    requested_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_active: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.expires_at is None:
            # Default expiration: 1 hour
            self.expires_at = self.requested_at + timedelta(hours=1)
    
    def is_expired(self) -> bool:
        """Check if escalation is expired"""
        return datetime.now() > self.expires_at
    
    def is_active(self) -> bool:
        """Check if escalation is currently active"""
        return self.is_active and not self.is_expired()
    
    def approve(self, approved_by: str) -> None:
        """Approve the escalation"""
        self.approved_by = approved_by
        self.is_active = True
    
    def reject(self, rejected_by: str) -> None:
        """Reject the escalation"""
        self.approved_by = rejected_by
        self.is_active = False
    
    def allows_operation(self, operation: OperationType, level: PermissionLevel) -> bool:
        """Check if escalation allows specific operation"""
        if not self.is_active():
            return False
        
        return (operation == self.operation and 
                level.value <= self.requested_level.value)


class PermissionChecker:
    """Main permission checking and management system"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.roles: Dict[str, Role] = {}
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.pending_escalations: List[PermissionEscalation] = []
        self.logger = None  # Will be set by parent logger
        
        # Initialize default roles
        self._create_default_roles()
        
        # Session cache for performance
        self._session_cache: Dict[str, SecurityContext] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_cleanup = datetime.now()
    
    def set_logger(self, logger) -> None:
        """Set logger for permission operations"""
        self.logger = logger
    
    def _create_default_roles(self) -> None:
        """Create default system roles"""
        # Guest role - minimal read access
        guest_role = Role("guest", RoleType.GUEST, description="Guest access")
        guest_role.add_permission(OperationType.FILE_READ, PermissionLevel.READ)
        guest_role.add_permission(OperationType.CONFIG_READ, PermissionLevel.READ)
        guest_role.is_system_role = True
        self.roles["guest"] = guest_role
        
        # User role - standard access
        user_role = Role("user", RoleType.USER, description="Standard user access")
        user_role.add_permission(OperationType.FILE_READ, PermissionLevel.READ)
        user_role.add_permission(OperationType.FILE_WRITE, PermissionLevel.WRITE)
        user_role.add_permission(OperationType.DATA_READ, PermissionLevel.READ)
        user_role.add_permission(OperationType.DATA_WRITE, PermissionLevel.WRITE)
        user_role.add_permission(OperationType.CONFIG_READ, PermissionLevel.READ)
        user_role.is_system_role = True
        self.roles["user"] = user_role
        
        # Power user role - extended privileges
        power_user_role = Role("power_user", RoleType.POWER_USER, description="Power user access")
        power_user_role.add_permission(OperationType.FILE_READ, PermissionLevel.READ)
        power_user_role.add_permission(OperationType.FILE_WRITE, PermissionLevel.WRITE)
        power_user_role.add_permission(OperationType.FILE_DELETE, PermissionLevel.WRITE)
        power_user_role.add_permission(OperationType.DATA_READ, PermissionLevel.READ)
        power_user_role.add_permission(OperationType.DATA_WRITE, PermissionLevel.WRITE)
        power_user_role.add_permission(OperationType.DATA_DELETE, PermissionLevel.WRITE)
        power_user_role.add_permission(OperationType.CONFIG_READ, PermissionLevel.READ)
        power_user_role.add_permission(OperationType.CONFIG_WRITE, PermissionLevel.WRITE)
        power_user_role.add_permission(OperationType.SYSTEM_EXECUTE, PermissionLevel.EXECUTE)
        power_user_role.is_system_role = True
        self.roles["power_user"] = power_user_role
        
        # Admin role - administrative access
        admin_role = Role("admin", RoleType.ADMIN, description="Administrator access")
        admin_role.add_permission(OperationType.FILE_READ, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.FILE_WRITE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.FILE_DELETE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.FILE_EXECUTE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.DATA_READ, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.DATA_WRITE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.DATA_DELETE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.CONFIG_READ, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.CONFIG_WRITE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.CONFIG_ADMIN, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.SYSTEM_EXECUTE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.SYSTEM_ADMIN, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.PROCESS_CREATE, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.PROCESS_KILL, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.SECURITY_AUDIT, PermissionLevel.ADMIN)
        admin_role.add_permission(OperationType.USER_MANAGE, PermissionLevel.ADMIN)
        admin_role.is_system_role = True
        self.roles["admin"] = admin_role
        
        # System role - full access
        system_role = Role("system", RoleType.SYSTEM, description="System-level access")
        for operation in OperationType:
            system_role.add_permission(operation, PermissionLevel.SUPERUSER)
        system_role.is_system_role = True
        self.roles["system"] = system_role
    
    def create_user(self, username: str, password: str, role_name: str, 
                   user_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user account"""
        if username in [u.username for u in self.users.values()]:
            raise ValueError(f"Username '{username}' already exists")
        
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        user_id = user_id or str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user = User(
            user_id=user_id,
            username=username,
            password_hash=password_hash,
            role=self.roles[role_name],
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        
        if self.logger:
            self.logger.info(f"User '{username}' created with role '{role_name}'")
        
        return user_id
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session ID"""
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            if self.logger:
                self.logger.warning(f"Login attempt with invalid username: {username}")
            return None
        
        # Check if account is locked
        if user.is_locked:
            if self.logger:
                self.logger.warning(f"Login attempt on locked account: {username}")
            return None
        
        # Check if account is active
        if not user.is_active:
            if self.logger:
                self.logger.warning(f"Login attempt on inactive account: {username}")
            return None
        
        # Verify password
        if not user.verify_password(password):
            user.increment_failed_attempts()
            
            # Lock account after 5 failed attempts
            if user.failed_attempts >= 5:
                user.lock_account()
                if self.logger:
                    self.logger.warning(f"Account locked due to failed attempts: {username}")
            
            return None
        
        # Successful authentication
        user.reset_failed_attempts()
        user.update_last_login()
        
        # Create security context
        session_id = str(uuid.uuid4())
        context = SecurityContext(
            session_id=session_id,
            user_id=user.user_id,
            role=user.role,
            metadata={"username": username}
        )
        
        self.active_sessions[session_id] = context
        
        if self.logger:
            self.logger.info(f"User '{username}' authenticated successfully")
        
        return session_id
    
    def check_permission(self, session_id: str, operation: OperationType, 
                        min_level: PermissionLevel = PermissionLevel.READ,
                        resource: Optional[str] = None) -> bool:
        """Check if session has permission for operation"""
        # Get security context
        context = self._get_security_context(session_id)
        if not context:
            if self.logger:
                self.logger.warning(f"Invalid session ID: {session_id}")
            return False
        
        # Update activity
        context.update_activity()
        
        # Check permission
        has_permission = context.has_permission(operation, min_level, resource)
        
        # Log permission check
        if self.logger:
            status = "GRANTED" if has_permission else "DENIED"
            resource_info = f" on resource '{resource}'" if resource else ""
            self.logger.debug(f"Permission check {status}: {operation.value}{resource_info} (session: {session_id[:8]}...)")
        
        return has_permission
    
    def require_permission(self, session_id: str, operation: OperationType,
                          min_level: PermissionLevel = PermissionLevel.READ,
                          resource: Optional[str] = None) -> None:
        """Require permission for operation, raise error if denied"""
        if not self.check_permission(session_id, operation, min_level, resource):
            raise PermissionError(
                f"Permission denied: {operation.value} "
                f"(required: {min_level.name}, session: {session_id[:8]}...)"
            )
    
    def request_escalation(self, session_id: str, operation: OperationType,
                          requested_level: PermissionLevel, reason: str = "",
                          resource_pattern: Optional[str] = None,
                          duration_minutes: int = 60) -> str:
        """Request temporary permission escalation"""
        # Get current user
        context = self._get_security_context(session_id)
        if not context:
            raise PermissionError("Invalid session for escalation request")
        
        # Create escalation request
        escalation = PermissionEscalation(
            operation=operation,
            requested_level=requested_level,
            reason=reason,
            requested_by=context.user_id,
            resource_pattern=resource_pattern
        )
        
        # Set expiration
        escalation.expires_at = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Add to pending escalations
        self.pending_escalations.append(escalation)
        
        if self.logger:
            self.logger.info(
                f"Escalation requested: {operation.value} to {requested_level.name} "
                f"by session {session_id[:8]}... (ID: {escalation.escalation_id})"
            )
        
        return escalation.escalation_id
    
    def approve_escalation(self, escalation_id: str, admin_session_id: str) -> bool:
        """Approve a pending escalation request"""
        # Find escalation
        escalation = None
        escalation_index = None
        for i, esc in enumerate(self.pending_escalations):
            if esc.escalation_id == escalation_id:
                escalation = esc
                escalation_index = i
                break
        
        if not escalation:
            return False
        
        # Check if admin has permission to approve
        if not self.check_permission(admin_session_id, OperationType.PERMISSION_ESCALATE):
            raise PermissionError("Insufficient permissions to approve escalation")
        
        # Get admin context
        admin_context = self._get_security_context(admin_session_id)
        
        # Approve escalation
        escalation.approve(admin_context.user_id)
        
        # Add to user's context
        user_context = self._get_security_context(
            next(
                (sid for sid, ctx in self.active_sessions.items() 
                 if ctx.user_id == escalation.requested_by),
                None
            )
        )
        
        if user_context:
            user_context.escalations.append(escalation)
        
        # Remove from pending
        if escalation_index is not None:
            self.pending_escalations.pop(escalation_index)
        
        if self.logger:
            self.logger.info(
                f"Escalation approved: {escalation.escalation_id} "
                f"by admin session {admin_session_id[:8]}..."
            )
        
        return True
    
    def reject_escalation(self, escalation_id: str, admin_session_id: str, 
                         reason: str = "") -> bool:
        """Reject a pending escalation request"""
        # Find escalation
        escalation = None
        escalation_index = None
        for i, esc in enumerate(self.pending_escalations):
            if esc.escalation_id == escalation_id:
                escalation = esc
                escalation_index = i
                break
        
        if not escalation:
            return False
        
        # Check if admin has permission
        if not self.check_permission(admin_session_id, OperationType.PERMISSION_ESCALATE):
            raise PermissionError("Insufficient permissions to reject escalation")
        
        # Get admin context
        admin_context = self._get_security_context(admin_session_id)
        
        # Reject escalation
        escalation.reject(admin_context.user_id)
        
        # Remove from pending
        if escalation_index is not None:
            self.pending_escalations.pop(escalation_index)
        
        if self.logger:
            self.logger.info(
                f"Escalation rejected: {escalation.escalation_id} "
                f"by admin session {admin_session_id[:8]}... (Reason: {reason})"
            )
        
        return True
    
    def revoke_session(self, session_id: str, admin_session_id: Optional[str] = None) -> bool:
        """Revoke an active session"""
        # Check if caller has permission
        if admin_session_id and not self.check_permission(admin_session_id, OperationType.USER_MANAGE):
            raise PermissionError("Insufficient permissions to revoke session")
        
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            del self.active_sessions[session_id]
            
            # Also remove from cache
            if session_id in self._session_cache:
                del self._session_cache[session_id]
            
            if self.logger:
                self.logger.info(f"Session revoked: {session_id[:8]}...")
            
            return True
        
        return False
    
    def get_active_sessions(self, admin_session_id: str) -> List[Dict[str, Any]]:
        """Get list of all active sessions (admin only)"""
        self.require_permission(admin_session_id, OperationType.USER_MANAGE)
        
        sessions = []
        for session_id, context in self.active_sessions.items():
            sessions.append({
                "session_id": session_id,
                "user_id": context.user_id,
                "role": context.role.name,
                "start_time": context.start_time.isoformat(),
                "last_activity": context.last_activity.isoformat(),
                "duration_minutes": context.get_session_duration().total_seconds() / 60,
                "metadata": context.metadata
            })
        
        return sessions
    
    def get_pending_escalations(self, admin_session_id: str) -> List[Dict[str, Any]]:
        """Get list of pending escalations (admin only)"""
        self.require_permission(admin_session_id, OperationType.PERMISSION_ESCALATE)
        
        escalations = []
        for escalation in self.pending_escalations:
            escalations.append({
                "escalation_id": escalation.escalation_id,
                "operation": escalation.operation.value,
                "requested_level": escalation.requested_level.name,
                "reason": escalation.reason,
                "requested_by": escalation.requested_by,
                "requested_at": escalation.requested_at.isoformat(),
                "expires_at": escalation.expires_at.isoformat(),
                "resource_pattern": escalation.resource_pattern
            })
        
        return escalations
    
    def create_role(self, name: str, role_type: RoleType, description: str = "") -> Role:
        """Create a new custom role"""
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        role = Role(name=name, role_type=role_type, description=description)
        self.roles[name] = role
        
        if self.logger:
            self.logger.info(f"Custom role created: {name}")
        
        return role
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to existing user"""
        if user_id not in self.users:
            return False
        
        if role_name not in self.roles:
            return False
        
        self.users[user_id].role = self.roles[role_name]
        
        # Update active sessions for this user
        for session in self.active_sessions.values():
            if session.user_id == user_id:
                session.role = self.roles[role_name]
                session._update_permissions_from_role()
        
        if self.logger:
            self.logger.info(f"Role '{role_name}' assigned to user {user_id}")
        
        return True
    
    def _get_security_context(self, session_id: str) -> Optional[SecurityContext]:
        """Get security context with caching"""
        # Cleanup expired cache entries periodically
        now = datetime.now()
        if (now - self._last_cache_cleanup).total_seconds() > self._cache_ttl:
            expired_sessions = [
                sid for sid, ctx in self._session_cache.items()
                if ctx.is_expired() or sid not in self.active_sessions
            ]
            for sid in expired_sessions:
                self._session_cache.pop(sid, None)
            self._last_cache_cleanup = now
        
        # Check cache first
        if session_id in self._session_cache:
            return self._session_cache[session_id]
        
        # Get from active sessions
        if session_id in self.active_sessions:
            context = self.active_sessions[session_id]
            
            # Cache it
            self._session_cache[session_id] = context
            return context
        
        return None
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        expired_sessions = []
        for session_id, context in self.active_sessions.items():
            if context.is_expired():
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.revoke_session(session_id)
        
        return len(expired_sessions)
    
    def export_permissions_config(self) -> Dict[str, Any]:
        """Export current permission configuration"""
        return {
            "roles": {
                name: {
                    "name": role.name,
                    "role_type": role.role_type.value,
                    "description": role.description,
                    "is_system_role": role.is_system_role,
                    "permissions": [
                        {
                            "operation": perm.operation.value,
                            "level": perm.level.value,
                            "resource_pattern": perm.resource_pattern,
                            "conditions": perm.conditions
                        }
                        for perm in role.permissions
                    ]
                }
                for name, role in self.roles.items()
            },
            "users": {
                user_id: {
                    "username": user.username,
                    "role": user.role.name,
                    "is_active": user.is_active,
                    "is_locked": user.is_locked,
                    "failed_attempts": user.failed_attempts,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "created_at": user.created_at.isoformat(),
                    "metadata": user.metadata
                }
                for user_id, user in self.users.items()
            },
            "active_sessions": len(self.active_sessions),
            "pending_escalations": len(self.pending_escalations)
        }
    
    def import_permissions_config(self, config: Dict[str, Any]) -> None:
        """Import permission configuration"""
        # Import roles
        for role_name, role_config in config.get("roles", {}).items():
            if role_name not in self.roles:
                role = Role(
                    name=role_config["name"],
                    role_type=RoleType(role_config["role_type"]),
                    description=role_config.get("description", "")
                )
                
                # Import permissions
                for perm_config in role_config.get("permissions", []):
                    role.add_permission(
                        OperationType(perm_config["operation"]),
                        PermissionLevel(perm_config["level"]),
                        perm_config.get("resource_pattern"),
                        perm_config.get("conditions", {})
                    )
                
                self.roles[role_name] = role
        
        # Note: User passwords are not imported for security reasons
        # This would need to be handled separately


# Export main classes and functions
__all__ = [
    "PermissionChecker",
    "User",
    "Role", 
    "Permission",
    "SecurityContext",
    "PermissionEscalation",
    "PermissionLevel",
    "RoleType",
    "OperationType"
]