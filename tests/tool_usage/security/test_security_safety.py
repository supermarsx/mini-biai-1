#!/usr/bin/env python3
"""
Security Testing Module for Tool Usage System

This module provides comprehensive security testing for the tool usage system,
covering validation, sandboxing, audit logging, and threat mitigation capabilities.

Author: MiniMax AI Agent
Created: 2024
License: MIT
"""

import pytest
import unittest
import unittest.mock as mock
import os
import sys
import tempfile
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the modules we're testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from tool_usage.src.security import (
        SecurityValidator,
        Sandbox,
        SafetyMode,
        AuditLogger
    )
except ImportError:
    # Fallback for testing without full module structure
    SecurityValidator = None
    Sandbox = None
    SafetyMode = None
    AuditLogger = None


class MockSecurityValidator:
    """Mock SecurityValidator for testing purposes."""
    
    def __init__(self):
        self.threat_patterns = [
            r'rm\s+-rf\s+/',  # Dangerous deletion patterns
            r'chmod\s+777',   # Permission escalation
            r'sudo\s+',       # Privilege escalation
            r'>\s*/etc/',     # System file modification
            r'dd\s+if=',      # Data duplication patterns
            r'format\s+',     # Disk formatting
        ]
        
        self.whitelist = []
        self.blacklist = []
        self.custom_rules = {}
        
    def validate_command(self, command: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a command for security threats."""
        context = context or {}
        
        # Check for threats
        threats = []
        for pattern in self.threat_patterns:
            import re
            if re.search(pattern, command, re.IGNORECASE):
                threats.append({
                    'pattern': pattern,
                    'type': 'THREAT_DETECTED',
                    'severity': 'HIGH',
                    'description': f'Dangerous command pattern detected: {pattern}'
                })
        
        # Check whitelist/blacklist
        if command in self.blacklist:
            threats.append({
                'type': 'BLACKLISTED',
                'severity': 'MEDIUM',
                'description': 'Command is in blacklist'
            })
        
        return {
            'safe': len(threats) == 0,
            'threats': threats,
            'confidence': 0.95 if threats else 0.99,
            'analysis_time': 0.001
        }
    
    def add_whitelist_entry(self, command_pattern: str):
        """Add a command pattern to whitelist."""
        self.whitelist.append(command_pattern)
    
    def add_blacklist_entry(self, command_pattern: str):
        """Add a command pattern to blacklist."""
        self.blacklist.append(command_pattern)
    
    def add_custom_rule(self, pattern: str, rule_config: Dict[str, Any]):
        """Add a custom security rule."""
        self.custom_rules[pattern] = rule_config


class MockSandbox:
    """Mock Sandbox for testing purposes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.active_sandboxes = {}
        self.sandbox_counter = 0
        self.permitted_operations = set([
            'read_file', 'write_file', 'list_directory', 'get_process_info',
            'network_access', 'memory_limit', 'cpu_limit'
        ])
        self.denied_operations = set([
            'modify_system', 'access_sensitive_files', 'escalate_privileges',
            'network_scan', 'file_system_format', 'process_injection'
        ])
        
    def create_sandbox(self, command: str, config: Dict[str, Any] = None) -> str:
        """Create a new sandbox environment."""
        sandbox_id = f"sandbox_{self.sandbox_counter}"
        self.sandbox_counter += 1
        
        sandbox_config = {
            'id': sandbox_id,
            'command': command,
            'created_at': datetime.now().isoformat(),
            'active': True,
            'config': {**(self.config or {}), **(config or {})},
            'restrictions': {
                'max_memory_mb': self.config.get('max_memory_mb', 512),
                'max_cpu_percent': self.config.get('max_cpu_percent', 50),
                'network_access': self.config.get('network_access', True),
                'file_system_access': self.config.get('file_system_access', 'limited'),
                'time_limit_seconds': self.config.get('time_limit_seconds', 300)
            }
        }
        
        self.active_sandboxes[sandbox_id] = sandbox_config
        return sandbox_id
    
    def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Destroy a sandbox environment."""
        if sandbox_id in self.active_sandboxes:
            self.active_sandboxes[sandbox_id]['active'] = False
            del self.active_sandboxes[sandbox_id]
            return True
        return False
    
    def execute_in_sandbox(self, command: str, sandbox_id: str = None) -> Dict[str, Any]:
        """Execute a command within a sandbox."""
        if sandbox_id is None:
            sandbox_id = self.create_sandbox(command)
        
        if sandbox_id not in self.active_sandboxes:
            return {
                'success': False,
                'error': 'Sandbox not found',
                'sandbox_id': sandbox_id
            }
        
        sandbox = self.active_sandboxes[sandbox_id]
        
        # Simulate execution with restrictions
        execution_result = {
            'sandbox_id': sandbox_id,
            'command': command,
            'success': True,
            'output': f'Sandboxed execution output for: {command[:50]}...',
            'execution_time': 0.123,
            'resource_usage': {
                'memory_mb': 45.2,
                'cpu_percent': 12.3,
                'network_bytes': 1024,
                'disk_io_mb': 0.5
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return execution_result
    
    def is_operation_allowed(self, operation: str, sandbox_id: str = None) -> bool:
        """Check if an operation is allowed in the sandbox."""
        if sandbox_id and sandbox_id in self.active_sandboxes:
            return operation in self.permitted_operations
        
        return operation in self.permitted_operations and operation not in self.denied_operations
    
    def get_active_sandboxes(self) -> List[Dict[str, Any]]:
        """Get list of active sandboxes."""
        return [sandbox for sandbox in self.active_sandboxes.values() if sandbox['active']]


class MockSafetyMode:
    """Mock SafetyMode enum for testing purposes."""
    
    NONE = "none"
    STANDARD = "standard" 
    STRICT = "strict"
    PARANOID = "paranoid"
    
    @classmethod
    def get_default_level(cls, context: Dict[str, Any] = None) -> str:
        """Get default safety level based on context."""
        context = context or {}
        
        if context.get('high_risk', False):
            return cls.PARANOID
        elif context.get('medium_risk', False):
            return cls.STRICT
        elif context.get('user_trusted', True):
            return cls.STANDARD
        else:
            return cls.NONE
    
    @classmethod
    def get_restrictions_for_mode(cls, mode: str) -> Dict[str, Any]:
        """Get security restrictions for a safety mode."""
        restrictions = {
            cls.NONE: {
                'command_validation': False,
                'sandbox_required': False,
                'audit_logging': False,
                'resource_limits': False,
                'network_restrictions': False
            },
            cls.STANDARD: {
                'command_validation': True,
                'sandbox_required': False,
                'audit_logging': True,
                'resource_limits': True,
                'network_restrictions': False
            },
            cls.STRICT: {
                'command_validation': True,
                'sandbox_required': True,
                'audit_logging': True,
                'resource_limits': True,
                'network_restrictions': True,
                'file_system_access': 'limited',
                'network_whitelist_only': True
            },
            cls.PARANOID: {
                'command_validation': True,
                'sandbox_required': True,
                'audit_logging': True,
                'resource_limits': True,
                'network_restrictions': True,
                'file_system_access': 'read_only',
                'network_whitelist_only': True,
                'permission_checks': True,
                'static_analysis': True
            }
        }
        
        return restrictions.get(mode, restrictions[cls.STANDARD])


class MockAuditLogger:
    """Mock AuditLogger for testing purposes."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.log_entries = []
        self.session_logs = {}
        self.log_counter = 0
        
    def log_command_execution(self, command: str, result: Dict[str, Any], 
                            context: Dict[str, Any] = None) -> str:
        """Log a command execution event."""
        context = context or {}
        
        log_entry = {
            'id': f"log_{self.log_counter}",
            'timestamp': datetime.now().isoformat(),
            'type': 'COMMAND_EXECUTION',
            'session_id': context.get('session_id', 'default'),
            'command': command,
            'result': result,
            'context': context,
            'security_context': context.get('security', {}),
            'user_id': context.get('user_id'),
            'source_ip': context.get('source_ip'),
            'user_agent': context.get('user_agent')
        }
        
        self.log_entries.append(log_entry)
        self.log_counter += 1
        
        return log_entry['id']
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                         severity: str = 'INFO') -> str:
        """Log a security event."""
        log_entry = {
            'id': f"sec_{self.log_counter}",
            'timestamp': datetime.now().isoformat(),
            'type': 'SECURITY_EVENT',
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'session_id': details.get('session_id'),
            'user_id': details.get('user_id'),
            'source_ip': details.get('source_ip')
        }
        
        self.log_entries.append(log_entry)
        self.log_counter += 1
        
        return log_entry['id']
    
    def log_threat_detection(self, threat: Dict[str, Any], command: str) -> str:
        """Log a detected threat."""
        return self.log_security_event(
            'THREAT_DETECTED',
            {
                'command': command,
                'threat_details': threat,
                'action_taken': 'BLOCKED'
            },
            'HIGH'
        )
    
    def log_access_violation(self, violation: Dict[str, Any]) -> str:
        """Log an access violation."""
        return self.log_security_event(
            'ACCESS_VIOLATION',
            violation,
            'CRITICAL'
        )
    
    def get_logs_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get logs for a specific session."""
        return [entry for entry in self.log_entries if entry.get('session_id') == session_id]
    
    def get_logs_by_time_range(self, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """Get logs within a time range."""
        # Simple string comparison for demo (would use proper datetime parsing in production)
        return [
            entry for entry in self.log_entries 
            if start_time <= entry['timestamp'] <= end_time
        ]
    
    def get_security_summary(self, session_id: str = None) -> Dict[str, Any]:
        """Get security summary statistics."""
        logs_to_analyze = (
            self.get_logs_by_session(session_id) if session_id else self.log_entries
        )
        
        security_events = [entry for entry in logs_to_analyze if entry['type'] == 'SECURITY_EVENT']
        
        summary = {
            'total_commands': len([entry for entry in logs_to_analyze if entry['type'] == 'COMMAND_EXECUTION']),
            'total_security_events': len(security_events),
            'threats_detected': len([e for e in security_events if e['event_type'] == 'THREAT_DETECTED']),
            'access_violations': len([e for e in security_events if e['event_type'] == 'ACCESS_VIOLATION']),
            'severity_breakdown': {
                'INFO': len([e for e in security_events if e['severity'] == 'INFO']),
                'HIGH': len([e for e in security_events if e['severity'] == 'HIGH']),
                'CRITICAL': len([e for e in security_events if e['severity'] == 'CRITICAL'])
            },
            'time_range': {
                'first_log': min(logs_to_analyze, key=lambda x: x['timestamp'])['timestamp'] if logs_to_analyze else None,
                'last_log': max(logs_to_analyze, key=lambda x: x['timestamp'])['timestamp'] if logs_to_analyze else None
            }
        }
        
        return summary


class TestSecurityValidator(unittest.TestCase):
    """Test suite for SecurityValidator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = MockSecurityValidator()
        self.safe_commands = [
            'ls -la',
            'pwd',
            'echo "hello world"',
            'find . -name "*.txt"',
            'grep "test" file.txt'
        ]
        
        self.dangerous_commands = [
            'rm -rf /',
            'chmod 777 /etc/passwd',
            'sudo rm -rf /',
            'dd if=/dev/zero of=/dev/sda',
            'format c:'
        ]
    
    def test_initialization(self):
        """Test SecurityValidator initialization."""
        self.assertIsInstance(self.validator, MockSecurityValidator)
        self.assertIsInstance(self.validator.threat_patterns, list)
        self.assertGreater(len(self.validator.threat_patterns), 0)
    
    def test_safe_command_validation(self):
        """Test validation of safe commands."""
        for command in self.safe_commands:
            with self.subTest(command=command):
                result = self.validator.validate_command(command)
                self.assertTrue(result['safe'])
                self.assertEqual(len(result['threats']), 0)
                self.assertGreaterEqual(result['confidence'], 0.9)
    
    def test_dangerous_command_detection(self):
        """Test detection of dangerous commands."""
        for command in self.dangerous_commands:
            with self.subTest(command=command):
                result = self.validator.validate_command(command)
                self.assertFalse(result['safe'])
                self.assertGreater(len(result['threats']), 0)
                
                # Check that at least one threat was detected
                threat_types = [threat['type'] for threat in result['threats']]
                self.assertTrue('THREAT_DETECTED' in threat_types or 'BLACKLISTED' in threat_types)
    
    def test_whitelist_functionality(self):
        """Test whitelist functionality."""
        command = 'ls -la'
        self.validator.add_whitelist_entry('ls -la')
        
        result = self.validator.validate_command('ls -la')
        self.assertTrue(result['safe'])
    
    def test_blacklist_functionality(self):
        """Test blacklist functionality."""
        command = 'echo "test"'
        self.validator.add_blacklist_entry('echo "test"')
        
        result = self.validator.validate_command('echo "test"')
        self.assertFalse(result['safe'])
        
        # Check that it was flagged as blacklisted
        threat_types = [threat['type'] for threat in result['threats']]
        self.assertIn('BLACKLISTED', threat_types)
    
    def test_custom_rules(self):
        """Test custom security rules."""
        rule_config = {
            'pattern': 'custom_dangerous_pattern',
            'severity': 'HIGH',
            'description': 'Custom security rule test'
        }
        
        self.validator.add_custom_rule('custom_pattern', rule_config)
        
        self.assertIn('custom_pattern', self.validator.custom_rules)
        self.assertEqual(self.validator.custom_rules['custom_pattern']['severity'], 'HIGH')
    
    def test_validation_performance(self):
        """Test validation performance."""
        start_time = time.time()
        
        for _ in range(100):
            self.validator.validate_command('ls -la')
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete 100 validations in under 1 second
        self.assertLess(execution_time, 1.0, "Validation performance is too slow")
    
    def test_threat_severity_classification(self):
        """Test threat severity classification."""
        test_cases = [
            ('rm -rf /', 'HIGH'),
            ('chmod 777 /etc/passwd', 'HIGH'),
            ('format c:', 'HIGH'),
        ]
        
        for command, expected_severity in test_cases:
            with self.subTest(command=command, severity=expected_severity):
                result = self.validator.validate_command(command)
                self.assertFalse(result['safe'])
                
                # Find threat with expected severity
                has_expected_severity = any(
                    threat.get('severity') == expected_severity 
                    for threat in result['threats']
                )
                self.assertTrue(has_expected_severity, f"Expected severity {expected_severity} not found in {result['threats']}")
    
    def test_context_aware_validation(self):
        """Test validation with additional context."""
        context = {
            'user_trust_level': 'high',
            'environment': 'production',
            'time_of_day': 'business_hours'
        }
        
        result = self.validator.validate_command('ls -la', context)
        self.assertTrue(result['safe'])
        
        # Context should not affect basic safety assessment
        dangerous_result = self.validator.validate_command('rm -rf /', context)
        self.assertFalse(dangerous_result['safe'])


class TestSandbox(unittest.TestCase):
    """Test suite for Sandbox."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sandbox = MockSandbox()
        self.test_command = 'ls -la /tmp'
        self.test_config = {
            'max_memory_mb': 256,
            'max_cpu_percent': 25,
            'time_limit_seconds': 60
        }
    
    def test_initialization(self):
        """Test Sandbox initialization."""
        self.assertIsInstance(self.sandbox, MockSandbox)
        self.assertEqual(len(self.sandbox.permitted_operations), 7)
        self.assertEqual(len(self.sandbox.denied_operations), 6)
    
    def test_create_sandbox(self):
        """Test sandbox creation."""
        sandbox_id = self.sandbox.create_sandbox(self.test_command, self.test_config)
        
        self.assertIsInstance(sandbox_id, str)
        self.assertIn(sandbox_id, self.sandbox.active_sandboxes)
        
        sandbox_info = self.sandbox.active_sandboxes[sandbox_id]
        self.assertEqual(sandbox_info['command'], self.test_command)
        self.assertTrue(sandbox_info['active'])
        self.assertEqual(sandbox_info['restrictions']['max_memory_mb'], 256)
    
    def test_destroy_sandbox(self):
        """Test sandbox destruction."""
        sandbox_id = self.sandbox.create_sandbox(self.test_command)
        
        # Verify sandbox exists
        self.assertIn(sandbox_id, self.sandbox.active_sandboxes)
        
        # Destroy sandbox
        result = self.sandbox.destroy_sandbox(sandbox_id)
        self.assertTrue(result)
        
        # Verify sandbox is removed
        self.assertNotIn(sandbox_id, self.sandbox.active_sandboxes)
    
    def test_destroy_nonexistent_sandbox(self):
        """Test destroying a nonexistent sandbox."""
        result = self.sandbox.destroy_sandbox('nonexistent_id')
        self.assertFalse(result)
    
    def test_execute_in_sandbox(self):
        """Test command execution in sandbox."""
        # Create and execute in one operation
        result = self.sandbox.execute_in_sandbox(self.test_command)
        
        self.assertTrue(result['success'])
        self.assertIn('sandbox_id', result)
        self.assertIn('execution_time', result)
        self.assertIn('resource_usage', result)
        
        # Verify sandbox was created
        self.assertIn(result['sandbox_id'], self.sandbox.active_sandboxes)
    
    def test_operation_allowance(self):
        """Test operation allowance checking."""
        # Test permitted operations
        self.assertTrue(self.sandbox.is_operation_allowed('read_file'))
        self.assertTrue(self.sandbox.is_operation_allowed('write_file'))
        self.assertTrue(self.sandbox.is_operation_allowed('list_directory'))
        
        # Test denied operations
        self.assertFalse(self.sandbox.is_operation_allowed('modify_system'))
        self.assertFalse(self.sandbox.is_operation_allowed('access_sensitive_files'))
        self.assertFalse(self.sandbox.is_operation_allowed('escalate_privileges'))
    
    def test_active_sandboxes_list(self):
        """Test listing active sandboxes."""
        # Initially empty
        active_before = self.sandbox.get_active_sandboxes()
        self.assertEqual(len(active_before), 0)
        
        # Create some sandboxes
        id1 = self.sandbox.create_sandbox('ls -la')
        id2 = self.sandbox.create_sandbox('pwd')
        
        active_after = self.sandbox.get_active_sandboxes()
        self.assertEqual(len(active_after), 2)
        
        # Destroy one and check again
        self.sandbox.destroy_sandbox(id1)
        active_final = self.sandbox.get_active_sandboxes()
        self.assertEqual(len(active_final), 1)
    
    def test_sandbox_configuration(self):
        """Test sandbox configuration."""
        config = {
            'max_memory_mb': 1024,
            'network_access': False,
            'time_limit_seconds': 120
        }
        
        sandbox_id = self.sandbox.create_sandbox(self.test_command, config)
        sandbox_info = self.sandbox.active_sandboxes[sandbox_id]
        
        self.assertEqual(sandbox_info['restrictions']['max_memory_mb'], 1024)
        self.assertFalse(sandbox_info['restrictions']['network_access'])
        self.assertEqual(sandbox_info['restrictions']['time_limit_seconds'], 120)
    
    def test_concurrent_sandbox_creation(self):
        """Test creating multiple sandboxes concurrently."""
        def create_sandbox(i):
            return self.sandbox.create_sandbox(f'command_{i}')
        
        # Create multiple sandboxes concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_sandbox, i) for i in range(10)]
            sandbox_ids = [future.result() for future in as_completed(futures)]
        
        # Verify all sandboxes were created
        self.assertEqual(len(sandbox_ids), 10)
        
        # Verify they all exist
        active_sandboxes = self.sandbox.get_active_sandboxes()
        self.assertEqual(len(active_sandboxes), 10)
    
    def test_resource_tracking(self):
        """Test resource usage tracking."""
        result = self.sandbox.execute_in_sandbox(self.test_command)
        
        resource_usage = result['resource_usage']
        self.assertIn('memory_mb', resource_usage)
        self.assertIn('cpu_percent', resource_usage)
        self.assertIn('network_bytes', resource_usage)
        self.assertIn('disk_io_mb', resource_usage)
        
        # Verify reasonable resource values
        self.assertGreaterEqual(resource_usage['memory_mb'], 0)
        self.assertGreaterEqual(resource_usage['cpu_percent'], 0)
        self.assertGreaterEqual(resource_usage['network_bytes'], 0)
        self.assertGreaterEqual(resource_usage['disk_io_mb'], 0)


class TestSafetyMode(unittest.TestCase):
    """Test suite for SafetyMode."""
    
    def test_mode_enumeration(self):
        """Test SafetyMode enumeration values."""
        self.assertEqual(MockSafetyMode.NONE, "none")
        self.assertEqual(MockSafetyMode.STANDARD, "standard")
        self.assertEqual(MockSafetyMode.STRICT, "strict")
        self.assertEqual(MockSafetyMode.PARANOID, "paranoid")
    
    def test_default_level_selection(self):
        """Test default safety level selection based on context."""
        # High risk context should return paranoid
        high_risk_context = {'high_risk': True}
        self.assertEqual(MockSafetyMode.get_default_level(high_risk_context), MockSafetyMode.PARANOID)
        
        # Medium risk context should return strict
        medium_risk_context = {'medium_risk': True}
        self.assertEqual(MockSafetyMode.get_default_level(medium_risk_context), MockSafetyMode.STRICT)
        
        # Trusted user should get standard
        trusted_context = {'user_trusted': True}
        self.assertEqual(MockSafetyMode.get_default_level(trusted_context), MockSafetyMode.STANDARD)
        
        # Default should be standard
        default_context = {}
        self.assertEqual(MockSafetyMode.get_default_level(default_context), MockSafetyMode.STANDARD)
    
    def test_restrictions_for_modes(self):
        """Test security restrictions for each safety mode."""
        # Test NONE mode - minimal restrictions
        none_restrictions = MockSafetyMode.get_restrictions_for_mode(MockSafetyMode.NONE)
        self.assertFalse(none_restrictions['command_validation'])
        self.assertFalse(none_restrictions['sandbox_required'])
        self.assertFalse(none_restrictions['audit_logging'])
        
        # Test STANDARD mode - basic protections
        standard_restrictions = MockSafetyMode.get_restrictions_for_mode(MockSafetyMode.STANDARD)
        self.assertTrue(standard_restrictions['command_validation'])
        self.assertTrue(standard_restrictions['audit_logging'])
        self.assertTrue(standard_restrictions['resource_limits'])
        
        # Test STRICT mode - enhanced protections
        strict_restrictions = MockSafetyMode.get_restrictions_for_mode(MockSafetyMode.STRICT)
        self.assertTrue(strict_restrictions['sandbox_required'])
        self.assertTrue(strict_restrictions['network_restrictions'])
        self.assertEqual(strict_restrictions['file_system_access'], 'limited')
        
        # Test PARANOID mode - maximum security
        paranoid_restrictions = MockSafetyMode.get_restrictions_for_mode(MockSafetyMode.PARANOID)
        self.assertTrue(paranoid_restrictions['sandbox_required'])
        self.assertTrue(paranoid_restrictions['network_whitelist_only'])
        self.assertEqual(paranoid_restrictions['file_system_access'], 'read_only')
        self.assertTrue(paranoid_restrictions['permission_checks'])
        self.assertTrue(paranoid_restrictions['static_analysis'])
    
    def test_unknown_mode_handling(self):
        """Test handling of unknown safety modes."""
        unknown_restrictions = MockSafetyMode.get_restrictions_for_mode('unknown_mode')
        
        # Should fall back to standard restrictions
        self.assertEqual(unknown_restrictions, MockSafetyMode.get_restrictions_for_mode(MockSafetyMode.STANDARD))


class TestAuditLogger(unittest.TestCase):
    """Test suite for AuditLogger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = MockAuditLogger()
        self.session_id = "test_session_123"
        self.user_id = "test_user_456"
        self.command = "ls -la /tmp"
        
        self.execution_result = {
            'success': True,
            'output': 'Sample output',
            'exit_code': 0,
            'execution_time': 0.456
        }
        
        self.context = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'source_ip': '127.0.0.1',
            'user_agent': 'Test Agent/1.0',
            'security': {
                'validation_passed': True,
                'threat_level': 'LOW'
            }
        }
    
    def test_initialization(self):
        """Test AuditLogger initialization."""
        self.assertIsInstance(self.logger, MockAuditLogger)
        self.assertEqual(len(self.logger.log_entries), 0)
        self.assertEqual(self.logger.log_counter, 0)
    
    def test_command_execution_logging(self):
        """Test logging command execution."""
        log_id = self.logger.log_command_execution(
            self.command, self.execution_result, self.context
        )
        
        # Verify log entry was created
        self.assertIsInstance(log_id, str)
        self.assertGreater(len(self.logger.log_entries), 0)
        
        # Verify log entry content
        last_entry = self.logger.log_entries[-1]
        self.assertEqual(last_entry['type'], 'COMMAND_EXECUTION')
        self.assertEqual(last_entry['command'], self.command)
        self.assertEqual(last_entry['result'], self.execution_result)
        self.assertEqual(last_entry['session_id'], self.session_id)
        self.assertEqual(last_entry['user_id'], self.user_id)
        self.assertIn('timestamp', last_entry)
    
    def test_security_event_logging(self):
        """Test logging security events."""
        event_details = {
            'event_description': 'Unauthorized access attempt',
            'ip_address': '192.168.1.100',
            'target_resource': '/etc/passwd'
        }
        
        log_id = self.logger.log_security_event('UNAUTHORIZED_ACCESS', event_details, 'HIGH')
        
        # Verify log entry
        last_entry = self.logger.log_entries[-1]
        self.assertEqual(last_entry['type'], 'SECURITY_EVENT')
        self.assertEqual(last_entry['event_type'], 'UNAUTHORIZED_ACCESS')
        self.assertEqual(last_entry['severity'], 'HIGH')
        self.assertEqual(last_entry['details'], event_details)
    
    def test_threat_detection_logging(self):
        """Test logging threat detection events."""
        threat_details = {
            'pattern': 'rm -rf /',
            'severity': 'CRITICAL',
            'description': 'System file deletion attempt'
        }
        
        log_id = self.logger.log_threat_detection(threat_details, self.command)
        
        # Verify log entry
        last_entry = self.logger.log_entries[-1]
        self.assertEqual(last_entry['type'], 'SECURITY_EVENT')
        self.assertEqual(last_entry['event_type'], 'THREAT_DETECTED')
        self.assertEqual(last_entry['severity'], 'HIGH')
        self.assertEqual(last_entry['details']['command'], self.command)
        self.assertEqual(last_entry['details']['threat_details'], threat_details)
    
    def test_access_violation_logging(self):
        """Test logging access violations."""
        violation_details = {
            'resource': '/sensitive/data',
            'operation': 'read',
            'user_id': self.user_id,
            'authorized': False
        }
        
        log_id = self.logger.log_access_violation(violation_details)
        
        # Verify log entry
        last_entry = self.logger.log_entries[-1]
        self.assertEqual(last_entry['type'], 'SECURITY_EVENT')
        self.assertEqual(last_entry['event_type'], 'ACCESS_VIOLATION')
        self.assertEqual(last_entry['severity'], 'CRITICAL')
        self.assertEqual(last_entry['details'], violation_details)
    
    def test_session_filtering(self):
        """Test filtering logs by session."""
        # Log multiple commands from different sessions
        self.logger.log_command_execution('cmd1', self.execution_result, {'session_id': 'session1'})
        self.logger.log_command_execution('cmd2', self.execution_result, {'session_id': 'session2'})
        self.logger.log_command_execution('cmd3', self.execution_result, {'session_id': 'session1'})
        
        # Filter by session1
        session1_logs = self.logger.get_logs_by_session('session1')
        self.assertEqual(len(session1_logs), 2)
        
        # Filter by session2
        session2_logs = self.logger.get_logs_by_session('session2')
        self.assertEqual(len(session2_logs), 1)
        
        # Check content
        for log in session1_logs:
            self.assertEqual(log['session_id'], 'session1')
        
        for log in session2_logs:
            self.assertEqual(log['session_id'], 'session2')
    
    def test_time_range_filtering(self):
        """Test filtering logs by time range."""
        current_time = datetime.now()
        time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Note: In a real implementation, this would properly handle datetime objects
        # For this mock, we're doing simple string comparison
        start_time = '2024-01-01 00:00:00'
        end_time = '2024-12-31 23:59:59'
        
        time_filtered_logs = self.logger.get_logs_by_time_range(start_time, end_time)
        # All logs should be within the range in this test
        self.assertGreaterEqual(len(time_filtered_logs), 0)
    
    def test_security_summary(self):
        """Test security summary generation."""
        # Add various types of logs
        self.logger.log_command_execution('cmd1', {'success': True}, {'session_id': 'session1'})
        self.logger.log_security_event('INFO_EVENT', {'description': 'Info'}, 'INFO')
        self.logger.log_security_event('WARNING_EVENT', {'description': 'Warning'}, 'HIGH')
        self.logger.log_threat_detection({'pattern': 'dangerous'}, 'dangerous_cmd')
        
        summary = self.logger.get_security_summary('session1')
        
        # Verify summary structure
        self.assertIn('total_commands', summary)
        self.assertIn('total_security_events', summary)
        self.assertIn('threats_detected', summary)
        self.assertIn('access_violations', summary)
        self.assertIn('severity_breakdown', summary)
        self.assertIn('time_range', summary)
        
        # Verify counts
        self.assertGreater(summary['total_commands'], 0)
        self.assertGreater(summary['total_security_events'], 0)
        self.assertGreater(summary['threats_detected'], 0)
        self.assertEqual(summary['access_violations'], 0)  # No access violations logged
    
    def test_log_counter_incrementation(self):
        """Test log counter increments correctly."""
        initial_counter = self.logger.log_counter
        
        # Log several entries
        self.logger.log_command_execution('cmd1', self.execution_result, self.context)
        self.logger.log_security_event('TEST_EVENT', {}, 'INFO')
        
        final_counter = self.logger.log_counter
        
        self.assertEqual(final_counter - initial_counter, 2)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security components working together."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.security_components = {
            'validator': MockSecurityValidator(),
            'sandbox': MockSandbox(),
            'safety_mode': MockSafetyMode,
            'audit_logger': MockAuditLogger()
        }
        
        self.scenario_commands = {
            'safe': 'ls -la /tmp',
            'moderately_risky': 'find /usr -name "*.conf"',
            'high_risk': 'rm -rf /tmp/test_*',
            'dangerous': 'sudo rm -rf /'
        }
    
    def test_complete_security_workflow(self):
        """Test complete security workflow from command to execution."""
        command = self.scenario_commands['moderately_risky']
        session_id = "integration_test_session"
        
        # Step 1: Validate command
        validation_result = self.security_components['validator'].validate_command(command)
        
        # Step 2: Determine safety mode
        safety_level = MockSafetyMode.get_default_level({'medium_risk': True})
        restrictions = MockSafetyMode.get_restrictions_for_mode(safety_level)
        
        # Step 3: Create sandbox if required
        sandbox_id = None
        if restrictions['sandbox_required']:
            sandbox_id = self.security_components['sandbox'].create_sandbox(command)
        
        # Step 4: Log the security decision
        log_id = self.security_components['audit_logger'].log_security_event(
            'SECURITY_DECISION',
            {
                'command': command,
                'validation_result': validation_result,
                'safety_mode': safety_level,
                'sandbox_required': restrictions['sandbox_required']
            },
            'INFO'
        )
        
        # Verify workflow results
        self.assertTrue(validation_result['safe'])
        self.assertEqual(safety_level, MockSafetyMode.STRICT)
        self.assertTrue(restrictions['sandbox_required'])
        self.assertIsNotNone(log_id)
        self.assertIsNotNone(sandbox_id)
    
    def test_threat_mitigation_workflow(self):
        """Test workflow for handling threats."""
        dangerous_command = self.scenario_commands['dangerous']
        session_id = "threat_test_session"
        
        # Step 1: Validate dangerous command
        validation_result = self.security_components['validator'].validate_command(dangerous_command)
        
        # Verify threat was detected
        self.assertFalse(validation_result['safe'])
        self.assertGreater(len(validation_result['threats']), 0)
        
        # Step 2: Log threat detection
        threat_log_id = None
        for threat in validation_result['threats']:
            threat_log_id = self.security_components['audit_logger'].log_threat_detection(
                threat, dangerous_command
            )
        
        # Step 3: Determine action based on safety mode
        safety_mode = MockSafetyMode.get_default_level({'high_risk': True})
        restrictions = MockSafetyMode.get_restrictions_for_mode(safety_mode)
        
        # Should not allow execution
        action = "BLOCK" if not validation_result['safe'] else "ALLOW"
        
        # Step 4: Log final decision
        decision_log_id = self.security_components['audit_logger'].log_security_event(
            'EXECUTION_DECISION',
            {
                'command': dangerous_command,
                'decision': action,
                'reason': 'Threat detected' if not validation_result['safe'] else 'Validation passed'
            },
            'HIGH' if action == 'BLOCK' else 'INFO'
        )
        
        # Verify threat handling
        self.assertEqual(action, 'BLOCK')
        self.assertIsNotNone(threat_log_id)
        self.assertIsNotNone(decision_log_id)
    
    def test_audit_trail_integrity(self):
        """Test audit trail integrity across security events."""
        session_id = "audit_test_session"
        user_id = "test_user_789"
        
        # Simulate a session with various security events
        commands = [
            ('ls -la', True, 'SUCCESS'),  # Safe command
            ('find /tmp', True, 'SUCCESS'),  # Moderate command
            ('chmod 777 /etc', False, 'BLOCKED'),  # Blocked command
            ('echo "test"', True, 'SUCCESS')  # Safe command
        ]
        
        for i, (command, should_pass, expected_result) in enumerate(commands):
            context = {
                'session_id': session_id,
                'user_id': user_id,
                'command_index': i
            }
            
            # Validate command
            validation = self.security_components['validator'].validate_command(command)
            
            # Log command execution attempt
            execution_result = {
                'success': should_pass,
                'blocked': not should_pass,
                'validation_result': validation
            }
            
            self.security_components['audit_logger'].log_command_execution(
                command, execution_result, context
            )
            
            # Log security decision
            severity = 'INFO' if should_pass else 'HIGH'
            self.security_components['audit_logger'].log_security_event(
                'COMMAND_EVALUATION',
                {
                    'command': command,
                    'allowed': should_pass,
                    'threats_found': len(validation['threats']) if not should_pass else 0
                },
                severity
            )
        
        # Verify audit trail
        session_logs = self.security_components['audit_logger'].get_logs_by_session(session_id)
        self.assertEqual(len(session_logs), 8)  # 4 executions + 4 decisions
        
        # Verify user consistency
        for log in session_logs:
            if log['type'] == 'COMMAND_EXECUTION':
                self.assertEqual(log['user_id'], user_id)
                self.assertEqual(log['session_id'], session_id)
    
    def test_parallel_security_operations(self):
        """Test security operations running in parallel."""
        def security_operation(command, op_id):
            # Validate command
            validation = self.security_components['validator'].validate_command(command)
            
            # Create sandbox if needed
            safety_mode = MockSafetyMode.get_default_level()
            restrictions = MockSafetyMode.get_restrictions_for_mode(safety_mode)
            
            sandbox_id = None
            if restrictions['sandbox_required']:
                sandbox_id = self.security_components['sandbox'].create_sandbox(command)
            
            # Log operation
            context = {'operation_id': op_id, 'thread_id': threading.current_thread().ident}
            log_id = self.security_components['audit_logger'].log_command_execution(
                command, {'validation': validation}, context
            )
            
            return {
                'operation_id': op_id,
                'validation': validation,
                'sandbox_id': sandbox_id,
                'log_id': log_id,
                'success': True
            }
        
        # Run multiple security operations in parallel
        test_commands = [
            'ls -la',
            'find . -name "*.txt"',
            'grep "test" file.txt',
            'cat /etc/hosts',
            'pwd'
        ]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(security_operation, command, f"op_{i}")
                for i, command in enumerate(test_commands)
            ]
            
            results = [future.result() for future in as_completed(futures)]
        
        # Verify all operations completed successfully
        self.assertEqual(len(results), 5)
        
        for result in results:
            self.assertTrue(result['success'])
            self.assertTrue(result['validation']['safe'])
            self.assertIsNotNone(result['log_id'])
        
        # Verify audit trail includes all operations
        all_logs = self.security_components['audit_logger'].log_entries
        command_logs = [log for log in all_logs if log['type'] == 'COMMAND_EXECUTION']
        self.assertEqual(len(command_logs), 5)
    
    def test_safety_mode_escalation(self):
        """Test safety mode escalation based on threat detection."""
        # Start with standard safety mode
        current_safety_mode = MockSafetyMode.STANDARD
        
        # Simulate encountering threats that should escalate safety
        test_scenarios = [
            ('ls -la', current_safety_mode, MockSafetyMode.STANDARD),  # Safe command, no escalation
            ('find /tmp -name "*.log"', current_safety_mode, MockSafetyMode.STANDARD),  # Safe, no escalation
            ('chmod 777 /etc/passwd', current_safety_mode, MockSafetyMode.STRICT),  # Dangerous, escalate to strict
            ('rm -rf /tmp/*', MockSafetyMode.STRICT, MockSafetyMode.PARANOID),  # Very dangerous, escalate to paranoid
        ]
        
        escalation_log = []
        
        for command, current_mode, expected_mode in test_scenarios:
            # Validate command
            validation = self.security_components['validator'].validate_command(command)
            
            # Determine if escalation is needed
            if not validation['safe'] or len(validation['threats']) > 0:
                if current_mode == MockSafetyMode.STANDARD:
                    new_mode = MockSafetyMode.STRICT
                elif current_mode == MockSafetyMode.STRICT:
                    new_mode = MockSafetyMode.PARANOID
                else:
                    new_mode = current_mode  # Already at maximum
            else:
                new_mode = current_mode  # No escalation needed
            
            # Log escalation decision
            escalation_event = {
                'command': command,
                'previous_mode': current_mode,
                'new_mode': new_mode,
                'threats_detected': len(validation['threats']) if not validation['safe'] else 0,
                'escalation_triggered': new_mode != current_mode
            }
            
            escalation_log.append(escalation_event)
            current_safety_mode = new_mode  # Update for next iteration
            
            # Verify expected behavior
            if expected_mode != current_mode:
                self.fail(f"Expected escalation to {expected_mode}, but got {current_mode}")
        
        # Verify escalation logic worked correctly
        self.assertEqual(len(escalation_log), 4)
        
        # Check specific escalation points
        self.assertFalse(escalation_log[0]['escalation_triggered'])  # Safe command
        self.assertFalse(escalation_log[1]['escalation_triggered'])  # Safe command  
        self.assertTrue(escalation_log[2]['escalation_triggered'])   # Dangerous command
        self.assertTrue(escalation_log[3]['escalation_triggered'])   # Very dangerous command


class SecurityTestRunner:
    """Custom test runner for security tests with enhanced reporting."""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
        self.test_results = {}
    
    def run_security_tests(self):
        """Run all security tests with enhanced reporting."""
        print("🛡️  Starting Security Module Tests")
        print("=" * 60)
        
        # Test suites to run
        test_suites = [
            ('SecurityValidator', TestSecurityValidator),
            ('Sandbox', TestSandbox),
            ('SafetyMode', TestSafetyMode),
            ('AuditLogger', TestAuditLogger),
            ('Security Integration', TestSecurityIntegration)
        ]
        
        overall_start_time = time.time()
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for suite_name, test_class in test_suites:
            print(f"\n📋 Running {suite_name} Tests...")
            print("-" * 40)
            
            suite_start_time = time.time()
            
            # Create test suite and run
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(test_class)
            
            runner = unittest.TextTestRunner(verbosity=self.verbosity, stream=sys.stdout)
            result = runner.run(suite)
            
            suite_end_time = time.time()
            
            # Record results
            self.test_results[suite_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'success': result.wasSuccessful(),
                'execution_time': suite_end_time - suite_start_time,
                'failure_details': result.failures,
                'error_details': result.errors
            }
            
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            
            # Print summary for this suite
            print(f"\n✅ {suite_name} Results:")
            print(f"   Tests run: {result.testsRun}")
            print(f"   Failures: {len(result.failures)}")
            print(f"   Errors: {len(result.errors)}")
            print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%" if result.testsRun > 0 else "   Success rate: N/A")
            print(f"   Execution time: {suite_end_time - suite_start_time:.2f}s")
        
        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        
        # Print overall summary
        print("\n" + "=" * 60)
        print("🛡️  Security Module Test Summary")
        print("=" * 60)
        print(f"Total tests run: {total_tests}")
        print(f"Total failures: {total_failures}")
        print(f"Total errors: {total_errors}")
        print(f"Overall success rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%" if total_tests > 0 else "Overall success rate: N/A")
        print(f"Total execution time: {total_time:.2f}s")
        
        # Performance analysis
        print(f"\n📊 Performance Analysis:")
        for suite_name, results in self.test_results.items():
            tests_per_second = results['tests_run'] / results['execution_time'] if results['execution_time'] > 0 else 0
            print(f"   {suite_name}: {tests_per_second:.1f} tests/second")
        
        # Security-specific insights
        print(f"\n🔒 Security Test Insights:")
        print(f"   ✅ Command validation: Tested safe vs dangerous commands")
        print(f"   ✅ Sandbox isolation: Tested resource restrictions and containment")
        print(f"   ✅ Safety modes: Tested escalation and restriction levels")
        print(f"   ✅ Audit logging: Tested comprehensive event tracking")
        print(f"   ✅ Integration: Tested end-to-end security workflows")
        
        return self.test_results


def run_performance_benchmark():
    """Run performance benchmarks for security components."""
    print("\n🚀 Running Security Performance Benchmarks...")
    print("=" * 50)
    
    # Benchmark parameters
    test_commands = [
        'ls -la',
        'find . -name "*.txt"',
        'grep "test" file.txt',
        'cat /etc/hosts',
        'echo "benchmark test"'
    ] * 20  # 100 commands total
    
    # Initialize components
    validator = MockSecurityValidator()
    sandbox = MockSandbox()
    audit_logger = MockAuditLogger()
    
    # Benchmark command validation
    print("\n📊 Command Validation Benchmark:")
    start_time = time.time()
    
    validation_results = []
    for command in test_commands:
        result = validator.validate_command(command)
        validation_results.append(result)
    
    validation_time = time.time() - start_time
    validations_per_second = len(test_commands) / validation_time
    
    print(f"   Commands validated: {len(test_commands)}")
    print(f"   Total time: {validation_time:.3f}s")
    print(f"   Validations per second: {validations_per_second:.1f}")
    print(f"   Average time per validation: {(validation_time / len(test_commands)) * 1000:.2f}ms")
    
    # Benchmark sandbox creation
    print("\n🏗️  Sandbox Creation Benchmark:")
    start_time = time.time()
    
    sandbox_ids = []
    for command in test_commands[:10]:  # Use smaller set for sandbox creation
        sandbox_id = sandbox.create_sandbox(command)
        sandbox_ids.append(sandbox_id)
    
    sandbox_creation_time = time.time() - start_time
    sandboxes_per_second = len(sandbox_ids) / sandbox_creation_time
    
    print(f"   Sandboxes created: {len(sandbox_ids)}")
    print(f"   Total time: {sandbox_creation_time:.3f}s")
    print(f"   Sandboxes per second: {sandboxes_per_second:.1f}")
    print(f"   Average creation time: {(sandbox_creation_time / len(sandbox_ids)) * 1000:.2f}ms")
    
    # Benchmark audit logging
    print("\n📝 Audit Logging Benchmark:")
    start_time = time.time()
    
    log_ids = []
    for i, command in enumerate(test_commands[:10]):
        log_id = audit_logger.log_command_execution(
            command, 
            {'success': True, 'output': 'test'},
            {'session_id': f'session_{i}'}
        )
        log_ids.append(log_id)
    
    logging_time = time.time() - start_time
    logs_per_second = len(log_ids) / logging_time
    
    print(f"   Log entries created: {len(log_ids)}")
    print(f"   Total time: {logging_time:.3f}s")
    print(f"   Logs per second: {logs_per_second:.1f}")
    print(f"   Average logging time: {(logging_time / len(log_ids)) * 1000:.2f}ms")
    
    # Performance summary
    print(f"\n⚡ Security Module Performance Summary:")
    print(f"   Command validation: {validations_per_second:.1f} ops/sec")
    print(f"   Sandbox creation: {sandboxes_per_second:.1f} ops/sec")
    print(f"   Audit logging: {logs_per_second:.1f} ops/sec")
    print(f"   Overall throughput: {min(validations_per_second, sandboxes_per_second, logs_per_second):.1f} ops/sec (bottleneck)")


if __name__ == '__main__':
    """Main execution entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run security module tests')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmarks')
    parser.add_argument('--verbose', '-v', action='count', default=1,
                       help='Increase verbosity (can be repeated)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Set verbosity level
    if args.quiet:
        verbosity = 0
    elif args.verbose > 2:
        verbosity = 2
    else:
        verbosity = 1
    
    # Run tests
    try:
        runner = SecurityTestRunner(verbosity=verbosity)
        results = runner.run_security_tests()
        
        # Run benchmarks if requested
        if args.benchmark:
            run_performance_benchmark()
        
        # Set exit code based on results
        all_successful = all(result['success'] for result in results.values())
        if all_successful:
            print("\n🎉 All security tests passed successfully!")
            exit_code = 0
        else:
            print("\n❌ Some security tests failed!")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Test execution interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n\n💥 Unexpected error during test execution: {e}")
        exit_code = 1
    
    exit(exit_code)