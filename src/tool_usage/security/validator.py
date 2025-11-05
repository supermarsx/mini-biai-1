#!/usr/bin/env python3
"""
Advanced Command Validation and Security Analysis Module

This module provides comprehensive command validation with threat detection,
input sanitization, and security assessment capabilities.

Author: Mini-Biai Framework Team
Version: 1.0.0
Date: 2024
"""

import ast
import json
import logging
import re
import string
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote, unquote

from ..core.exceptions import ValidationError, SecurityError


class ValidationLevel(Enum):
    """Command validation severity levels"""
    PERMISSIVE = "permissive"  # Minimal validation, only obvious threats
    MODERATE = "moderate"      # Balanced validation for most scenarios
    STRICT = "strict"          # Aggressive validation, fewer false negatives
    PARANOID = "paranoid"      # Maximum security, may have more false positives


class ThreatLevel(Enum):
    """Threat assessment severity levels"""
    NONE = "none"          # No threats detected
    LOW = "low"            # Minor security concerns
    MEDIUM = "medium"      # Moderate security risk
    HIGH = "high"          # Significant security risk
    CRITICAL = "critical"  # Severe security threat


@dataclass
class ValidationResult:
    """Result of command validation"""
    is_valid: bool
    threat_level: ThreatLevel
    confidence: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]
    sanitized_command: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ValidationRule:
    """Individual validation rule configuration"""
    name: str
    pattern: str
    threat_level: ThreatLevel
    description: str
    enabled: bool = True
    case_sensitive: bool = False


class CommandValidator:
    """Advanced command validation with threat detection"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """Initialize validator with specified security level"""
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        self._setup_validation_rules()
        self._setup_threat_patterns()
        
    def _setup_validation_rules(self) -> None:
        """Configure validation rules based on security level"""
        self.validation_rules = []
        
        # Shell injection patterns - Critical threat
        self._add_rule("shell_injection", r"[;&|`$()<>]", ThreatLevel.CRITICAL, 
                      "Shell injection attempt detected")
        self._add_rule("command_substitution", r"[$`]", ThreatLevel.CRITICAL,
                      "Command substitution detected")
        self._add_rule("pipeline_manipulation", r"[|]{2,}", ThreatLevel.HIGH,
                      "Potential pipeline manipulation")
        
        # File system threats - High threat
        self._add_rule("file_deletion", r"rm\s+-[rf]+", ThreatLevel.HIGH,
                      "Potential file deletion command")
        self._add_rule("system_files", r"(etc|proc|sys|dev)", ThreatLevel.HIGH,
                      "System directory access attempt")
        self._add_rule("path_traversal", r"[.]{2,}[/]", ThreatLevel.MEDIUM,
                      "Potential path traversal attempt")
        
        # Network threats - Medium to High
        self._add_rule("network_download", r"(wget|curl|nc|ncat)", ThreatLevel.MEDIUM,
                      "Network download command")
        self._add_rule("port_scan", r"(-p\s+\d+|port\s+\d+)", ThreatLevel.MEDIUM,
                      "Port scanning activity")
        
        # Privilege escalation - Critical threat
        self._add_rule("sudo_usage", r"^sudo\s+", ThreatLevel.HIGH,
                      "Privilege escalation attempt")
        self._add_rule("su_command", r"^su\s+", ThreatLevel.HIGH,
                      "User switch attempt")
        
        # Data exfiltration - High threat
        self._add_rule("data_export", r"(>|>>)", ThreatLevel.MEDIUM,
                      "Data redirection detected")
        self._add_rule("copy_operations", r"(cp|mv)\s+", ThreatLevel.LOW,
                      "File copy/move operation")
        
        # Python-specific threats
        self._add_rule("eval_usage", r"\beval\s*\(", ThreatLevel.HIGH,
                      "Dynamic code execution")
        self._add_rule("exec_usage", r"\bexec\s*\(", ThreatLevel.HIGH,
                      "Dynamic code execution")
        
        # Adjust rules based on validation level
        if self.validation_level == ValidationLevel.PERMISSIVE:
            self._filter_rules([ThreatLevel.CRITICAL, ThreatLevel.HIGH])
        elif self.validation_level == ValidationLevel.STRICT:
            self._add_rule("base64_decode", r"base64", ThreatLevel.MEDIUM,
                          "Base64 encoding detected")
        elif self.validation_level == ValidationLevel.PARANOID:
            self._add_rule("hex_pattern", r"0x[0-9a-fA-F]+", ThreatLevel.LOW,
                          "Hexadecimal pattern detected")
            self._add_rule("base64_pattern", r"[A-Za-z0-9+/]{20,}={0,2}", ThreatLevel.LOW,
                          "Potential base64 encoding")
    
    def _add_rule(self, name: str, pattern: str, threat_level: ThreatLevel, 
                  description: str) -> None:
        """Add a validation rule"""
        rule = ValidationRule(
            name=name,
            pattern=pattern,
            threat_level=threat_level,
            description=description
        )
        self.validation_rules.append(rule)
    
    def _filter_rules(self, min_threat_levels: List[ThreatLevel]) -> None:
        """Filter validation rules by minimum threat level"""
        threat_values = {level.value for level in min_threat_levels}
        self.validation_rules = [
            rule for rule in self.validation_rules 
            if rule.threat_level.value in threat_values
        ]
    
    def _setup_threat_patterns(self) -> None:
        """Setup advanced threat detection patterns"""
        # SQL injection patterns
        self.sql_patterns = [
            r"['\";].*--",
            r"union\s+select",
            r"insert\s+into",
            r"update\s+set",
            r"delete\s+from",
            r"drop\s+table",
            r"alter\s+table"
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>"
        ]
        
        # File inclusion patterns
        self.file_inclusion_patterns = [
            r"../",
            r"\.\./\.\./",
            r"%2e%2e%2f",
            r"php://",
            r"data://",
            r"file://"
        ]
        
        # Python AST patterns
        self.python_threat_patterns = [
            "__import__",
            "__builtins__",
            "globals",
            "locals",
            "vars",
            "eval",
            "exec",
            "compile"
        ]
    
    def validate_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a command for security threats"""
        if not isinstance(command, str):
            raise ValidationError("Command must be a string")
        
        # Input sanitization
        sanitized_command = self._sanitize_input(command)
        
        # Threat detection
        threats = self._detect_threats(sanitized_command)
        
        # Python AST analysis if applicable
        python_threats = []
        if self._appears_to_be_python(sanitized_command):
            python_threats = self._analyze_python_syntax(sanitized_command)
            threats.extend(python_threats)
        
        # Determine overall threat level
        max_threat = max([t["level"] for t in threats], default=ThreatLevel.NONE)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(threats)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(threats, sanitized_command)
        
        # Determine if command is valid
        is_valid = (
            max_threat == ThreatLevel.NONE or 
            (max_threat == ThreatLevel.LOW and self.validation_level in [ValidationLevel.PERMISSIVE, ValidationLevel.MODERATE])
        )
        
        return ValidationResult(
            is_valid=is_valid,
            threat_level=max_threat,
            confidence=confidence,
            issues=[t["description"] for t in threats],
            suggestions=suggestions,
            sanitized_command=sanitized_command,
            metadata={
                "original_command": command,
                "validation_level": self.validation_level.value,
                "threat_count": len(threats),
                "python_analysis": len(python_threats) > 0
            }
        )
    
    def _sanitize_input(self, command: str) -> str:
        """Sanitize input to remove obvious threats"""
        # URL decode
        try:
            command = unquote(command)
        except Exception:
            pass
        
        # HTML entity decoding
        html_entities = {
            "&lt;": "<",
            "&gt;": ">",
            "&amp;": "&",
            "&quot;": '"',
            "&#x27;": "'"
        }
        for entity, char in html_entities.items():
            command = command.replace(entity, char)
        
        # Remove null bytes
        command = command.replace('\x00', '')
        
        # Strip whitespace
        command = command.strip()
        
        return command
    
    def _detect_threats(self, command: str) -> List[Dict[str, Any]]:
        """Detect various types of threats in command"""
        threats = []
        
        # Apply validation rules
        for rule in self.validation_rules:
            if rule.enabled:
                flags = 0 if rule.case_sensitive else re.IGNORECASE
                if re.search(rule.pattern, command, flags):
                    threats.append({
                        "rule": rule.name,
                        "level": rule.threat_level,
                        "description": rule.description,
                        "pattern": rule.pattern
                    })
        
        # SQL injection detection
        for pattern in self.sql_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                threats.append({
                    "rule": "sql_injection",
                    "level": ThreatLevel.HIGH,
                    "description": "Potential SQL injection attempt",
                    "pattern": pattern
                })
        
        # XSS detection
        for pattern in self.xss_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                threats.append({
                    "rule": "xss_attempt",
                    "level": ThreatLevel.MEDIUM,
                    "description": "Potential XSS attempt",
                    "pattern": pattern
                })
        
        # File inclusion detection
        for pattern in self.file_inclusion_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                threats.append({
                    "rule": "file_inclusion",
                    "level": ThreatLevel.HIGH,
                    "description": "Potential file inclusion attempt",
                    "pattern": pattern
                })
        
        return threats
    
    def _analyze_python_syntax(self, command: str) -> List[Dict[str, Any]]:
        """Analyze Python code for security threats using AST"""
        threats = []
        
        try:
            # Parse the code
            tree = ast.parse(command)
            
            # Walk the AST
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in self.python_threat_patterns:
                            threats.append({
                                "rule": "python_dangerous_function",
                                "level": ThreatLevel.HIGH,
                                "description": f"Dangerous Python function: {func_name}",
                                "node": func_name
                            })
                
                # Check for attribute access to dangerous objects
                elif isinstance(node, ast.Attribute):
                    if node.attr in ["__globals__", "__locals__", "__dict__"]:
                        threats.append({
                            "rule": "python_dangerous_attribute",
                            "level": ThreatLevel.MEDIUM,
                            "description": f"Dangerous attribute access: {node.attr}",
                            "node": node.attr
                        })
        
        except SyntaxError:
            # Not valid Python, return empty list
            pass
        except Exception as e:
            self.logger.warning(f"Error analyzing Python syntax: {e}")
        
        return threats
    
    def _calculate_confidence(self, threats: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on detected threats"""
        if not threats:
            return 1.0
        
        # Weight by threat level
        weights = {
            ThreatLevel.CRITICAL: 1.0,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.MEDIUM: 0.6,
            ThreatLevel.LOW: 0.3,
            ThreatLevel.NONE: 0.0
        }
        
        total_weight = sum(weights[t["level"]] for t in threats)
        max_possible = len(threats) * 1.0  # Assume max weight
        
        confidence = 1.0 - (total_weight / max_possible)
        return max(0.0, min(1.0, confidence))
    
    def _generate_suggestions(self, threats: List[Dict[str, Any]], command: str) -> List[str]:
        """Generate security suggestions based on detected threats"""
        suggestions = []
        
        for threat in threats:
            rule = threat["rule"]
            
            if rule == "shell_injection":
                suggestions.append("Use parameterized commands to avoid shell injection")
            elif rule == "file_deletion":
                suggestions.append("Verify file paths and permissions before deletion")
            elif rule == "sudo_usage":
                suggestions.append("Minimize sudo usage and implement proper authorization")
            elif rule == "eval_usage":
                suggestions.append("Replace eval() with safer alternatives like ast.literal_eval()")
            elif rule == "sql_injection":
                suggestions.append("Use prepared statements to prevent SQL injection")
            elif rule == "path_traversal":
                suggestions.append("Validate and normalize file paths")
            else:
                suggestions.append(f"Review security implications of: {threat['description']}")
        
        return suggestions
    
    def _assess_threat_level(self, threats: List[Dict[str, Any]]) -> ThreatLevel:
        """Assess the overall threat level from detected threats"""
        if not threats:
            return ThreatLevel.NONE
        
        # Priority: Critical > High > Medium > Low
        for level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH, ThreatLevel.MEDIUM, ThreatLevel.LOW]:
            if any(t["level"] == level for t in threats):
                return level
        
        return ThreatLevel.NONE
    
    def _appears_to_be_python(self, command: str) -> bool:
        """Check if command appears to be Python code"""
        python_indicators = [
            "def ", "class ", "import ", "from ", "print(",
            "if __name__", "try:", "except:", "for ", "while ",
            "lambda ", "assert ", "yield "
        ]
        
        return any(indicator in command for indicator in python_indicators)
    
    def add_custom_rule(self, name: str, pattern: str, threat_level: ThreatLevel, 
                       description: str, case_sensitive: bool = False) -> None:
        """Add a custom validation rule"""
        rule = ValidationRule(
            name=name,
            pattern=pattern,
            threat_level=threat_level,
            description=description,
            case_sensitive=case_sensitive
        )
        self.validation_rules.append(rule)
    
    def remove_rule(self, name: str) -> bool:
        """Remove a validation rule by name"""
        original_count = len(self.validation_rules)
        self.validation_rules = [rule for rule in self.validation_rules if rule.name != name]
        return len(self.validation_rules) < original_count
    
    def get_rules(self) -> List[ValidationRule]:
        """Get all current validation rules"""
        return self.validation_rules.copy()
    
    def export_rules(self) -> Dict[str, Any]:
        """Export rules configuration as dictionary"""
        return {
            "validation_level": self.validation_level.value,
            "rules": [
                {
                    "name": rule.name,
                    "pattern": rule.pattern,
                    "threat_level": rule.threat_level.value,
                    "description": rule.description,
                    "enabled": rule.enabled,
                    "case_sensitive": rule.case_sensitive
                }
                for rule in self.validation_rules
            ]
        }
    
    def import_rules(self, rules_config: Dict[str, Any]) -> None:
        """Import rules configuration from dictionary"""
        if "validation_level" in rules_config:
            self.validation_level = ValidationLevel(rules_config["validation_level"])
        
        if "rules" in rules_config:
            self.validation_rules = []
            for rule_config in rules_config["rules"]:
                rule = ValidationRule(
                    name=rule_config["name"],
                    pattern=rule_config["pattern"],
                    threat_level=ThreatLevel(rule_config["threat_level"]),
                    description=rule_config["description"],
                    enabled=rule_config.get("enabled", True),
                    case_sensitive=rule_config.get("case_sensitive", False)
                )
                self.validation_rules.append(rule)
    
    def validate_batch(self, commands: List[str], 
                      context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Validate multiple commands efficiently"""
        results = []
        for command in commands:
            try:
                result = self.validate_command(command, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to validate command '{command}': {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    threat_level=ThreatLevel.CRITICAL,
                    confidence=0.0,
                    issues=[f"Validation error: {str(e)}"],
                    suggestions=["Fix validation error and retry"]
                ))
        
        return results
    
    def get_security_report(self, commands: List[str]) -> Dict[str, Any]:
        """Generate comprehensive security report for multiple commands"""
        results = self.validate_batch(commands)
        
        # Aggregate statistics
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        
        threat_counts = {level.value: 0 for level in ThreatLevel}
        for result in results:
            threat_counts[result.threat_level.value] += 1
        
        # Common issues
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        from collections import Counter
        common_issues = Counter(all_issues).most_common(10)
        
        return {
            "summary": {
                "total_commands": total,
                "valid_commands": valid,
                "invalid_commands": total - valid,
                "security_score": valid / total if total > 0 else 1.0
            },
            "threat_distribution": threat_counts,
            "common_issues": common_issues,
            "recommendations": [
                "Implement stricter validation for critical operations",
                "Add input sanitization for user-provided data",
                "Monitor commands with high threat levels",
                "Regular security audit of validation rules"
            ]
        }


class SecurityError(Exception):
    """Security-related exceptions"""
    pass


class ValidationError(Exception):
    """Validation-related exceptions"""
    pass


def validate_input(input_data: str, validator: Optional[CommandValidator] = None) -> ValidationResult:
    """Convenience function for input validation"""
    if validator is None:
        validator = CommandValidator(ValidationLevel.MODERATE)
    
    return validator.validate_command(input_data)


def create_strict_validator() -> CommandValidator:
    """Create a strict validator for high-security environments"""
    return CommandValidator(ValidationLevel.STRICT)


def create_permissive_validator() -> CommandValidator:
    """Create a permissive validator for development environments"""
    return CommandValidator(ValidationLevel.PERMISSIVE)


def create_paranoid_validator() -> CommandValidator:
    """Create a paranoid validator for maximum security"""
    return CommandValidator(ValidationLevel.PARANOID)


# Export main classes and functions
__all__ = [
    "CommandValidator",
    "ValidationResult", 
    "ValidationRule",
    "ValidationLevel",
    "ThreatLevel",
    "validate_input",
    "create_strict_validator",
    "create_permissive_validator",
    "create_paranoid_validator"
]