#!/usr/bin/env python3
"""
Configuration Validation and Migration Tools for mini-biai-1 Step 2

This module provides comprehensive tools for:
1. Validating Step 2 configuration files
2. Migrating configurations between versions
3. Configuration template inheritance
4. Configuration consistency checking
5. Performance impact analysis
"""

import yaml
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import re
from datetime import datetime

try:
    from pydantic import BaseModel, Field, validator, ValidationError
except ImportError:
    print("Warning: pydantic not available. Using basic validation.")


@dataclass
class ValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        self.warnings.append(warning)
        
    def add_suggestion(self, suggestion: str):
        self.suggestions.append(suggestion)


class ConfigValidator:
    """Validates Step 2 configuration files"""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validation_rules = self._load_validation_rules()
        
    def validate_config(self, config_path: Union[str, Path]) -> ValidationResult:
        """Validate a configuration file"""
        config_path = Path(config_path)
        result = ValidationResult(is_valid=True, errors=[], warnings=[], suggestions=[])
        
        try:
            # Load configuration
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    config = json.load(f)
                else:
                    config = yaml.safe_load(f)
        except Exception as e:
            result.add_error(f"Failed to load configuration: {e}")
            return result
            
        # Validate structure
        self._validate_structure(config, result)
        
        # Validate required fields
        self._validate_required_fields(config, result)
        
        # Validate value ranges
        self._validate_value_ranges(config, result)
        
        # Validate expert configurations
        self._validate_expert_configs(config, result)
        
        # Validate routing configuration
        self._validate_routing_config(config, result)
        
        # Validate affect modulation
        self._validate_affect_config(config, result)
        
        # Validate SSM configuration
        self._validate_ssm_config(config, result)
        
        # Validate auto-learning
        self._validate_auto_learning_config(config, result)
        
        # Validate performance constraints
        self._validate_performance_config(config, result)
        
        return result
        
    def _validate_structure(self, config: Dict, result: ValidationResult):
        """Validate basic configuration structure"""
        required_top_level = [
            'general', 'routing', 'affect', 'language_backbone', 
            'auto_learning', 'memory', 'performance_tuning', 'training',
            'evaluation', 'logging'
        ]
        
        for key in required_top_level:
            if key not in config:
                result.add_error(f"Missing required top-level key: {key}")
                
        # Check for expert configurations
        if 'routing' in config and 'experts' in config['routing']:
            required_experts = ['language', 'vision', 'symbolic', 'affect']
            configured_experts = list(config['routing']['experts'].keys())
            
            for expert in required_experts:
                if expert not in configured_experts:
                    result.add_error(f"Missing required expert: {expert}")
                    
    def _validate_required_fields(self, config: Dict, result: ValidationResult):
        """Validate required field presence"""
        # General settings
        if 'general' in config:
            general = config['general']
            required_general = ['project_name', 'version', 'random_seed']
            for field in required_general:
                if field not in general:
                    result.add_error(f"Missing required field: general.{field}")
                    
        # Routing configuration
        if 'routing' in config:
            routing = config['routing']
            required_routing = ['n_experts', 'top_k', 'temperature', 'spike_threshold']
            for field in required_routing:
                if field not in routing:
                    result.add_error(f"Missing required field: routing.{field}")
                    
        # Affect configuration
        if 'affect' in config:
            affect = config['affect']
            if not affect.get('enabled', False):
                result.add_warning("Affect system is disabled. Consider enabling for full functionality.")
                
        # Performance tuning
        if 'performance_tuning' in config:
            perf = config['performance_tuning']
            if 'latency' in perf and 'target_latency_ms' in perf['latency']:
                target_latency = perf['latency']['target_latency_ms']
                if target_latency > 200:
                    result.add_warning(f"Target latency {target_latency}ms may be too high for real-time applications")
                    
    def _validate_value_ranges(self, config: Dict, result: ValidationResult):
        """Validate configuration values are within acceptable ranges"""
        # Routing parameters
        if 'routing' in config:
            routing = config['routing']
            
            if 'n_experts' in routing:
                n_experts = routing['n_experts']
                if not isinstance(n_experts, int) or n_experts < 2 or n_experts > 16:
                    result.add_error(f"n_experts must be between 2 and 16, got {n_experts}")
                    
            if 'top_k' in routing:
                top_k = routing['top_k']
                if not isinstance(top_k, int) or top_k < 1:
                    result.add_error(f"top_k must be a positive integer, got {top_k}")
                    
            if 'temperature' in routing:
                temp = routing['temperature']
                if not isinstance(temp, (int, float)) or temp <= 0 or temp > 2:
                    result.add_error(f"temperature must be between 0 and 2, got {temp}")
                    
        # Affect parameters
        if 'affect' in config:
            affect = config['affect']
            if 'state_representation' in affect:
                rep = affect['state_representation']
                if 'emotion_categories' in rep:
                    cats = rep['emotion_categories']
                    if not isinstance(cats, int) or cats < 4 or cats > 20:
                        result.add_error(f"emotion_categories must be between 4 and 20, got {cats}")
                        
        # Performance parameters
        if 'performance_tuning' in config:
            perf = config['performance_tuning']
            if 'latency' in perf and 'target_latency_ms' in perf['latency']:
                latency = perf['latency']['target_latency_ms']
                if not isinstance(latency, (int, float)) or latency < 10 or latency > 1000:
                    result.add_error(f"target_latency_ms must be between 10 and 1000, got {latency}")
                    
    def _validate_expert_configs(self, config: Dict, result: ValidationResult):
        """Validate expert-specific configurations"""
        if 'routing' not in config or 'experts' not in config['routing']:
            return
            
        experts = config['routing']['experts']
        required_expert_fields = ['name', 'domain', 'specializations']
        
        for expert_name, expert_config in experts.items():
            for field in required_expert_fields:
                if field not in expert_config:
                    result.add_error(f"Expert {expert_name} missing required field: {field}")
                    
            # Validate activation threshold
            if 'activation_threshold' in expert_config:
                threshold = expert_config['activation_threshold']
                if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
                    result.add_error(f"Expert {expert_name} activation_threshold must be between 0 and 1, got {threshold}")
                    
    def _validate_routing_config(self, config: Dict, result: ValidationResult):
        """Validate routing-specific configuration"""
        if 'routing' not in config:
            return
            
        routing = config['routing']
        
        # Check top_k vs n_experts consistency
        if 'n_experts' in routing and 'top_k' in routing:
            n_experts = routing['n_experts']
            top_k = routing['top_k']
            if top_k > n_experts:
                result.add_error(f"top_k ({top_k}) cannot be greater than n_experts ({n_experts})")
                
        # Validate gating configuration
        if 'gating' in routing:
            gating = routing['gating']
            if 'method' in gating:
                method = gating['method']
                valid_methods = ['top_k', 'sparse_gating', 'learned_gating']
                if method not in valid_methods:
                    result.add_error(f"Invalid gating method: {method}. Must be one of {valid_methods}")
                    
    def _validate_affect_config(self, config: Dict, result: ValidationResult):
        """Validate affect modulation configuration"""
        if 'affect' not in config:
            return
            
        affect = config['affect']
        
        # Check affect dimensions
        if 'state_representation' in affect:
            rep = affect['state_representation']
            if 'dimensions' in rep:
                dimensions = rep['dimensions']
                valid_dimensions = ['valence', 'arousal', 'dominance']
                for dim in dimensions:
                    if dim not in valid_dimensions:
                        result.add_error(f"Invalid affect dimension: {dim}. Must be one of {valid_dimensions}")
                        
    def _validate_ssm_config(self, config: Dict, result: ValidationResult):
        """Validate SSM/linear-attention configuration"""
        if 'language_backbone' not in config:
            return
            
        backbone = config['language_backbone']
        
        # Check backbone type
        if 'type' in backbone:
            btype = backbone['type']
            valid_types = ['mamba', 'linear_attention', 'transformer']
            if btype not in valid_types:
                result.add_error(f"Invalid backbone type: {btype}. Must be one of {valid_types}")
                
        # Validate Mamba configuration
        if btype == 'mamba' and 'mamba' in backbone:
            mamba = backbone['mamba']
            required_mamba = ['d_state', 'd_conv', 'expand']
            for field in required_mamba:
                if field not in mamba:
                    result.add_error(f"Mamba configuration missing required field: {field}")
                    
    def _validate_auto_learning_config(self, config: Dict, result: ValidationResult):
        """Validate auto-learning system configuration"""
        if 'auto_learning' not in config:
            return
            
        auto_learn = config['auto_learning']
        
        # Check STDP configuration
        if 'stdp' in auto_learn:
            stdp = auto_learn['stdp']
            if stdp.get('enabled', False):
                # Validate STDP parameters
                if 'learning_rate' in stdp:
                    lr = stdp['learning_rate']
                    if not isinstance(lr, (int, float)) or lr <= 0 or lr > 0.1:
                        result.add_error(f"STDP learning_rate must be between 0 and 0.1, got {lr}")
                        
    def _validate_performance_config(self, config: Dict, result: ValidationResult):
        """Validate performance tuning configuration"""
        if 'performance_tuning' not in config:
            return
            
        perf = config['performance_tuning']
        
        # Check latency budget allocation
        if 'latency' in perf and 'budget_allocation' in perf['latency']:
            budget = perf['latency']['budget_allocation']
            total_budget = sum(budget.values())
            target = perf['latency'].get('target_latency_ms', 0)
            
            if abs(total_budget - target) > 5:  # Allow 5ms tolerance
                result.add_warning(f"Latency budget allocation ({total_budget}ms) doesn't match target ({target}ms)")
                
    def _load_validation_rules(self) -> Dict:
        """Load validation rules from configuration"""
        return {
            'min_latency': 10,
            'max_latency': 1000,
            'min_experts': 2,
            'max_experts': 16,
            'min_emotions': 4,
            'max_emotions': 20
        }


class ConfigMigrator:
    """Migrates configurations between versions"""
    
    def __init__(self):
        self.migration_history = {}
        
    def migrate_step1_to_step2(self, step1_config: Union[str, Path], 
                             step2_config: Union[str, Path]) -> bool:
        """Migrate from Step 1 to Step 2 configuration"""
        try:
            # Load Step 1 config
            with open(step1_config, 'r') as f:
                if Path(step1_config).suffix.lower() == '.json':
                    step1_data = json.load(f)
                else:
                    step1_data = yaml.safe_load(f)
                    
            # Create Step 2 configuration structure
            step2_data = self._create_step2_structure(step1_data)
            
            # Migrate data
            self._migrate_general_settings(step1_data, step2_data)
            self._migrate_memory_settings(step1_data, step2_data)
            self._migrate_training_settings(step1_data, step2_data)
            self._migrate_inference_settings(step1_data, step2_data)
            self._add_step2_features(step2_data)
            
            # Save migrated configuration
            with open(step2_config, 'w') as f:
                if Path(step2_config).suffix.lower() == '.json':
                    json.dump(step2_data, f, indent=2)
                else:
                    yaml.dump(step2_data, f, default_flow_style=False, indent=2)
                    
            print(f"Successfully migrated Step 1 config to {step2_config}")
            return True
            
        except Exception as e:
            print(f"Migration failed: {e}")
            return False
            
    def _create_step2_structure(self, step1_data: Dict) -> Dict:
        """Create Step 2 configuration structure"""
        # Use the step2_base.yaml as template
        template_path = Path(__file__).parent / "step2_base.yaml"
        with open(template_path, 'r') as f:
            step2_data = yaml.safe_load(f)
            
        # Add migration metadata
        step2_data['migration'] = {
            'from_version': '0.1.0',
            'to_version': '0.2.0',
            'migration_date': datetime.now().isoformat(),
            'automated': True
        }
        
        return step2_data
        
    def _migrate_general_settings(self, step1: Dict, step2: Dict):
        """Migrate general settings"""
        if 'general' in step1:
            general1 = step1['general']
            general2 = step2['general']
            
            # Migrate existing fields
            for field in ['project_name', 'version', 'debug', 'random_seed', 'device']:
                if field in general1:
                    general2[field] = general1[field]
                    
            # Update version
            general2['version'] = '0.2.0'
            general2['project_name'] = f"{general1.get('project_name', 'mini-biai-1')}_step2"
            
    def _migrate_memory_settings(self, step1: Dict, step2: Dict):
        """Migrate memory settings"""
        if 'memory' in step1:
            memory1 = step1['memory']
            memory2 = step2['memory']
            
            # Migrate existing memory settings
            for mem_type in ['working_memory', 'episodic_memory', 'semantic_memory']:
                if mem_type in memory1:
                    if mem_type not in memory2:
                        memory2[mem_type] = {}
                    memory2[mem_type].update(memory1[mem_type])
                    
    def _migrate_training_settings(self, step1: Dict, step2: Dict):
        """Migrate training settings"""
        if 'training' in step1:
            train1 = step1['training']
            train2 = step2['training']
            
            # Migrate existing training parameters
            for field in ['optimizer', 'learning_rate', 'weight_decay', 'max_epochs']:
                if field in train1:
                    train2[field] = train1[field]
                    
    def _migrate_inference_settings(self, step1: Dict, step2: Dict):
        """Migrate inference settings"""
        if 'inference' in step1:
            inf1 = step1['inference']
            inf2 = step2['inference']
            
            # Migrate inference parameters
            for field in ['temperature', 'top_k', 'top_p', 'max_length']:
                if field in inf1:
                    inf2[field] = inf1[field]
                    
    def _add_step2_features(self, step2: Dict):
        """Add Step 2 specific features with default values"""
        # Ensure all required Step 2 sections exist with defaults
        step2_defaults = {
            'routing': {
                'n_experts': 4,
                'top_k': 2,
                'temperature': 0.1,
                'experts': {
                    'language': {
                        'name': 'Language Expert',
                        'domain': 'text_processing',
                        'specializations': ['text_generation', 'natural_language_understanding']
                    },
                    'vision': {
                        'name': 'Vision Expert',
                        'domain': 'visual_processing',
                        'specializations': ['image_understanding', 'multimodal_alignment']
                    },
                    'symbolic': {
                        'name': 'Symbolic Expert',
                        'domain': 'symbolic_processing',
                        'specializations': ['logical_reasoning', 'mathematical_computation']
                    },
                    'affect': {
                        'name': 'Affect Expert',
                        'domain': 'affect_processing',
                        'specializations': ['emotion_recognition', 'sentiment_analysis']
                    }
                }
            },
            'affect': {
                'enabled': True,
                'log_only': True,
                'state_representation': {
                    'dimensions': ['valence', 'arousal', 'dominance'],
                    'emotion_categories': 8
                }
            },
            'auto_learning': {
                'enabled': True,
                'stdp': {
                    'enabled': True,
                    'learning_rate': 0.001
                }
            }
        }
        
        for section, defaults in step2_defaults.items():
            if section not in step2:
                step2[section] = defaults
            else:
                self._deep_update(step2[section], defaults)
                
    def _deep_update(self, target: Dict, source: Dict):
        """Deep update target dictionary with source"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value


class ConfigTemplateManager:
    """Manages configuration templates and inheritance"""
    
    def __init__(self, template_dir: Union[str, Path]):
        self.template_dir = Path(template_dir)
        self.templates = {}
        self._load_templates()
        
    def _load_templates(self):
        """Load all template files"""
        for template_file in self.template_dir.glob("*_template.yaml"):
            template_name = template_file.stem.replace("_template", "")
            try:
                with open(template_file, 'r') as f:
                    self.templates[template_name] = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load template {template_file}: {e}")
                
    def merge_templates(self, base_config: Dict, expert_templates: List[str]) -> Dict:
        """Merge base configuration with expert templates"""
        merged_config = base_config.copy()
        
        for template_name in expert_templates:
            if template_name in self.templates:
                template = self.templates[template_name]
                merged_config = self._apply_template(merged_config, template)
            else:
                print(f"Warning: Template {template_name} not found")
                
        return merged_config
        
    def _apply_template(self, config: Dict, template: Dict) -> Dict:
        """Apply template to configuration"""
        merged = config.copy()
        
        # Handle inheritance
        if 'inherit_from' in template:
            inherit_from = template['inherit_from']
            # In a real implementation, this would resolve inheritance
            print(f"Template inheritance from {inherit_from} not implemented")
            
        # Merge template sections
        for section_name, section_config in template.items():
            if section_name == 'inherit_from':
                continue
                
            if section_name not in merged:
                merged[section_name] = {}
                
            # Deep merge
            self._deep_update(merged[section_name], section_config)
            
        return merged


class ConfigConsistencyChecker:
    """Checks configuration consistency and dependencies"""
    
    def __init__(self):
        self.dependency_rules = self._load_dependency_rules()
        
    def check_consistency(self, config: Dict) -> List[str]:
        """Check configuration consistency"""
        issues = []
        
        # Check expert-routing consistency
        issues.extend(self._check_expert_routing_consistency(config))
        
        # Check latency budget consistency
        issues.extend(self._check_latency_consistency(config))
        
        # Check memory configuration consistency
        issues.extend(self._check_memory_consistency(config))
        
        # Check affect integration consistency
        issues.extend(self._check_affect_consistency(config))
        
        return issues
        
    def _check_expert_routing_consistency(self, config: Dict) -> List[str]:
        """Check expert and routing configuration consistency"""
        issues = []
        
        if 'routing' not in config or 'experts' not in config['routing']:
            return issues
            
        routing = config['routing']
        experts = routing['experts']
        
        # Check n_experts consistency
        declared_experts = routing.get('n_experts', 0)
        actual_experts = len(experts)
        
        if declared_experts != actual_experts:
            issues.append(
                f"Inconsistent expert count: declared {declared_experts}, "
                f"but found {actual_experts} expert configurations"
            )
            
        # Check top_k feasibility
        if 'top_k' in routing and routing['top_k'] > actual_experts:
            issues.append(
                f"top_k ({routing['top_k']}) cannot exceed number of experts ({actual_experts})"
            )
            
        return issues
        
    def _check_latency_consistency(self, config: Dict) -> List[str]:
        """Check latency budget consistency"""
        issues = []
        
        if 'performance_tuning' not in config or 'latency' not in config['performance_tuning']:
            return issues
            
        latency = config['performance_tuning']['latency']
        
        if 'target_latency_ms' in latency and 'budget_allocation' in latency:
            target = latency['target_latency_ms']
            allocation = latency['budget_allocation']
            total_allocated = sum(allocation.values())
            
            if total_allocated > target + 10:  # 10ms tolerance
                issues.append(
                    f"Total latency budget allocation ({total_allocated}ms) exceeds "
                    f"target ({target}ms)"
                )
                
        return issues
        
    def _check_memory_consistency(self, config: Dict) -> List[str]:
        """Check memory configuration consistency"""
        issues = []
        
        if 'memory' not in config:
            return issues
            
        memory = config['memory']
        
        # Check STM-LTM consistency
        if 'stm' in memory and 'ltm' in memory:
            stm_capacity = memory['stm'].get('capacity', 0)
            ltm_capacity = memory['ltm'].get('capacity', 0)
            
            if stm_capacity >= ltm_capacity:
                issues.append(
                    f"STM capacity ({stm_capacity}) should be less than "
                    f"LTM capacity ({ltm_capacity})"
                )
                
        return issues
        
    def _check_affect_consistency(self, config: Dict) -> List[str]:
        """Check affect system consistency"""
        issues = []
        
        if 'affect' not in config:
            return issues
            
        affect = config['affect']
        
        # Check if affect expert exists when affect is enabled
        if affect.get('enabled', False):
            if 'routing' in config and 'experts' in config['routing']:
                if 'affect' not in config['routing']['experts']:
                    issues.append("Affect system is enabled but no affect expert configured")
                    
        return issues
        
    def _load_dependency_rules(self) -> Dict:
        """Load dependency rules for consistency checking"""
        return {
            'expert_routing': {
                'description': 'Expert count must match routing configuration',
                'severity': 'error'
            },
            'latency_budget': {
                'description': 'Latency allocation must not exceed target',
                'severity': 'warning'
            },
            'memory_hierarchy': {
                'description': 'Memory hierarchy should follow STM < LTM',
                'severity': 'warning'
            },
            'affect_expert': {
                'description': 'Affect system requires affect expert',
                'severity': 'error'
            }
        }


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="mini-biai-1 Step 2 Configuration Tools")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('config_file', help='Configuration file to validate')
    validate_parser.add_argument('--strict', action='store_true', help='Enable strict validation')
    
    # Migration command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate configuration')
    migrate_parser.add_argument('step1_config', help='Step 1 configuration file')
    migrate_parser.add_argument('step2_config', help='Output Step 2 configuration file')
    
    # Template merge command
    template_parser = subparsers.add_parser('merge_templates', help='Merge configuration templates')
    template_parser.add_argument('base_config', help='Base configuration file')
    template_parser.add_argument('experts', nargs='+', help='Expert templates to merge')
    template_parser.add_argument('output', help='Output merged configuration')
    
    # Consistency check command
    check_parser = subparsers.add_parser('check_consistency', help='Check configuration consistency')
    check_parser.add_argument('config_file', help='Configuration file to check')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        validator = ConfigValidator(strict_mode=args.strict)
        result = validator.validate_config(args.config_file)
        
        print(f"Validation result: {'PASS' if result.is_valid else 'FAIL'}")
        for error in result.errors:
            print(f"ERROR: {error}")
        for warning in result.warnings:
            print(f"WARNING: {warning}")
        for suggestion in result.suggestions:
            print(f"SUGGESTION: {suggestion}")
            
    elif args.command == 'migrate':
        migrator = ConfigMigrator()
        success = migrator.migrate_step1_to_step2(args.step1_config, args.step2_config)
        sys.exit(0 if success else 1)
        
    elif args.command == 'merge_templates':
        # Load base config
        with open(args.base_config, 'r') as f:
            if Path(args.base_config).suffix.lower() == '.json':
                base_config = json.load(f)
            else:
                base_config = yaml.safe_load(f)
                
        # Merge templates
        template_dir = Path(__file__).parent
        template_manager = ConfigTemplateManager(template_dir)
        merged_config = template_manager.merge_templates(base_config, args.experts)
        
        # Save result
        with open(args.output, 'w') as f:
            if Path(args.output).suffix.lower() == '.json':
                json.dump(merged_config, f, indent=2)
            else:
                yaml.dump(merged_config, f, default_flow_style=False, indent=2)
                
        print(f"Merged configuration saved to {args.output}")
        
    elif args.command == 'check_consistency':
        # Load config
        with open(args.config_file, 'r') as f:
            if Path(args.config_file).suffix.lower() == '.json':
                config = json.load(f)
            else:
                config = yaml.safe_load(f)
                
        # Check consistency
        checker = ConfigConsistencyChecker()
        issues = checker.check_consistency(config)
        
        if not issues:
            print("Configuration is consistent")
        else:
            print("Found consistency issues:")
            for issue in issues:
                print(f"  - {issue}")
                
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

if __name__ == '__main__':
    main()