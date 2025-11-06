"""
Meta-Learning Framework Foundation
Advanced meta-learning capabilities for adaptive AI systems.

This module provides comprehensive meta-learning implementations including:
- Model-Agnostic Meta-Learning (MAML)
- Few-shot learning with prototypical networks
- Continual learning with catastrophic forgetting prevention
- Neural Architecture Search integration
- Tool usage optimization through meta-learning
- Meta-learning adapters for various model architectures

Key Components:
- MAML with first-order and second-order gradient support
- Prototypical networks for few-shot classification
- Elastic Weight Consolidation (EWC) for continual learning
- Neural Turing Machine integration
- Adapters for language, vision, and multi-modal tasks
- Meta-learning for routing optimization
- Bayesian hyperparameter optimization
"""

from .maml import MAML, MAMLFirstOrder, MAMLSecondOrder
from .few_shot import PrototypicalNetwork, FewShotLearner, RelationNetwork
from .continual import ContinualLearner, EWC, ProgressiveNetworks
from .optimizer import AdaptiveOptimizer, BayesianOptimizer, LearningRateScheduler
from .nas_integration import NASOptimizer, NeuralArchitectureSearch, DARTS
from .tool_optimization import ToolMetaLearner, ToolRoutingOptimizer
from .adapter import (
    AdapterModel, 
    MultiModalAdapter, 
    LanguageAdapter, 
    VisionAdapter,
    LinearAdapter,
    LoraAdapter,
    PrefixTuningAdapter
)

__all__ = [
    # MAML variants
    'MAML',
    'MAMLFirstOrder', 
    'MAMLSecondOrder',
    
    # Few-shot learning
    'PrototypicalNetwork',
    'FewShotLearner',
    'RelationNetwork',
    
    # Continual learning
    'ContinualLearner',
    'EWC',
    'ProgressiveNetworks',
    
    # Optimizers
    'AdaptiveOptimizer',
    'BayesianOptimizer', 
    'LearningRateScheduler',
    
    # NAS
    'NASOptimizer',
    'NeuralArchitectureSearch',
    'DARTS',
    
    # Tool optimization
    'ToolMetaLearner',
    'ToolRoutingOptimizer',
    
    # Adapters
    'AdapterModel',
    'MultiModalAdapter',
    'LanguageAdapter',
    'VisionAdapter',
    'LinearAdapter',
    'LoraAdapter',
    'PrefixTuningAdapter',
]

__version__ = "1.0.0"

# Framework configuration
FRAMEWORK_CONFIG = {
    'default_maml_order': 'first',  # 'first' or 'second'
    'default_few_shot_method': 'prototypical',  # 'prototypical' or 'relation'
    'default_continual_method': 'ewc',  # 'ewc', 'progressive', or 'replay'
    'enable_nas': True,
    'enable_tool_optimization': True,
    'enable_bayesian_optimization': True,
    'max_meta_steps': 1000,
    'adapters_support': {
        'language': ['linear', 'lora', 'prefix'],
        'vision': ['linear', 'lora', 'adapter'],
        'multimodal': ['multi_adapter']
    }
}

def get_config():
    """Get the current framework configuration."""
    return FRAMEWORK_CONFIG.copy()

def set_config(**kwargs):
    """Update the framework configuration."""
    FRAMEWORK_CONFIG.update(kwargs)