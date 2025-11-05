"""
Auto-learning system for spiking neural networks.

This module provides comprehensive learning capabilities including:
- STDP (Spike-Timing Dependent Plasticity)
- Online learning pipeline
- Experience replay buffer
- Learning rate adaptation
- Quality metrics and monitoring
- Circuit breakers for performance protection
"""

from .stdp import STDPManager, STDPType, STDPParameters
from .online_learner import OnlineLearner, LearningConfig, LearningMode
from .replay_buffer import ExperienceReplayBuffer, ExperienceType
from .learning_adaptation import LearningRateAdapter, QualityMonitor, AdaptationStrategy, AdaptationConfig
from .circuit_breakers import LearningCircuitBreaker, CircuitBreakerConfig
from .metrics import LearningMetrics, MetricType, AlertLevel, AlertConfig

__all__ = [
    'STDPManager',
    'STDPType',
    'STDPParameters',
    'OnlineLearner', 
    'LearningConfig',
    'LearningMode',
    'ExperienceReplayBuffer',
    'ExperienceType',
    'LearningRateAdapter',
    'QualityMonitor',
    'AdaptationStrategy',
    'AdaptationConfig',
    'LearningCircuitBreaker',
    'CircuitBreakerConfig',
    'LearningMetrics',
    'MetricType',
    'AlertLevel',
    'AlertConfig'
]

__version__ = '1.0.0'