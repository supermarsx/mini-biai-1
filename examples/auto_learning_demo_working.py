#!/usr/bin/env python3
"""
Auto-Learning System Demo - Working Version

Demonstrates the core functionality of the auto-learning system.
"""

import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components
import sys
sys.path.insert(0, '/workspace/src')

from learning.stdp import STDPManager, STDPType, STDPParameters
from learning.online_learner import OnlineLearner, LearningConfig, LearningMode
from learning.replay_buffer import ExperienceReplayBuffer, ExperienceType
from learning.learning_adaptation import LearningRateAdapter, QualityMonitor, AdaptationStrategy, AdaptationConfig
from learning.circuit_breakers import LearningCircuitBreaker, CircuitBreakerConfig
from learning.metrics import LearningMetrics, MetricType


def demo_stdp():
    """Demonstrate STDP functionality."""
    print("\n=== STDP Learning Demo ===")
    
    # Initialize STDP manager
    stdp = STDPManager(
        n_neurons=20,
        stdp_type=STDPType.ADAPTIVE,
        parameters=STDPParameters(a_plus=0.01, a_minus=0.01)
    )
    
    print(f"Initialized STDP with {stdp.n_neurons} neurons")
    
    # Create spike patterns
    for episode in range(10):
        pre_spikes = np.random.randint(0, 2, 20).astype(np.float32)
        post_spikes = np.random.randint(0, 2, 20).astype(np.float32)
        
        # Update weights
        result = stdp.update_weights(pre_spikes, post_spikes)
        
        if episode % 3 == 0:
            print(f"Episode {episode}: {result['updates']} connections updated")