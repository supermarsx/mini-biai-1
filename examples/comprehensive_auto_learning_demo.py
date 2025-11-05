#!/usr/bin/env python3
"""
Comprehensive Auto-Learning System Demo and Test Suite

This script demonstrates the complete auto-learning system with:
1. STDP (Spike-Timing Dependent Plasticity)
2. Online learning pipeline with experience replay
3. Learning rate adaptation and performance monitoring
4. Circuit breakers for performance protection
5. Comprehensive metrics and quality assessment

Usage:
    python comprehensive_auto_learning_demo.py
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any
import json

# Import the learning system components
import sys
sys.path.insert(0, '/workspace/src')

from learning import (
    STDPManager, STDPType, STDPParameters,
    OnlineLearner, LearningConfig, LearningMode,
    ExperienceReplayBuffer, ExperienceType,
    LearningRateAdapter, QualityMonitor, AdaptationStrategy,
    LearningCircuitBreaker, CircuitBreakerConfig,
    LearningMetrics, MetricType, AlertLevel, AlertConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveAutoLearningDemo:
    """Complete demonstration of the auto-learning system."""
    
    def __init__(self):
        self.n_neurons = 50
        self.setup_components()
        
    def setup_components(self):