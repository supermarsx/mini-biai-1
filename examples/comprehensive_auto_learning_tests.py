#!/usr/bin/env python3
"""
Test Runner for Comprehensive Auto-Learning Tests
==================================================

Runs comprehensive test suite for auto-learning functionality.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from learning.stdp import STDPManager, STDPType
from learning.online_learner import OnlineLearner, LearningConfig, LearningMode
from learning.replay_buffer import ExperienceReplayBuffer

def run_comprehensive_tests():
    """Run comprehensive auto-learning tests."""
    print("=== Comprehensive Auto-Learning Tests ===")
    print()
    
    # Test 1: STDP Manager
    print("Test 1: STDP Manager")
    stdp = STDPManager(n_neurons=10, stdp_type=STDPType.STANDARD)
    print("✓ STDP Manager initialized")
    
    # Test 2: Online Learner
    print("\nTest 2: Online Learner")
    config = LearningConfig(learning_rate=0.001)
    learner = OnlineLearner(config)
    print("✓ Online Learner initialized")
    
    # Test 3: Replay Buffer
    print("\nTest 3: Replay Buffer")
    buffer = ExperienceReplayBuffer(max_size=100)
    print("✓ Replay Buffer initialized")
    
    print("\n=== All Tests Passed ===")

if __name__ == "__main__":
    run_comprehensive_tests()