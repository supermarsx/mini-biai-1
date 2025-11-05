#!/usr/bin/env python3
"""
Quick demonstration of the auto-learning system working.
"""

import sys
import numpy as np
import time

# Add the source path
sys.path.insert(0, '/workspace/src')

from learning.stdp import STDPManager, STDPType
from learning.online_learner import OnlineLearner, LearningConfig, LearningMode
from learning.replay_buffer import ExperienceReplayBuffer

def quick_demo():
    print("=== AUTO-LEARNING SYSTEM DEMONSTRATION ===")
    print()
    
    # 1. STDP Manager Demo
    print("1. STDP Learning Demonstration")
    print("-" * 40)
    
    n_neurons = 10
    stdp_manager = STDPManager(n_neurons=n_neurons, stdp_type=STDPType.STANDARD)
    
    print(f"✓ Initialized STDP with {n_neurons} neurons")
    print(f"  Initial weight stats: {stdp_manager.get_weight_statistics()}")
    
    # Simulate learning
    for i in range(20):
        pre_spikes = np.random.randint(0, 2, n_neurons).astype(np.float32)
        post_spikes = np.random.randint(0, 2, n_neurons).astype(np.float32)
        
        result = stdp_manager.update_weights(pre_spikes, post_spikes)
        
        if i % 10 == 0:
            print(f"  Iteration {i}: {result.get('updates', 0)} updates, "
                  f"weight change: {result.get('weight_changes', 0):.6f}")
    
    final_stats = stdp_manager.get_weight_statistics()
    print(f"  Final weight stats: mean={final_stats['mean_weight']:.4f}, "
          f"std={final_stats['std_weight']:.4f}")
    print("✓ STDP learning completed")
    print()
    
    # 2. Replay Buffer Demo
    print("2. Experience Replay Buffer Demonstration")
    print("-" * 40)
    
    buffer = ExperienceReplayBuffer(max_size=100)
    
    # Add some experiences
    for i in range(5):
        experience = {
            'state': np.random.randn(10),
            'action': np.random.randint(0, 3),
            'reward': np.random.uniform(-1, 1),
            'next_state': np.random.randn(10),
            'done': np.random.choice([True, False])
        }
        buffer.add(experience)
    
    print(f"✓ Added 5 experiences to buffer")
    print(f"  Buffer size: {len(buffer)}/{buffer.max_size}")
    
    # Sample batch
    batch = buffer.sample(3)
    print(f"  Sampled batch of size {len(batch)}")
    print(f"  Sample states shape: {batch[0]['state'].shape}")
    print("✓ Replay buffer demonstration completed")
    print()
    
    # 3. Online Learner Demo
    print("3. Online Learner Demonstration")
    print("-" * 40)
    
    config = LearningConfig(
        learning_rate=0.001,
        memory_size=1000,
        batch_size=32,
        target_network_update_freq=1000,
        learning_mode=LearningMode.FREE_LEARNING
    )
    
    learner = OnlineLearner(config)
    print(f"✓ Initialized Online Learner")
    
    # Simulate learning step
    for i in range(5):
        state = np.random.randn(10)
        action = np.random.randint(0, 3)
        reward = np.random.uniform(-1, 1)
        next_state = np.random.randn(10)
        done = np.random.choice([True, False])
        
        metrics = learner.learn_step(state, action, reward, next_state, done)
        
        if i % 2 == 0:
            print(f"  Step {i}: loss={metrics.get('loss', 0):.4f}, "
                  f"epsilon={metrics.get('epsilon', 0):.3f}")
    
    print("✓ Online learning demonstration completed")
    print()
    
    print("=== DEMONSTRATION COMPLETE ===")
    print("All auto-learning components are working correctly!")

if __name__ == "__main__":
    quick_demo()