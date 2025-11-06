"""
Example Usage of Auto-Learning System for Spiking Neural Networks.

This script demonstrates how to use the comprehensive auto-learning system
including STDP, online learning, replay buffer, adaptation, and monitoring.
"""

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from src.learning import (
    STDPManager, STDPType, STDPParameters,
    OnlineLearner, LearningConfig, LearningMode,
    ExperienceReplayBuffer, ExperienceType,
    LearningRateAdapter, QualityMonitor, AdaptationStrategy,
    LearningCircuitBreaker, CircuitBreakerConfig,
    LearningMetrics, MetricType
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_data(n_neurons: int, n_samples: int) -> List[Dict[str, Any]]:
    """Create sample spike data for demonstration."""
    data = []
    
    for i in range(n_samples):
        # Create correlated spike patterns
        base_pattern = np.random.random(n_neurons) < 0.1
        
        # Add some structure
        if i % 10 == 0:
            # Special patterns
            base_pattern = np.zeros(n_neurons)
            base_pattern[i % n_neurons] = 1.0
        elif i % 5 == 0:
            # Burst patterns
            base_pattern = np.random.random(n_neurons) < 0.3
            
        # Create pre and post spike patterns with some correlation
        pre_spikes = base_pattern.copy()
        post_spikes = np.zeros(n_neurons)
        
        # Simulate some synaptic transmission
        for j in range(n_neurons):
            if pre_spikes[j] and np.random.random() < 0.4:
                post_spikes[j] = 1.0
                
        # Add reward signal
        reward = np.array([1.0 if np.sum(post_spikes) > 0 else 0.0])
        
        data.append({
            "pre_spikes": pre_spikes,
            "post_spikes": post_spikes,
            "rewards": reward,
            "metadata": {
                "sample_id": i,
                "pattern_type": "special" if i % 10 == 0 else "normal"
            }
        })
        
    return data


def basic_stdp_demo():
    """Demonstrate basic STDP functionality."""
    print("=" * 60)
    print("BASIC STDP DEMONSTRATION")
    print("=" * 60)
    
    # Initialize STDP manager
    n_neurons = 20
    stdp_manager = STDPManager(
        n_neurons=n_neurons,
        stdp_type=STDPType.STANDARD,
        parameters=STDPParameters(
            a_plus=0.01,
            a_minus=0.01,
            tau_plus=20.0,
            tau_minus=20.0
        )
    )
    
    print(f"Initialized STDP manager with {n_neurons} neurons")
    print(f"Connection matrix shape: {stdp_manager.connection_matrix.shape}")
    print(f"Initial weight statistics: {stdp_manager.get_weight_statistics()}")
    print()
    
    # Process learning data
    learning_data = create_sample_data(n_neurons, 100)
    
    total_updates = 0
    total_weight_changes = 0.0
    
    for i, data in enumerate(learning_data):
        result = stdp_manager.update_weights(
            pre_spikes=data["pre_spikes"],
            post_spikes=data["post_spikes"]
        )
        
        total_updates += result.get("updates", 0)
        total_weight_changes += result.get("weight_changes", 0.0)
        
        # Print progress
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/100 samples:")
            print(f"  Updates: {result.get('updates', 0)}, "
                  f"Total updates: {total_updates}")
            print(f"  Weight changes: {result.get('weight_changes', 0):.6f}, "
                  f"Total: {total_weight_changes:.6f}")
            print()
    
    # Final statistics
    print("Final STDP Statistics:")
    print(f"  {stdp_manager.get_weight_statistics()}")
    print(f"  {stdp_manager.get_recent_update_stats()}")
    print()


def replay_buffer_demo():
    """Demonstrate experience replay buffer."""
    print("=" * 60)
    print("EXPERIENCE REPLAY BUFFER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize replay buffer
    buffer = ExperienceReplayBuffer(
        max_size=50,
        n_neurons=10,
        enable_priority=True,
        enable_novelty=True
    )
    
    print("Initialized replay buffer with priority and novelty detection")
    
    # Add experiences
    sample_data = create_sample_data(10, 30)
    
    for i, data in enumerate(sample_data):
        exp_type = ExperienceType.SPIKE_PATTERN
        if i % 10 == 0:
            exp_type = ExperienceType.NOVEL_STIMULUS
        elif i % 5 == 0:
            exp_type = ExperienceType.REWARDED_ACTION
            
        exp_id = buffer.add_experience(data, exp_type)
        
        if i % 10 == 0:
            print(f"Added experience {exp_id}, type: {exp_type.value}")
    
    print(f"\nBuffer statistics: {buffer.get_statistics()}")
    
    # Sample batches
    print("\nSampling batches:")
    for batch_size in [5, 10]:
        batch = buffer.sample_batch(
            batch_size=batch_size,
            require_novelty=False
        )
        print(f"  Sampled {len(batch)} experiences (requested {batch_size})")
        
        # Show some batch details
        if batch:
            avg_priority = np.mean([exp["priority"] for exp in batch])
            avg_novelty = np.mean([exp["novelty_score"] for exp in batch])
            print(f"    Average priority: {avg_priority:.3f}")
            print(f"    Average novelty: {avg_novelty:.3f}")
    
    print()


def online_learner_demo():
    """Demonstrate online learner functionality."""
    print("=" * 60)
    print("ONLINE LEARNER DEMONSTRATION")
    print("=" * 60)
    
    # Configure learning
    config = LearningConfig(
        learning_mode=LearningMode.CONTINUOUS,
        batch_size=16,
        replay_buffer_size=100,
        min_replay_samples=20,
        update_interval=0.1,
        adaptation_interval=5.0,
        circuit_breaker_enabled=True
    )
    
    # Initialize online learner
    learner = OnlineLearner(
        n_neurons=15,
        config=config,
        stdp_type=STDPType.ADAPTIVE
    )
    
    print(f"Initialized online learner")
    print(f"  Learning mode: {config.learning_mode.value}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Update interval: {config.update_interval}s")
    print()
    
    # Start learning process
    learner.start_learning()
    
    try:
        # Process learning data
        sample_data = create_sample_data(15, 80)
        
        print("Processing experiences...")
        for i, data in enumerate(sample_data):
            result = learner.process_experience(
                pre_spikes=data["pre_spikes"],
                post_spikes=data["post_spikes"],
                rewards=data["rewards"],
                metadata=data["metadata"]
            )
            
            # Print progress
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/80 samples:")
                print(f"    Status: {result.get('status', 'unknown')}")
                print(f"    Updates: {result.get('updates', 0)}")
                print(f"    Quality: {result.get('quality_score', 0.0):.3f}")
                print(f"    Buffer size: {result.get('replay_buffer_size', 0)}")
                print()
                
            # Allow some time for periodic updates
            time.sleep(0.05)
            
    finally:
        learner.stop_learning()
    
    # Get final statistics
    print("Final Learning Statistics:")
    stats = learner.get_learning_statistics()
    
    print(f"  Learning stage: {stats['state']['stage']}")
    print(f"  Learning enabled: {stats['state']['learning_enabled']}")
    print(f"  Current quality: {stats['state']['current_quality']:.3f}")
    print(f"  Learning rate: {stats['state']['learning_rate']:.6f}")
    print(f"  Total updates: {stats['performance']['update_count']}")
    print(f"  Weight changes: {stats['performance']['total_weight_changes']:.6f}")
    print(f"  Circuit breaker: {stats['circuit_breaker']}")
    print()
    
    # Reset for next demo
    learner.reset()
    print("Learner reset for next demonstration")
    print()


def adaptation_demo():
    """Demonstrate learning rate adaptation and quality monitoring."""
    print("=" * 60)
    print("LEARNING RATE ADAPTATION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize adaptation components
    config = AdaptationConfig(
        strategy=AdaptationStrategy.ADAPTIVE,
        initial_rate=0.1,
        min_rate=0.001,
        max_rate=1.0,
        adaptation_threshold=0.01
    )
    
    adapter = LearningRateAdapter(config=config)
    monitor = QualityMonitor(
        window_size=30,
        target_score=0.7
    )
    
    print("Initialized learning rate adapter and quality monitor")
    print(f"  Initial rate: {adapter.current_rate:.6f}")
    print(f"  Adaptation strategy: {config.strategy.value}")
    print(f"  Target quality: {monitor.target_score}")
    print()
    
    # Simulate learning progression
    learning_progress = []
    
    for episode in range(50):
        # Simulate different performance phases
        if episode < 15:
            # Initial poor performance
            quality_score = 0.3 + 0.05 * episode + np.random.normal(0, 0.1)
            performance_history = [0.4 + 0.05 * episode] * 10
        elif episode < 30:
            # Improving performance
            quality_score = 0.7 + 0.02 * (episode - 15) + np.random.normal(0, 0.05)
            performance_history = [0.7 + 0.02 * (episode - 15)] * 10
        else:
            # Stable high performance
            quality_score = 0.9 + np.random.normal(0, 0.05)
            performance_history = [0.9] * 10
            
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        # Adapt learning rate
        new_rate = adapter.adapt_learning_rate(
            current_rate=adapter.current_rate,
            quality_score=quality_score,
            performance_history=performance_history
        )
        
        # Update quality monitor
        updates = max(1, int(quality_score * 10))
        errors = max(0, int((1.0 - quality_score) * 5))
        
        monitor.update_quality(
            updates=updates,
            total_attempts=10,
            error_count=errors
        )
        
        learning_progress.append({
            "episode": episode,
            "quality_score": quality_score,
            "learning_rate": new_rate,
            "updates": updates,
            "errors": errors
        })
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/50:")
            print(f"  Quality: {quality_score:.3f}")
            print(f"  Learning rate: {new_rate:.6f}")
            print(f"  Should adapt rate: {monitor.should_adapt_rate()}")
            print()
    
    # Final analysis
    print("Adaptation Results:")
    final_quality = learning_progress[-1]["quality_score"]
    final_rate = learning_progress[-1]["learning_rate"]
    rate_changes = sum(1 for i in range(1, len(learning_progress)) 
                      if learning_progress[i]["learning_rate"] != learning_progress[i-1]["learning_rate"])
    
    print(f"  Final quality: {final_quality:.3f}")
    print(f"  Final learning rate: {final_rate:.6f}")
    print(f"  Rate adaptations: {rate_changes}")
    print(f"  Quality assessment: {monitor.assess_quality()}")
    print()


def circuit_breaker_demo():
    """Demonstrate circuit breaker functionality."""
    print("=" * 60)
    print("CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 60)
    
    # Configure circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=0.3,
        success_threshold=0.8,
        quality_threshold=0.4,
        timeout=10.0,
        monitoring_window=20
    )
    
    breaker = LearningCircuitBreaker(config=config)
    
    print("Initialized circuit breaker")
    print(f"  Failure threshold: {config.failure_threshold}")
    print(f"  Timeout: {config.timeout}s")
    print(f"  Initial state: {breaker.state.value}")
    print()
    
    # Simulate normal operation
    print("Phase 1: Normal operation")
    for i in range(10):
        performance_metrics = {
            "updates": 8 + np.random.randint(0, 3),
            "total_attempts": 10,
            "weight_std": 0.1 + np.random.normal(0, 0.05),
            "weight_mean": 0.5 + np.random.normal(0, 0.1)
        }
        
        quality_metrics = {
            "overall_quality": 0.7 + np.random.normal(0, 0.1)
        }
        
        allowed = breaker.check_operation(performance_metrics, quality_metrics)
        print(f"  Step {i + 1}: Allowed={allowed}, State={breaker.state.value}")
    
    print()
    
    # Simulate failure conditions
    print("Phase 2: Failure conditions")
    for i in range(15):
        performance_metrics = {
            "updates": 1 + np.random.randint(0, 2),
            "total_attempts": 10,
            "weight_std": 0.8 + np.random.normal(0, 0.2),
            "weight_mean": 0.2 + np.random.normal(0, 0.1)
        }
        
        quality_metrics = {
            "overall_quality": 0.1 + np.random.normal(0, 0.05)
        }
        
        allowed = breaker.check_operation(performance_metrics, quality_metrics)
        
        if breaker.state.value == "open":
            print(f"  Step {i + 1}: CIRCUIT TRIPPED! Allowed={allowed}, State={breaker.state.value}")
            break
        else:
            print(f"  Step {i + 1}: Allowed={allowed}, State={breaker.state.value}")
    
    print()
    
    # Simulate recovery
    print("Phase 3: Recovery simulation")
    # Force timeout to allow half-open testing
    breaker.last_failure_time = time.time() - config.timeout - 1
    
    for i in range(10):
        # Improved performance
        performance_metrics = {
            "updates": 8 + np.random.randint(0, 2),
            "total_attempts": 10,
            "weight_std": 0.2 + np.random.normal(0, 0.1),
            "weight_mean": 0.5 + np.random.normal(0, 0.1)
        }
        
        quality_metrics = {
            "overall_quality": 0.8 + np.random.normal(0, 0.1)
        }
        
        allowed = breaker.check_operation(performance_metrics, quality_metrics)
        print(f"  Recovery step {i + 1}: Allowed={allowed}, State={breaker.state.value}")
        
        if breaker.state.value == "closed":
            print("  Circuit breaker recovered successfully!")
            break
    
    print()
    
    # Final statistics
    print("Circuit Breaker Statistics:")
    stats = breaker.get_statistics()
    print(f"  Final state: {stats['state']}")
    print(f"  Total trips: {stats['total_trips']}")
    print(f"  Total recoveries: {stats['total_recoveries']}")
    print(f"  Success count: {stats['success_count']}")
    print(f"  Failure count: {stats['failure_count']}")
    print()


def metrics_demo():
    """Demonstrate comprehensive metrics system."""
    print("=" * 60)
    print("LEARNING METRICS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize metrics system
    metrics = LearningMetrics(history_size=1000)
    
    print("Initialized learning metrics system")
    print(f"  History size: {metrics.history_size}")
    print(f"  Default alerts: {len(metrics.alert_manager.alerts)}")
    print()
    
    # Add custom alerts
    custom_alerts = [
        AlertConfig(
            metric_name="custom_quality",
            threshold=0.6,
            comparison="lt",
            level=AlertLevel.WARNING
        ),
        AlertConfig(
            metric_name="learning_rate",
            threshold=0.01,
            comparison="lt",
            level=AlertLevel.ERROR
        )
    ]
    
    for alert in custom_alerts:
        metrics.add_custom_alert(alert)
    
    print(f"Added {len(custom_alerts)} custom alerts")
    print()
    
    # Simulate learning with metrics collection
    print("Simulating learning with metrics collection...")
    
    for episode in range(40):
        # Simulate learning metrics
        learning_rate = max(0.001, 0.1 * (1.0 - episode * 0.02) + np.random.normal(0, 0.01))
        quality_score = min(0.95, 0.5 + episode * 0.02 + np.random.normal(0, 0.05))
        memory_usage = 0.4 + 0.4 * (episode / 40.0) + np.random.normal(0, 0.05)
        
        # Record metrics
        metrics.record_metric("learning_rate", learning_rate, MetricType.LEARNING)
        metrics.record_metric("custom_quality", quality_score, MetricType.QUALITY)
        metrics.record_metric("memory_usage", memory_usage, MetricType.RESOURCE)
        
        # Simulate some alerts
        if episode % 8 == 0:
            alerts = metrics.check_alerts()
            if alerts:
                print(f"  Episode {episode + 1}: {len(alerts)} alerts triggered")
                for alert in alerts:
                    print(f"    {alert['level']}: {alert['metric']} = {alert['value']:.3f} "
                          f"(threshold: {alert['threshold']})")
        
        # Print progress
        if (episode + 1) % 10 == 0:
            current_stats = metrics.get_metric_statistics("custom_quality", window_size=10)
            print(f"  Episode {episode + 1}/40: Quality mean = {current_stats['mean']:.3f}, "
                  f"Trend = {current_stats['trend']:.4f}")
    
    print()
    
    # Generate comprehensive report
    print("Comprehensive Performance Report:")
    report = metrics.get_performance_report()
    
    print(f"  Overall health: {report['overall_health']:.3f}")
    print(f"  Metrics tracked: {report['metrics_tracked']}")
    print(f"  Recent alerts: {report['alert_statistics']['recent_alerts']}")
    print(f"  Resource health: {report['resource_health']:.3f}")
    print()
    
    # Show quality assessment
    quality_assessment = report['quality_assessment']
    print(f"  Quality assessment:")
    print(f"    Overall score: {quality_assessment['overall_score']:.3f}")
    print(f"    Quality component: {quality_assessment['quality_component']:.3f}")
    print(f"    Stability component: {quality_assessment['stability_component']:.3f}")
    print()
    
    # Show convergence analysis
    convergence = report['convergence_analysis']
    print(f"  Convergence analysis:")
    print(f"    Status: {convergence['status']}")
    print(f"    Convergence ratio: {convergence['convergence_ratio']:.3f}")
    print(f"    Metrics assessed: {convergence['total_assessed']}")
    print()
    
    # Show specific metric statistics
    print("Key Metric Statistics:")
    for metric_name in ["learning_rate", "custom_quality", "memory_usage"]:
        if metric_name in [name for name in metrics.get_metric_names()]:
            stats = metrics.get_metric_statistics(metric_name, window_size=20)
            print(f"  {metric_name}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std: {stats['std']:.4f}")
            print(f"    Trend: {stats['trend']:.6f}")
            print(f"    Volatility: {stats['volatility']:.4f}")
    
    print()


def full_integration_demo():
    """Demonstrate full system integration."""
    print("=" * 60)
    print("FULL SYSTEM INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Configure integrated system
    config = LearningConfig(
        learning_mode=LearningMode.CONTINUOUS,
        batch_size=20,
        replay_buffer_size=200,
        min_replay_samples=50,
        update_interval=0.2,
        consolidation_interval=30.0,
        adaptation_interval=10.0,
        circuit_breaker_enabled=True,
        target_quality_score=0.8
    )
    
    # Initialize full system
    learner = OnlineLearner(
        n_neurons=25,
        config=config,
        stdp_type=STDPType.ADAPTIVE
    )
    
    # Initialize metrics
    metrics = LearningMetrics(history_size=2000)
    
    print("Initialized integrated learning system")
    print(f"  Neurons: 25")
    print(f"  Learning mode: {config.learning_mode.value}")
    print(f"  Target quality: {config.target_quality_score}")
    print(f"  Circuit breaker: {config.circuit_breaker_enabled}")
    print()
    
    # Set up callbacks
    def on_learning_update(state):
        if state.update_count % 20 == 0:
            metrics.record_metric("update_frequency", state.update_count, MetricType.PERFORMANCE)
            
    def on_quality_change(quality_score):
        metrics.record_metric("quality_changes", quality_score, MetricType.QUALITY)
        
    learner.set_callbacks(
        on_learning_update=on_learning_update,
        on_quality_change=on_quality_change
    )
    
    # Start learning
    learner.start_learning()
    
    try:
        # Create diverse learning data
        print("Processing integrated learning scenarios...")
        
        scenarios = [
            ("normal", 40),
            ("high_performance", 30),
            ("noisy", 30),
            ("recovery", 20)
        ]
        
        for scenario_name, num_samples in scenarios:
            print(f"\nScenario: {scenario_name} ({num_samples} samples)")
            
            for i in range(num_samples):
                # Create scenario-specific data
                if scenario_name == "high_performance":
                    # High success rate patterns
                    pre_spikes = np.random.random(25) < 0.2
                    post_spikes = pre_spikes.copy()
                    reward = np.array([1.0])
                elif scenario_name == "noisy":
                    # Noisy patterns
                    pre_spikes = np.random.random(25) < 0.5
                    post_spikes = np.random.random(25) < 0.3
                    reward = np.array([np.random.random()])
                elif scenario_name == "recovery":
                    # Recovery patterns
                    pre_spikes = np.random.random(25) < 0.15
                    post_spikes = np.random.random(25) < 0.15
                    reward = np.array([0.7])
                else:  # normal
                    pre_spikes = np.random.random(25) < 0.1
                    post_spikes = np.random.random(25) < 0.15
                    reward = np.array([0.8])
                
                # Process experience
                result = learner.process_experience(
                    pre_spikes=pre_spikes,
                    post_spikes=post_spikes,
                    rewards=reward,
                    metadata={"scenario": scenario_name, "sample": i}
                )
                
                # Record metrics
                if "updates" in result:
                    metrics.record_metric("successful_updates", result["updates"], MetricType.PERFORMANCE)
                if "quality_score" in result:
                    metrics.record_metric("instant_quality", result["quality_score"], MetricType.QUALITY)
                
                # Periodic updates
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{num_samples}")
                    current_stats = learner.get_learning_statistics()
                    print(f"    State: {current_stats['state']['stage']}")
                    print(f"    Quality: {current_stats['state']['current_quality']:.3f}")
                    print(f"    Learning rate: {current_stats['state']['learning_rate']:.6f}")
                    
                time.sleep(0.05)  # Allow processing time
    
    finally:
        learner.stop_learning()
    
    # Final comprehensive analysis
    print("\n" + "="*60)
    print("FINAL INTEGRATION RESULTS")
    print("="*60)
    
    # Learning statistics
    learning_stats = learner.get_learning_statistics()
    print("Learning System Statistics:")
    print(f"  Total updates: {learning_stats['performance']['update_count']}")
    print(f"  Weight changes: {learning_stats['performance']['total_weight_changes']:.6f}")
    print(f"  Final quality: {learning_stats['state']['current_quality']:.3f}")
    print(f"  Circuit breaker: {learning_stats['circuit_breaker']}")
    print()
    
    # Metrics analysis
    metrics_report = metrics.get_performance_report()
    print("Metrics Analysis:")
    print(f"  Overall health: {metrics_report['overall_health']:.3f}")
    print(f"  Metrics tracked: {metrics_report['metrics_tracked']}")
    print(f"  Active alerts: {metrics_report['alert_statistics']['active_alerts']}")
    print()
    
    # Quality assessment
    quality_assessment = metrics_report['quality_assessment']
    print("Final Quality Assessment:")
    print(f"  Overall score: {quality_assessment['overall_score']:.3f}")
    print(f"  Quality level: {quality_assessment['quality_component']:.3f}")
    print(f"  Stability level: {quality_assessment['stability_component']:.3f}")
    print()
    
    # Convergence status
    convergence = metrics_report['convergence_analysis']
    print("Convergence Analysis:")
    print(f"  Status: {convergence['status']}")
    print(f"  Convergence ratio: {convergence['convergence_ratio']:.3f}")
    print()
    
    print("✓ Integration demonstration completed successfully!")
    print()


def main():
    """Run all demonstrations."""
    print("AUTO-LEARNING SYSTEM FOR SPIKING NEURAL NETWORKS")
    print("=" * 70)
    print("This demonstration showcases the comprehensive auto-learning system")
    print("including STDP, online learning, replay buffer, adaptation, and monitoring.")
    print()
    
    # Setup logging
    setup_logging()
    
    try:
        # Run individual component demonstrations
        basic_stdp_demo()
        replay_buffer_demo()
        online_learner_demo()
        adaptation_demo()
        circuit_breaker_demo()
        metrics_demo()
        
        # Run full integration demo
        full_integration_demo()
        
        print("=" * 70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print("The auto-learning system demonstrates:")
        print("✓ STDP learning with multiple variants")
        print("✓ Experience replay with priority and novelty")
        print("✓ Online learning with adaptation")
        print("✓ Learning rate adaptation strategies")
        print("✓ Quality monitoring and assessment")
        print("✓ Circuit breaker protection")
        print("✓ Comprehensive metrics collection")
        print("✓ Full system integration")
        print()
        print("This system provides a robust foundation for online learning")
        print("in spiking neural networks with performance monitoring and")
        print("automatic adaptation capabilities.")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()