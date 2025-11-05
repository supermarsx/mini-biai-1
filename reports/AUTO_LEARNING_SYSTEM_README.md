# Auto-Learning System for Spiking Neural Networks

## Overview

This repository contains a comprehensive auto-learning system for spiking neural networks, implementing state-of-the-art online learning mechanisms with adaptive control, quality monitoring, and performance protection.

## 🚀 Key Features

### 1. STDP (Spike-Timing Dependent Plasticity) Implementation
- **Multiple STDP Variants**: Standard, Symmetric, Tri-phasic, Modified, and Adaptive STDP
- **Homeostatic Plasticity**: Automatic weight stabilization and activity regulation
- **Flexible Parameters**: Configurable learning rates, time constants, and weight limits
- **Thread-Safe Operations**: Efficient concurrent access with proper locking mechanisms

### 2. Online Learning Pipeline
- **Continuous Learning**: Real-time adaptation to new experiences
- **Multiple Learning Modes**: Continuous, Batch, Hybrid, and Frozen modes
- **Adaptive Parameters**: Dynamic adjustment of learning parameters based on performance
- **Integration Ready**: Seamless integration with existing neural network architectures

### 3. Experience Replay Buffer
- **Priority-Based Sampling**: Intelligent experience selection based on importance
- **Novelty Detection**: Automatic identification of new and unique experiences
- **Experience Categorization**: Support for different types of learning experiences
- **Efficient Storage**: Optimized memory management with automatic cleanup

### 4. Learning Rate Adaptation
- **Multiple Strategies**: Fixed, Decreasing, Exponential, Adaptive, Oscillating, and Momentum-based
- **Performance-Driven**: Automatic adjustment based on learning quality and stability
- **Quality Monitoring**: Real-time assessment of learning performance
- **Fallback Mechanisms**: Robust error handling and recovery procedures

### 5. Learning Quality Metrics and Monitoring
- **Comprehensive Metrics**: Performance, quality, resource usage, and convergence tracking
- **Alert System**: Configurable thresholds with callback support
- **Trend Analysis**: Advanced statistical analysis of learning progression
- **Export Capabilities**: JSON and CSV export for external analysis

### 6. Performance Protection (Circuit Breakers)
- **Three-State Design**: Closed, Open, and Half-Open states for robust operation
- **Failure Detection**: Automatic detection of learning failures and performance degradation
- **Recovery Mechanisms**: Gradual recovery testing with success rate monitoring
- **Customizable Thresholds**: Configurable failure and success thresholds

### 7. Error Handling and Fallbacks
- **Comprehensive Error Recovery**: Graceful handling of learning failures
- **Graceful Degradation**: System continues operating under adverse conditions
- **Detailed Logging**: Extensive logging for debugging and monitoring
- **State Persistence**: Automatic save/restore of learning state