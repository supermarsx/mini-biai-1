"""
Coordinator Module for mini-biai-1

This module provides comprehensive coordination and orchestration functionality
for the mini-biai-1 framework, implementing spiking neural network
components for intelligent routing and processing.

Main Components:
    - SpikingRouter: Neural network-based routing system
    - LIFNeuron: Leaky Integrate and Fire neuron implementation
    - LinearEncoder: Feature encoding for spiking compatibility
    - RoutingHead: Decision-making layer for expert selection
    - HardwareCompatibilityChecker: Device optimization and fallback handling

Key Features:
    - Biologically-inspired spiking neural networks
    - Adaptive threshold mechanisms for stable spiking
    - Hardware-aware optimization (CPU/CUDA/MPS)
    - Real-time performance monitoring
    - Comprehensive error handling and fallbacks

Architecture:
    The coordinator uses a three-layer architecture:
    1. LinearEncoder: Converts input features to spike-compatible format
    2. LIFNeuron: Generates spikes based on membrane potential dynamics
    3. RoutingHead: Makes routing decisions based on spiking patterns

Example Usage:
    >>> from src.coordinator import SpikingRouter, LIFParameters
    >>> import torch
    >>> 
    >>> # Create router with custom parameters
    >>> config = LIFParameters(tau=15.0, v_th=1.2, spike_target_rate=0.08)
    >>> router = SpikingRouter(input_dim=512, num_routes=8, lif_params=config)
    >>> 
    >>> # Process input features
    >>> input_features = torch.randn(32, 512)  # Batch of 32, 512-dim features
    >>> result = router(input_features, return_detailed=True)
    >>> 
    >>> print(f"Routing decisions: {result['routing_decision']}")
    >>> print(f"Spike rate: {result['spike_rate']:.3f}")
    >>> print(f"Confidence: {result['confidence'].mean():.3f}")

Dependencies:
    - torch: Neural network computations
    - numpy: Numerical operations
    - faiss: Vector similarity search (optional)

Version: 1.0.0
Author: mini-biai-1 Team
License: MIT
"""

from .spiking_router import (
    SpikingRouter,
    LIFNeuron,
    LinearEncoder,
    RoutingHead,
    LIFParameters,
    HardwareCompatibilityChecker
)

__all__ = [
    "SpikingRouter",
    "LIFNeuron", 
    "LinearEncoder",
    "RoutingHead",
    "LIFParameters",
    "HardwareCompatibilityChecker"
]