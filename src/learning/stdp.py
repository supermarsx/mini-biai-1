"""
STDP (Spike-Timing Dependent Plasticity) Implementation.

This module provides comprehensive STDP learning mechanisms for spiking neural networks,
including various STDP variants, weight updates, and plasticity control.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque
import time


class STDPType(Enum):
    """Types of STDP learning rules."""
    STANDARD = "standard"           # Classical STDP
    SYMMETRIC = "symmetric"         # Symmetric STDP
    TRI_PHASE = "tri_phase"         # Tri-phasic STDP
    MODIFIED = "modified"           # Modified STDP with homeostatic terms
    ADAPTIVE = "adaptive"           # Adaptive STDP with learning rate modulation


@dataclass
class STDPParameters:
    """Parameters for STDP learning rule."""
    # Core STDP parameters
    a_plus: float = 0.005           # Potentiation amplitude (positive dt)
    a_minus: float = 0.005          # Depression amplitude (negative dt)
    tau_plus: float = 20.0          # Time constant for positive dt (ms)
    tau_minus: float = 20.0         # Time constant for negative dt (ms)
    
    # Weight limits
    w_max: float = 1.0              # Maximum synaptic weight
    w_min: float = 0.0              # Minimum synaptic weight
    
    # Homeostatic mechanisms
    homeostatic_strength: float = 0.001  # Homeostatic regularization strength
    target_activity: float = 0.1         # Target firing rate for homeostasis
    
    # Adaptive parameters
    adaptation_rate: float = 0.001       # Rate for learning rate adaptation
    decay_rate: float = 0.95             # Exponential decay for traces
    
    # Learning control
    enable_learning: bool = True
    learning_decay: float = 0.98         # Decay factor for learning trace


class TraceManager:
    """Manages pre-synaptic and post-synaptic traces for STDP."""
    
    def __init__(self, n_neurons: int, dt: float = 1.0):
        self.n_neurons = n_neurons
        self.dt = dt
        self.pre_traces = np.zeros(n_neurons, dtype=np.float32)
        self.post_traces = np.zeros(n_neurons, dtype=np.float32)
        self._lock = threading.RLock()
        
    def update_traces(self, pre_spikes: np.ndarray, post_spikes: np.ndarray):
        """Update pre and post-synaptic traces."""
        with self._lock:
            # Exponential decay
            decay_factor = np.exp(-self.dt / 20.0)
            self.pre_traces *= decay_factor
            self.post_traces *= decay_factor
            
            # Add spike contributions
            self.pre_traces[pre_spikes > 0] += 1.0
            self.post_traces[post_spikes > 0] += 1.0
            
    def get_pre_trace(self, neuron_idx: int) -> float:
        """Get pre-synaptic trace for specific neuron."""
        with self._lock:
            return self.pre_traces[neuron_idx]
            
    def get_post_trace(self, neuron_idx: int) -> float:
        """Get post-synaptic trace for specific neuron."""
        with self._lock:
            return self.post_traces[neuron_idx]


class STDPManager:
    """
    Comprehensive STDP (Spike-Timing Dependent Plasticity) manager.
    
    Provides multiple STDP variants, weight updates, and plasticity control
    mechanisms for spiking neural networks.
    """
    
    def __init__(self, 
                 n_neurons: int,
                 connection_matrix: Optional[np.ndarray] = None,
                 stdp_type: STDPType = STDPType.STANDARD,
                 parameters: Optional[STDPParameters] = None,
                 dt: float = 1.0):
        """
        Initialize STDP manager.
        
        Args:
            n_neurons: Number of neurons in the network
            connection_matrix: Binary matrix indicating which neurons are connected
            stdp_type: Type of STDP rule to use
            parameters: STDP parameters
            dt: Time step (ms)
        """
        self.n_neurons = n_neurons
        self.dt = dt
        self.stdp_type = stdp_type
        self.parameters = parameters or STDPParameters()
        
        # Initialize connection matrix
        if connection_matrix is None:
            self.connection_matrix = self._create_random_connections()
        else:
            self.connection_matrix = connection_matrix.astype(bool)
            
        # Initialize synaptic weights
        self.weights = self._initialize_weights()
        
        # Initialize traces
        self.trace_manager = TraceManager(n_neurons, dt)
        
        # Learning state
        self.learning_enabled = self.parameters.enable_learning
        self.last_update_time = time.time()
        
        # Statistics
        self.update_history = deque(maxlen=1000)
        self.weight_history = deque(maxlen=100)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _create_random_connections(self, connection_prob: float = 0.1) -> np.ndarray:
        """Create random connection matrix."""
        return (np.random.random((self.n_neurons, self.n_neurons)) < connection_prob).astype(bool)
        
    def _initialize_weights(self) -> np.ndarray:
        """Initialize synaptic weights."""
        weights = np.zeros((self.n_neurons, self.n_neurons), dtype=np.float32)
        
        for i in range(self.n_neurons):
            for j in range(self.n_neurons):
                if self.connection_matrix[i, j]:
                    # Initialize with small random weights
                    weights[i, j] = np.random.uniform(0.1, 0.5)
                    
        return weights
        
    def update_weights(self, 
                      pre_spikes: np.ndarray, 
                      post_spikes: np.ndarray,
                      activity_history: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Update synaptic weights using STDP rule.
        
        Args:
            pre_spikes: Binary array indicating pre-synaptic spikes
            post_spikes: Binary array indicating post-synaptic spikes
            activity_history: Optional activity history for homeostasis
            
        Returns:
            Dictionary with update statistics
        """
        if not self.learning_enabled:
            return {"updates": 0, "weight_changes": 0.0, "reason": "learning_disabled"}
            
        start_time = time.time()
        
        try:
            # Update traces
            self.trace_manager.update_traces(pre_spikes, post_spikes)
            
            # Calculate weight updates
            updates = 0
            total_weight_change = 0.0
            
            for pre_idx in np.where(pre_spikes)[0]:
                for post_idx in range(self.n_neurons):
                    if self.connection_matrix[pre_idx, post_idx]:
                        delta_w = self._calculate_stdp_delta(pre_idx, post_idx)
                        
                        if delta_w != 0:
                            old_weight = self.weights[pre_idx, post_idx]
                            new_weight = np.clip(
                                old_weight + delta_w,
                                self.parameters.w_min,
                                self.parameters.w_max
                            )
                            
                            if new_weight != old_weight:
                                self.weights[pre_idx, post_idx] = new_weight
                                updates += 1
                                total_weight_change += abs(new_weight - old_weight)
                                
            # Apply homeostatic mechanism
            if activity_history and self.parameters.homeostatic_strength > 0:
                self._apply_homeostasis(activity_history)
                
            # Update statistics
            update_time = time.time() - start_time
            self.update_history.append(update_time)
            self.weight_history.append(total_weight_change)
            
            stats = {
                "updates": updates,
                "weight_changes": total_weight_change,
                "avg_update_time": np.mean(self.update_history),
                "learning_enabled": self.learning_enabled,
                "stdp_type": self.stdp_type.value,
                "update_time": update_time
            }
            
            self.logger.debug(f"STDP update: {updates} connections modified, "
                            f"total change: {total_weight_change:.6f}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error in STDP update: {e}")
            return {"error": str(e), "updates": 0, "weight_changes": 0.0}
            
    def _calculate_stdp_delta(self, pre_idx: int, post_idx: int) -> float:
        """Calculate STDP weight change for a specific connection."""
        if not self.connection_matrix[pre_idx, post_idx]:
            return 0.0
            
        pre_trace = self.trace_manager.get_pre_trace(pre_idx)
        post_trace = self.trace_manager.get_post_trace(post_idx)
        
        delta_w = 0.0
        
        if self.stdp_type == STDPType.STANDARD:
            delta_w = self._standard_stdp(pre_trace, post_trace)
        elif self.stdp_type == STDPType.SYMMETRIC:
            delta_w = self._symmetric_stdp(pre_trace, post_trace)
        elif self.stdp_type == STDPType.TRI_PHASE:
            delta_w = self._tri_phasic_stdp(pre_trace, post_trace)
        elif self.stdp_type == STDPType.MODIFIED:
            delta_w = self._modified_stdp(pre_trace, post_trace)
        elif self.stdp_type == STDPType.ADAPTIVE:
            delta_w = self._adaptive_stdp(pre_trace, post_trace)
            
        return delta_w
        
    def _standard_stdp(self, pre_trace: float, post_trace: float) -> float:
        """Standard STDP rule."""
        if pre_trace > 0 and post_trace > 0:
            # Both traces active - update based on timing difference
            return (self.parameters.a_plus * pre_trace * post_trace - 
                   self.parameters.a_minus * pre_trace * post_trace)
        return 0.0
        
    def _symmetric_stdp(self, pre_trace: float, post_trace: float) -> float:
        """Symmetric STDP rule."""
        if pre_trace > 0 and post_trace > 0:
            return (self.parameters.a_plus * pre_trace * post_trace - 
                   self.parameters.a_minus * pre_trace * post_trace)
        return 0.0
        
    def _tri_phasic_stdp(self, pre_trace: float, post_trace: float) -> float:
        """Tri-phasic STDP with inhibitory phase."""
        # Simplified implementation
        if pre_trace > 0 and post_trace > 0:
            return (self.parameters.a_plus * pre_trace * post_trace * 0.5)
        elif pre_trace > 0:
            return -self.parameters.a_minus * pre_trace * 0.3
        return 0.0
        
    def _modified_stdp(self, pre_trace: float, post_trace: float) -> float:
        """Modified STDP with additional terms."""
        base_change = self._standard_stdp(pre_trace, post_trace)
        
        # Add homeostatic term
        current_weight = self.weights[np.where(self.connection_matrix)[0][0], 
                                     np.where(self.connection_matrix)[1][0]]  # Simplified
        homeostasis = -self.parameters.homeostatic_strength * (current_weight - 0.5)
        
        return base_change + homeostasis
        
    def _adaptive_stdp(self, pre_trace: float, post_trace: float) -> float:
        """Adaptive STDP with learning rate modulation."""
        base_change = self._standard_stdp(pre_trace, post_trace)
        
        # Modulate by recent activity
        recent_activity = np.mean([len(self.update_history) / max(1, len(self.weight_history)), 0.1])
        adaptive_factor = 1.0 / (1.0 + recent_activity)
        
        return base_change * adaptive_factor
        
    def _apply_homeostasis(self, activity_history: Dict[str, float]):
        """Apply homeostatic plasticity mechanism."""
        if not activity_history:
            return
            
        target = self.parameters.target_activity
        
        for pre_idx in range(self.n_neurons):
            for post_idx in range(self.n_neurons):
                if self.connection_matrix[pre_idx, post_idx]:
                    # Calculate activity-dependent scaling
                    current_weight = self.weights[pre_idx, post_idx]
                    
                    # Homeostatic scaling factor
                    scaling = (1.0 - self.parameters.homeostatic_strength * 
                             (activity_history.get('pre_activity', 0) - target) * 
                             (activity_history.get('post_activity', 0) - target))
                    
                    # Apply scaling
                    new_weight = current_weight * scaling
                    self.weights[pre_idx, post_idx] = np.clip(
                        new_weight, self.parameters.w_min, self.parameters.w_max
                    )
                    
    def set_learning_enabled(self, enabled: bool):
        """Enable or disable learning."""
        self.learning_enabled = enabled
        self.logger.info(f"Learning {'enabled' if enabled else 'disabled'}")
        
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get current weight statistics."""
        connected_weights = self.weights[self.connection_matrix]
        
        return {
            "mean_weight": np.mean(connected_weights),
            "std_weight": np.std(connected_weights),
            "max_weight": np.max(connected_weights),
            "min_weight": np.min(connected_weights),
            "sparsity": np.mean(self.connection_matrix == False),
            "num_connections": np.sum(self.connection_matrix)
        }
        
    def get_recent_update_stats(self) -> Dict[str, float]:
        """Get recent update statistics."""
        if not self.update_history:
            return {"avg_update_time": 0.0, "recent_activity": 0.0}
            
        return {
            "avg_update_time": np.mean(self.update_history),
            "std_update_time": np.std(self.update_history),
            "recent_updates": len(self.update_history),
            "avg_weight_change": np.mean(self.weight_history) if self.weight_history else 0.0
        }
        
    def reset(self):
        """Reset STDP state."""
        self.weights = self._initialize_weights()
        self.trace_manager = TraceManager(self.n_neurons, self.dt)
        self.update_history.clear()
        self.weight_history.clear()
        self.logger.info("STDP state reset")
        
    def save_state(self, filepath: str):
        """Save current state to file."""
        state = {
            "weights": self.weights,
            "connection_matrix": self.connection_matrix,
            "stdp_type": self.stdp_type.value,
            "parameters": self.parameters,
            "learning_enabled": self.learning_enabled,
            "statistics": {
                "weight_stats": self.get_weight_statistics(),
                "update_stats": self.get_recent_update_stats()
            }
        }
        
        np.savez_compressed(filepath, **state)
        self.logger.info(f"STDP state saved to {filepath}")
        
    def load_state(self, filepath: str):
        """Load state from file."""
        try:
            data = np.load(filepath, allow_pickle=True)
            
            self.weights = data["weights"]
            self.connection_matrix = data["connection_matrix"]
            self.stdp_type = STDPType(data["stdp_type"].item())
            
            # Load parameters if available
            if "parameters" in data:
                self.parameters = data["parameters"].item()
                
            self.learning_enabled = bool(data["learning_enabled"])
            
            self.logger.info(f"STDP state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load STDP state: {e}")
            raise