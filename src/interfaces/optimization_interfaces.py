"""
Optimization interfaces for the brain-inspired AI system.

This module defines interfaces for advanced optimization algorithms including
federated learning, neural architecture search, quantization, and pruning.

The interfaces support distributed optimization, architecture search spaces,
multi-objective optimization, and hardware-aware model optimization.

Key Components:
    - FederatedClient: Federated learning client interface
    - ArchitectureSearchSpace: Neural architecture search space
    - OptimizationObjective: Multi-objective optimization
    - ModelCompression: Model compression parameters
    - FederatedMetrics: Federated learning metrics
    - SearchResult: Architecture search result

Architecture Benefits:
    - Distributed optimization support
    - Multi-objective optimization
    - Hardware-aware optimization
    - Architecture search automation
    - Privacy-preserving learning

Version: 1.0.0
Author: mini-biai-1 Team
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
from datetime import datetime
import numpy as np
import torch


class OptimizationAlgorithm(Enum):
    """Optimization algorithms."""
    FED_AVG = "fed_avg"
    FED_PROX = "fed_prox"
    FED_NOVA = "fed_nova"
    FED_BAL = "fed_bal"
    DARTS = "darts"
    PROGRESSIVE_DARTS = "progressive_darts"
    ENAS = "enas"
    PNAS = "pnas"
    GENETIC = "genetic"
    REINFORCE = "reinforce"


class CompressionType(Enum):
    """Model compression types."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    LOW_RANK = "low_rank"
    HASHING = "hashing"


class SearchStrategy(Enum):
    """Architecture search strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    DIFFERENTIABLE = "differentiable"


class HardwareTarget(Enum):
    """Hardware optimization targets."""
    CPU = "cpu"
    GPU = "gpu"
    MOBILE = "mobile"
    EDGE = "edge"
    FPGA = "fpga"
    ASIC = "asic"


@dataclass
class FederatedClient:
    """
    Federated learning client interface.
    
    Attributes:
        client_id: Unique client identifier
        data_size: Size of local dataset
        model_version: Current model version
        local_metrics: Local training metrics
        update_contribution: Contribution to global model
        privacy_budget: Differential privacy budget
        computational_resources: Available computational resources
        network_bandwidth: Network bandwidth availability
        participation_history: Participation history
    """
    client_id: str
    data_size: int
    model_version: str
    local_metrics: Dict[str, float]
    update_contribution: np.ndarray
    privacy_budget: float
    computational_resources: Dict[str, float]
    network_bandwidth: float
    participation_history: List[Dict[str, Any]]


@dataclass
class ArchitectureSearchSpace:
    """
    Neural architecture search space definition.
    
    Attributes:
        search_space_id: Unique search space identifier
        operations: Available operations
        connections: Possible connections
        constraints: Architectural constraints
        max_layers: Maximum number of layers
        target_metrics: Target performance metrics
        hardware_constraints: Hardware limitations
        search_strategy: Search strategy to use
        evaluation_budget: Evaluation budget
    """
    search_space_id: str
    operations: List[str]
    connections: List[Tuple[str, str]]
    constraints: Dict[str, Any]
    max_layers: int
    target_metrics: List[str]
    hardware_constraints: Dict[str, Any]
    search_strategy: SearchStrategy
    evaluation_budget: int


@dataclass
class OptimizationObjective:
    """
    Multi-objective optimization objective.
    
    Attributes:
        objective_id: Unique objective identifier
        primary_objective: Primary optimization objective
        secondary_objectives: Secondary optimization objectives
        weights: Objective weights
        constraints: Optimization constraints
        pareto_front: Pareto optimal solutions
        evaluation_function: Objective evaluation function
    """
    objective_id: str
    primary_objective: str
    secondary_objectives: List[str]
    weights: Dict[str, float]
    constraints: Dict[str, Any]
    pareto_front: List[Dict[str, float]]
    evaluation_function: Callable


@dataclass
class ModelCompression:
    """
    Model compression parameters.
    
    Attributes:
        compression_type: Type of compression
        compression_ratio: Target compression ratio
        accuracy_loss: Acceptable accuracy loss
        quantization_bits: Number of bits for quantization
        pruning_ratio: Pruning ratio for structured pruning
        sparsity_level: Target sparsity level
        distillation_student: Student model for distillation
        hardware_acceleration: Hardware acceleration preferences
    """
    compression_type: CompressionType
    compression_ratio: float
    accuracy_loss: float
    quantization_bits: int
    pruning_ratio: float
    sparsity_level: float
    distillation_student: Optional[str] = None
    hardware_acceleration: Optional[HardwareTarget] = None


@dataclass
class FederatedMetrics:
    """
    Federated learning metrics.
    
    Attributes:
        round_number: Current training round
        global_accuracy: Global model accuracy
        local_accuracies: Per-client accuracies
        convergence_rate: Model convergence rate
        communication_cost: Total communication cost
        client_participation: Client participation rate
        privacy_epsilon: Achieved privacy level
        fairness_metrics: Fairness across clients
    """
    round_number: int
    global_accuracy: float
    local_accuracies: Dict[str, float]
    convergence_rate: float
    communication_cost: float
    client_participation: float
    privacy_epsilon: float
    fairness_metrics: Dict[str, float]


@dataclass
class SearchResult:
    """
    Architecture search result.
    
    Attributes:
        result_id: Unique result identifier
        architecture: Discovered architecture
        performance_metrics: Performance metrics
        evaluation_time: Time taken for evaluation
        hardware_profile: Hardware utilization profile
        resource_usage: Computational resource usage
        deployment_ready: Deployment readiness score
        search_metadata: Search process metadata
    """
    result_id: str
    architecture: Dict[str, Any]
    performance_metrics: Dict[str, float]
    evaluation_time: float
    hardware_profile: Dict[str, float]
    resource_usage: Dict[str, float]
    deployment_ready: float
    search_metadata: Dict[str, Any]


@dataclass
class DistributedOptimizer:
    """
    Distributed optimization configuration.
    
    Attributes:
        optimizer_type: Type of distributed optimizer
        aggregation_method: Model aggregation method
        client_selection: Client selection strategy
        privacy_mechanism: Privacy preservation mechanism
        communication_rounds: Number of communication rounds
        local_epochs: Local training epochs per round
        learning_rate: Learning rate for local training
        momentum: Momentum for optimizer
    """
    optimizer_type: OptimizationAlgorithm
    aggregation_method: str
    client_selection: str
    privacy_mechanism: str
    communication_rounds: int
    local_epochs: int
    learning_rate: float
    momentum: float


@dataclass
class NASConfig:
    """
    Neural architecture search configuration.
    
    Attributes:
        search_algorithm: Search algorithm
        search_space: Architecture search space
        evaluation_method: Architecture evaluation method
        search_budget: Total search budget
        parallel_evaluation: Parallel evaluation settings
        early_stopping: Early stopping criteria
        resource_constraints: Computational resource limits
        performance_proxy: Performance estimation proxy
    """
    search_algorithm: OptimizationAlgorithm
    search_space: ArchitectureSearchSpace
    evaluation_method: str
    search_budget: int
    parallel_evaluation: Dict[str, Any]
    early_stopping: Dict[str, Any]
    resource_constraints: Dict[str, float]
    performance_proxy: Optional[str] = None


@dataclass
class HardwareProfile:
    """
    Hardware utilization profile.
    
    Attributes:
        target_hardware: Target hardware platform
        latency_requirement: Latency requirement
        memory_constraint: Memory usage constraint
        power_constraint: Power consumption constraint
        throughput_target: Throughput target
        utilization_efficiency: Target utilization efficiency
        acceleration_support: Hardware acceleration support
    """
    target_hardware: HardwareTarget
    latency_requirement: float
    memory_constraint: float
    power_constraint: float
    throughput_target: float
    utilization_efficiency: float
    acceleration_support: List[str]


@dataclass
class OptimizationResult:
    """
    Optimization result interface.
    
    Attributes:
        result_id: Unique result identifier
        optimized_model: Optimized model configuration
        performance_improvement: Performance improvement metrics
        resource_savings: Resource usage savings
        deployment_config: Deployment configuration
        optimization_metadata: Optimization process metadata
        quality_assurance: Quality assurance metrics
    """
    result_id: str
    optimized_model: Dict[str, Any]
    performance_improvement: Dict[str, float]
    resource_savings: Dict[str, float]
    deployment_config: Dict[str, Any]
    optimization_metadata: Dict[str, Any]
    quality_assurance: Dict[str, float]


# Export all interfaces
__all__ = [
    'OptimizationAlgorithm',
    'CompressionType',
    'SearchStrategy',
    'HardwareTarget',
    'FederatedClient',
    'ArchitectureSearchSpace',
    'OptimizationObjective',
    'ModelCompression',
    'FederatedMetrics',
    'SearchResult',
    'DistributedOptimizer',
    'NASConfig',
    'HardwareProfile',
    'OptimizationResult'
]