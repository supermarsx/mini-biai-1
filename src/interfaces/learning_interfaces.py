"""
Learning interfaces for the brain-inspired AI system.

This module defines interfaces for reinforcement learning algorithms and
adaptive learning mechanisms integrated with spiking neural networks.

The interfaces support value-based methods, policy-based methods, actor-critic
algorithms, and biological learning mechanisms like STDP.

Key Components:
    - RLEnvironment: Reinforcement learning environment
    - RLAgent: Learning agent interface
    - ExperienceReplay: Experience replay buffer
    - PolicyGradient: Policy gradient operations
    - ValueFunction: Value function approximation
    - BiologicalLearning: STDP and biological learning

Architecture Benefits:
    - SNN-integrated learning
    - Multiple algorithm support
    - Experience replay
    - Biological plausibility
    - Adaptive learning rates

Version: 1.0.0
Author: mini-biai-1 Team
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from enum import Enum
from datetime import datetime
import numpy as np
import torch


class RLAlgorithm(Enum):
    """Reinforcement learning algorithms."""
    DQN = "dqn"
    DDQN = "ddqn"
    DUELING_DQN = "dueling_dqn"
    PPO = "ppo"
    A3C = "a3c"
    A2C = "a2c"
    SAC = "sac"
    TD3 = "td3"
    DDPG = "ddpg"
    REINFORCE = "reinforce"
    ACER = "acer"


class ActionSpace(Enum):
    """Action space types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    MULTI_DISCRETE = "multi_discrete"
    MULTI_BINARY = "multi_binary"


class ObservationSpace(Enum):
    """Observation space types."""
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    IMAGE = "image"
    HISTOGRAM = "histogram"


class LearningRate(Enum):
    """Learning rate scheduling."""
    CONSTANT = "constant"
    LINEAR_DECAY = "linear_decay"
    EXPONENTIAL_DECAY = "exponential_decay"
    STEP_DECAY = "step_decay"
    ADAPTIVE = "adaptive"


@dataclass
class RLEnvironment:
    """
    Reinforcement learning environment interface.
    
    Attributes:
        env_id: Unique environment identifier
        action_space: Action space definition
        observation_space: Observation space definition
        max_steps: Maximum steps per episode
        reward_range: Reward value range
        metadata: Environment metadata
        reset_function: Environment reset function
        step_function: Environment step function
        render_function: Environment rendering function
    """
    env_id: str
    action_space: Dict[str, Any]
    observation_space: Dict[str, Any]
    max_steps: int
    reward_range: Tuple[float, float]
    metadata: Dict[str, Any]
    reset_function: Optional[Callable] = None
    step_function: Optional[Callable] = None
    render_function: Optional[Callable] = None


@dataclass
class RLState:
    """
    Reinforcement learning state representation.
    
    Attributes:
        state_id: Unique state identifier
        observation: Current observation
        action_mask: Available actions mask
        reward: Received reward
        done: Episode termination flag
        info: Additional state information
        timestamp: State timestamp
        episode_id: Episode identifier
        step_count: Step count in episode
    """
    state_id: str
    observation: np.ndarray
    action_mask: Optional[np.ndarray] = None
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = None
    timestamp: datetime = None
    episode_id: Optional[str] = None
    step_count: int = 0


@dataclass
class Experience:
    """
    Single experience tuple.
    
    Attributes:
        experience_id: Unique experience identifier
        state: Previous state
        action: Action taken
        reward: Reward received
        next_state: Resulting state
        done: Episode termination
        value: State value (for actor-critic)
        log_prob: Action log probability
        advantage: Advantage estimate
        importance_weight: Importance sampling weight
    """
    experience_id: str
    state: RLState
    action: Union[int, float, np.ndarray]
    reward: float
    next_state: RLState
    done: bool
    value: Optional[float] = None
    log_prob: Optional[float] = None
    advantage: Optional[float] = None
    importance_weight: float = 1.0


@dataclass
class ExperienceReplay:
    """
    Experience replay buffer.
    
    Attributes:
        buffer_id: Unique buffer identifier
        capacity: Buffer capacity
        current_size: Current buffer size
        experiences: Stored experiences
        sampling_strategy: Experience sampling strategy
        prioritization: Prioritization parameters
        update_frequency: Buffer update frequency
        learning_stats: Learning statistics
    """
    buffer_id: str
    capacity: int
    current_size: int
    experiences: List[Experience]
    sampling_strategy: str
    prioritization: Dict[str, Any]
    update_frequency: int
    learning_stats: Dict[str, float]


@dataclass
class PolicyNetwork:
    """
    Policy network configuration.
    
    Attributes:
        network_id: Unique network identifier
        architecture: Network architecture
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function
        initialization: Weight initialization
        parameters: Network parameters
        gradient_info: Gradient information
    """
    network_id: str
    architecture: str
    input_dim: int
    hidden_dims: List[int]
    output_dim: int
    activation: str
    initialization: str
    parameters: Dict[str, torch.Tensor]
    gradient_info: Dict[str, np.ndarray]


@dataclass
class ValueFunction:
    """
    Value function approximation.
    
    Attributes:
        value_id: Unique value function identifier
        function_type: Type of value function
        network: Value network configuration
        target_network: Target network for stability
        estimation_method: Value estimation method
        temporal_discount: Temporal discount factor
        bootstrap_steps: Bootstrap steps
        regularization: Regularization parameters
    """
    value_id: str
    function_type: str
    network: PolicyNetwork
    target_network: Optional[PolicyNetwork] = None
    estimation_method: str = "monte_carlo"
    temporal_discount: float = 0.99
    bootstrap_steps: int = 1
    regularization: Dict[str, float] = None


@dataclass
class PolicyGradient:
    """
    Policy gradient operation.
    
    Attributes:
        gradient_id: Unique gradient identifier
        algorithm: Policy gradient algorithm
        learning_rate: Learning rate
        clip_param: PPO clip parameter
        entropy_coef: Entropy coefficient
        value_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm
        advantage_estimator: Advantage estimation method
        baseline: Baseline function
    """
    gradient_id: str
    algorithm: RLAlgorithm
    learning_rate: float
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    advantage_estimator: str = "gae"
    baseline: Optional[ValueFunction] = None


@dataclass
class BiologicalLearning:
    """
    Biological learning mechanisms.
    
    Attributes:
        learning_id: Unique learning identifier
        stdp_params: STDP parameters
        spike_times: Spike timing information
        synaptic_weights: Synaptic weight changes
        plasticity_rules: Synaptic plasticity rules
        homeostatic_mechanisms: Homeostatic regulation
        neuromodulation: Neuromodulatory signals
        spike_history: Spike history for learning
    """
    learning_id: str
    stdp_params: Dict[str, float]
    spike_times: List[Tuple[int, float]]
    synaptic_weights: np.ndarray
    plasticity_rules: Dict[str, Callable]
    homeostatic_mechanisms: Dict[str, Any]
    neuromodulation: Dict[str, float]
    spike_history: List[Dict[str, Any]]


@dataclass
class RLAgent:
    """
    Reinforcement learning agent.
    
    Attributes:
        agent_id: Unique agent identifier
        algorithm: Learning algorithm
        environment: Associated environment
        policy_network: Policy network
        value_network: Value network
        experience_replay: Experience replay buffer
        learning_config: Learning configuration
        performance_metrics: Performance tracking
        exploration_strategy: Exploration parameters
    """
    agent_id: str
    algorithm: RLAlgorithm
    environment: RLEnvironment
    policy_network: PolicyNetwork
    value_network: Optional[ValueNetwork] = None
    experience_replay: Optional[ExperienceReplay] = None
    learning_config: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    exploration_strategy: Dict[str, Any] = None


@dataclass
class ValueNetwork:
    """
    Value network for actor-critic methods.
    
    Attributes:
        network_id: Unique network identifier
        architecture: Network architecture
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension (usually 1)
        activation: Activation function
        parameters: Network parameters
        target_parameters: Target network parameters
        update_frequency: Target network update frequency
    """
    network_id: str
    architecture: str
    input_dim: int
    hidden_dims: List[int]
    output_dim: int = 1
    activation: str = "relu"
    parameters: Dict[str, torch.Tensor] = None
    target_parameters: Dict[str, torch.Tensor] = None
    update_frequency: int = 1000


@dataclass
class RLMetrics:
    """
    Reinforcement learning metrics.
    
    Attributes:
        episode_rewards: Episode reward history
        episode_lengths: Episode length history
        success_rate: Task success rate
        exploration_rate: Exploration-exploitation balance
        learning_progress: Learning progress indicators
        stability_metrics: Training stability measures
        convergence_analysis: Convergence analysis
        policy_entropy: Policy entropy measurements
    """
    episode_rewards: List[float]
    episode_lengths: List[int]
    success_rate: float
    exploration_rate: float
    learning_progress: Dict[str, float]
    stability_metrics: Dict[str, float]
    convergence_analysis: Dict[str, Any]
    policy_entropy: float


@dataclass
class LearningConfig:
    """
    Learning configuration parameters.
    
    Attributes:
        batch_size: Training batch size
        learning_rates: Learning rates for components
        exploration_params: Exploration parameters
        replay_buffer_size: Experience replay size
        target_update_freq: Target network update frequency
        regularization: Regularization parameters
        scheduler_config: Learning rate scheduler
        validation_split: Validation data split
    """
    batch_size: int
    learning_rates: Dict[str, float]
    exploration_params: Dict[str, Any]
    replay_buffer_size: int
    target_update_freq: int
    regularization: Dict[str, float]
    scheduler_config: Dict[str, Any]
    validation_split: float


@dataclass
class TrainingResult:
    """
    Training result summary.
    
    Attributes:
        result_id: Unique result identifier
        final_performance: Final performance metrics
        learning_curve: Learning progress curve
        convergence_metrics: Convergence indicators
        hyperparameter_tuning: Hyperparameter optimization results
        model_checkpoints: Saved model checkpoints
        evaluation_results: Evaluation on test environment
        training_metadata: Training process metadata
    """
    result_id: str
    final_performance: Dict[str, float]
    learning_curve: List[float]
    convergence_metrics: Dict[str, float]
    hyperparameter_tuning: Dict[str, Any]
    model_checkpoints: List[str]
    evaluation_results: Dict[str, float]
    training_metadata: Dict[str, Any]


# Export all interfaces
__all__ = [
    'RLAlgorithm',
    'ActionSpace',
    'ObservationSpace',
    'LearningRate',
    'RLEnvironment',
    'RLState',
    'Experience',
    'ExperienceReplay',
    'PolicyNetwork',
    'ValueFunction',
    'PolicyGradient',
    'BiologicalLearning',
    'RLAgent',
    'ValueNetwork',
    'RLMetrics',
    'LearningConfig',
    'TrainingResult'
]