"""
Adaptive Optimization Framework
Advanced optimizers with meta-learning capabilities.

This module provides adaptive learning rate schedules, Bayesian hyperparameter
optimization, and meta-learning enhanced optimizers for efficient training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import copy
from scipy.optimize import minimize
from scipy.stats import norm


class LearningRateScheduler:
    """
    Meta-learning enhanced learning rate scheduler.
    
    Adaptively adjusts learning rates based on training dynamics
    and meta-learning feedback.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_type: str = 'cosine',  # 'cosine', 'step', 'exponential', 'adaptive'
        base_lr: float = 0.001,
        **kwargs
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler
            base_lr: Base learning rate
            **kwargs: Scheduler-specific parameters
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.base_lr = base_lr
        self.step_count = 0
        
        # Initialize base scheduler
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                optimizer, T_max=kwargs.get('T_max', 100), 
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                optimizer, step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'exponential':
            self.scheduler = ExponentialLR(
                optimizer, gamma=kwargs.get('gamma', 0.95)
            )
        elif scheduler_type == 'adaptive':
            self._init_adaptive_scheduler(**kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
        # Meta-learning state
        self.loss_history = deque(maxlen=100)
        self.gradient_norms = deque(maxlen=100)
        self.learning_rate_history = []
        self.meta_state = {
            'best_loss': float('inf'),
            'plateau_count': 0,
            'improvement_count': 0,
            'last_lr': base_lr
        }
        
    def _init_adaptive_scheduler(self, **kwargs):
        """Initialize adaptive scheduler components."""
        self.patience = kwargs.get('patience', 10)
        self.factor = kwargs.get('factor', 0.5)
        self.min_lr = kwargs.get('min_lr', 1e-6)
        self.max_lr = kwargs.get('max_lr', 0.1)
        self.gradient_threshold = kwargs.get('gradient_threshold', 1e-5)
        self.loss_threshold = kwargs.get('loss_threshold', 1e-4)
        
    def step(self, loss: Optional[float] = None, gradient_norm: Optional[float] = None):
        """
        Update learning rate.
        
        Args:
            loss: Current training loss
            gradient_norm: Current gradient norm
        """
        self.step_count += 1
        
        # Update histories
        if loss is not None:
            self.loss_history.append(loss)
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
            
        # Update learning rate
        if self.scheduler_type == 'adaptive':
            self._adaptive_step(loss, gradient_norm)
        else:
            self.scheduler.step()
            
        # Record current learning rate
        current_lr = self.get_last_lr()
        self.learning_rate_history.append(current_lr)
        self.meta_state['last_lr'] = current_lr
        
    def _adaptive_step(self, loss: Optional[float], gradient_norm: Optional[float]):
        """Adaptive learning rate adjustment."""
        if not self.loss_history:
            return
            
        current_loss = loss if loss is not None else self.loss_history[-1]
        
        # Check for plateau
        if len(self.loss_history) >= 2:
            improvement = self.meta_state['best_loss'] - current_loss
            
            if improvement <= self.loss_threshold:
                self.meta_state['plateau_count'] += 1
                self.meta_state['improvement_count'] = 0
            else:
                self.meta_state['plateau_count'] = 0
                self.meta_state['improvement_count'] += 1
                self.meta_state['best_loss'] = min(self.meta_state['best_loss'], current_loss)
                
        # Reduce learning rate on plateau
        if self.meta_state['plateau_count'] >= self.patience:
            self._reduce_learning_rate()
            self.meta_state['plateau_count'] = 0
            
        # Adjust based on gradient norms
        if gradient_norm is not None:
            if gradient_norm < self.gradient_threshold:
                self._increase_learning_rate()
            elif gradient_norm > self.gradient_threshold * 10:
                self._decrease_learning_rate()
                
    def _reduce_learning_rate(self):
        """Reduce learning rate by factor."""
        for param_group in self.optimizer.param_groups:
            new_lr = max(
                param_group['lr'] * self.factor,
                self.min_lr
            )
            param_group['lr'] = new_lr
            
    def _increase_learning_rate(self):
        """Increase learning rate by factor (capped at max_lr)."""
        for param_group in self.optimizer.param_groups:
            new_lr = min(
                param_group['lr'] / self.factor,
                self.max_lr
            )
            param_group['lr'] = new_lr
            
    def _decrease_learning_rate(self):
        """Decrease learning rate due to exploding gradients."""
        for param_group in self.optimizer.param_groups:
            new_lr = max(
                param_group['lr'] * 0.1,
                self.min_lr
            )
            param_group['lr'] = new_lr
            
    def get_last_lr(self) -> float:
        """Get last learning rate."""
        if self.optimizer.param_groups:
            return self.optimizer.param_groups[0]['lr']
        return self.base_lr
        
    def get_learning_rate_stats(self) -> Dict[str, float]:
        """Get learning rate statistics."""
        if not self.learning_rate_history:
            return {}
            
        return {
            'current_lr': self.get_last_lr(),
            'min_lr': min(self.learning_rate_history),
            'max_lr': max(self.learning_rate_history),
            'avg_lr': np.mean(self.learning_rate_history),
            'lr_changes': len(self.learning_rate_history) - 1
        }


class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    Uses Gaussian processes to efficiently search hyperparameter space.
    """
    
    def __init__(
        self,
        parameter_space: Dict[str, Tuple[float, float]],
        acquisition_function: str = 'ei',  # 'ei', 'ucb', 'pi'
        n_initial_points: int = 10,
        exploration_weight: float = 0.1
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            parameter_space: Dictionary mapping parameter names to (min, max) tuples
            acquisition_function: Acquisition function to use
            n_initial_points: Number of initial random points
            exploration_weight: Weight for exploration vs exploitation
        """
        self.parameter_space = parameter_space
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.exploration_weight = exploration_weight
        
        # Optimization history
        self.X = []  # Parameter combinations
        self.y = []  # Objective values
        
        # Gaussian Process state
        self.gp_params = {
            'length_scale': 1.0,
            'variance': 1.0,
            'noise_level': 1e-4
        }
        
        # Optimization state
        self.best_params = None
        self.best_score = -float('inf')
        
    def suggest_next_point(self) -> Dict[str, float]:
        """
        Suggest next hyperparameter combination using acquisition function.
        
        Returns:
            Dictionary of parameter values
        """
        if len(self.X) < self.n_initial_points:
            # Random sampling for initial points
            return self._random_sample()
            
        # Optimize acquisition function
        return self._optimize_acquisition_function()
        
    def _random_sample(self) -> Dict[str, float]:
        """Generate random parameter sample."""
        params = {}
        for param_name, (min_val, max_val) in self.parameter_space.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = np.random.randint(min_val, max_val + 1)
            else:
                params[param_name] = np.random.uniform(min_val, max_val)
        return params
        
    def _optimize_acquisition_function(self) -> Dict[str, float]:
        """Optimize acquisition function to find next point."""
        def objective(x):
            """Objective function for acquisition optimization."""
            params = self._unpack_params(x)
            return -self._acquisition_function(params)  # Minimize negative acquisition
            
        # Start from random points
        n_attempts = 100
        best_x = None
        best_value = float('inf')
        
        for _ in range(n_attempts):
            # Random starting point
            x0 = self._random_sample_params()
            
            try:
                # Optimize using scipy
                result = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B',
                    bounds=[(0, 1) for _ in range(len(self.parameter_space))]
                )
                
                if result.success and result.fun < best_value:
                    best_value = result.fun
                    best_x = result.x
            except:
                continue
                
        if best_x is None:
            # Fallback to random sampling
            return self._random_sample()
            
        return self._unpack_params(best_x)
        
    def _random_sample_params(self) -> List[float]:
        """Generate random parameter vector."""
        return [np.random.random() for _ in range(len(self.parameter_space))]
        
    def _unpack_params(self, x: List[float]) -> Dict[str, float]:
        """Convert parameter vector to dictionary."""
        params = {}
        for i, (param_name, (min_val, max_val)) in enumerate(self.parameter_space.items()):
            if isinstance(min_val, int) and isinstance(max_val, int):
                params[param_name] = int(np.round(min_val + x[i] * (max_val - min_val)))
            else:
                params[param_name] = min_val + x[i] * (max_val - min_val)
        return params
        
    def _acquisition_function(self, params: Dict[str, float]) -> float:
        """
        Compute acquisition function value for given parameters.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Acquisition function value
        """
        # Get Gaussian process predictions
        mu, sigma = self._gp_predict(params)
        
        if self.acquisition_function == 'ei':
            # Expected Improvement
            f_best = self.best_score
            if sigma == 0:
                return 0
                
            z = (mu - f_best - self.exploration_weight) / sigma
            ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
            return ei
            
        elif self.acquisition_function == 'ucb':
            # Upper Confidence Bound
            return mu + self.exploration_weight * sigma
            
        elif self.acquisition_function == 'pi':
            # Probability of Improvement
            f_best = self.best_score
            if sigma == 0:
                return 0
                
            z = (mu - f_best - self.exploration_weight) / sigma
            return norm.cdf(z)
            
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
            
    def _gp_predict(self, params: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict mean and variance using Gaussian Process.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Tuple of (mean, variance)
        """
        if not self.X:
            return 0.0, 1.0
            
        # Convert parameters to vector
        x = np.array([params[name] for name in self.parameter_space.keys()])
        
        # Compute kernel matrix
        K = np.zeros((len(self.X), len(self.X)))
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                K[i, j] = self._rbf_kernel(self.X[i], self.X[j])
                
        # Add noise
        K += self.gp_params['noise_level'] * np.eye(len(self.X))
        
        # Compute kernel with new point
        k_star = np.array([
            self._rbf_kernel(x, x_i) 
            for x_i in self.X
        ])
        
        # GP prediction
        try:
            K_inv = np.linalg.inv(K)
            mu = k_star.T @ K_inv @ np.array(self.y)
            k_star_star = self._rbf_kernel(x, x) + self.gp_params['noise_level']
            sigma_sq = k_star_star - k_star.T @ K_inv @ k_star
            sigma = max(0, sigma_sq)
            
            return mu, sigma
        except:
            # Fallback if matrix is singular
            return np.mean(self.y), np.var(self.y)
            
    def _rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute RBF kernel between two points."""
        length_scale = self.gp_params['length_scale']
        distance = np.sum((x1 - x2) ** 2)
        variance = self.gp_params['variance']
        return variance * np.exp(-0.5 * distance / (length_scale ** 2))
        
    def tell(self, params: Dict[str, float], score: float) -> None:
        """
        Update optimizer with new observation.
        
        Args:
            params: Parameter dictionary
            score: Objective value
        """
        # Convert to vector
        x = np.array([params[name] for name in self.parameter_space.keys()])
        
        # Store observation
        self.X.append(x)
        self.y.append(score)
        
        # Update best result
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            
    def optimize(
        self, 
        objective_function: Callable[[Dict[str, float]], float],
        n_iterations: int = 50,
        verbose: bool = True
    ) -> Tuple[Dict[str, float], float]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_function: Function to optimize
            n_iterations: Number of optimization iterations
            verbose: Print progress
            
        Returns:
            Tuple of (best_params, best_score)
        """
        if verbose:
            print(f"Starting Bayesian optimization with {n_iterations} iterations")
            
        for iteration in range(n_iterations):
            # Suggest next point
            next_params = self.suggest_next_point()
            
            # Evaluate objective
            score = objective_function(next_params)
            
            # Update optimizer
            self.tell(next_params, score)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}: "
                      f"Best score = {self.best_score:.4f}")
                
        if verbose:
            print(f"Optimization complete. Best score: {self.best_score:.4f}")
            
        return self.best_params, self.best_score
        
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history."""
        return {
            'X': copy.deepcopy(self.X),
            'y': copy.deepcopy(self.y),
            'best_params': copy.deepcopy(self.best_params),
            'best_score': self.best_score,
            'acquisition_function': self.acquisition_function,
            'parameter_space': self.parameter_space
        }


class AdaptiveOptimizer:
    """
    Meta-learning enhanced optimizer with automatic hyperparameter tuning.
    
    Combines standard optimizers with meta-learning for faster convergence
    and better generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: str = 'adam',  # 'adam', 'sgd', 'adamw'
        use_meta_learning: bool = True,
        meta_learning_rate: float = 0.001,
        adaptation_frequency: int = 100,
        **optimizer_kwargs
    ):
        """
        Initialize adaptive optimizer.
        
        Args:
            model: Neural network model
            base_optimizer: Base optimizer type
            use_meta_learning: Whether to use meta-learning
            meta_learning_rate: Learning rate for meta-parameters
            adaptation_frequency: Frequency of meta-parameter updates
            **optimizer_kwargs: Optimizer-specific parameters
        """
        self.model = model
        self.base_optimizer_type = base_optimizer
        self.use_meta_learning = use_meta_learning
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_frequency = adaptation_frequency
        
        # Initialize base optimizer
        if base_optimizer == 'adam':
            self.optimizer = Adam(model.parameters(), **optimizer_kwargs)
        elif base_optimizer == 'sgd':
            self.optimizer = SGD(model.parameters(), **optimizer_kwargs)
        elif base_optimizer == 'adamw':
            self.optimizer = AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {base_optimizer}")
            
        # Learning rate scheduler
        self.scheduler = LearningRateScheduler(
            self.optimizer, 
            scheduler_type='adaptive',
            **optimizer_kwargs.get('scheduler_config', {})
        )
        
        # Meta-learning components
        if use_meta_learning:
            self.meta_optimizer = Adam(model.parameters(), lr=meta_learning_rate)
            
            # Meta-parameters for different training phases
            self.meta_params = nn.ParameterDict({
                'learning_rate_modifier': nn.Parameter(torch.tensor(1.0)),
                'gradient_clip_modifier': nn.Parameter(torch.tensor(1.0)),
                'momentum_modifier': nn.Parameter(torch.tensor(1.0)),
                'weight_decay_modifier': nn.Parameter(torch.tensor(1.0))
            })
            
        # Optimization state
        self.step_count = 0
        self.loss_history = deque(maxlen=1000)
        self.gradient_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        self.optimizer.zero_grad()
        if self.use_meta_learning:
            self.meta_optimizer.zero_grad()
            
    def step(self, loss: Optional[float] = None) -> None:
        """
        Perform optimization step.
        
        Args:
            loss: Current loss for meta-learning adaptation
        """
        self.step_count += 1
        
        # Compute gradient norms for adaptation
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        total_norm = total_norm ** 0.5
        
        # Store metrics
        if loss is not None:
            self.loss_history.append(loss)
        self.gradient_history.append(total_norm)
        
        # Update learning rate
        self.scheduler.step(loss, total_norm)
        
        # Meta-learning adaptation
        if self.use_meta_learning and self.step_count % self.adaptation_frequency == 0:
            self._adapt_meta_parameters()
            
        # Apply meta-parameter modifiers
        if self.use_meta_learning:
            self._apply_meta_modifiers()
            
        # Step base optimizer
        self.optimizer.step()
        
    def _apply_meta_modifiers(self) -> None:
        """Apply meta-parameter modifiers to optimizer."""
        if not self.use_meta_learning:
            return
            
        lr_modifier = torch.clamp(self.meta_params['learning_rate_modifier'], 0.1, 10.0)
        clip_modifier = torch.clamp(self.meta_params['gradient_clip_modifier'], 0.1, 5.0)
        momentum_modifier = torch.clamp(self.meta_params['momentum_modifier'], 0.1, 2.0)
        weight_decay_modifier = torch.clamp(self.meta_params['weight_decay_modifier'], 0.1, 2.0)
        
        # Apply to optimizer parameters
        for param_group in self.optimizer.param_groups:
            # Modify learning rate
            base_lr = param_group.get('lr', 0.001)
            param_group['lr'] = base_lr * lr_modifier.item()
            
            # Modify gradient clipping
            if 'max_norm' in param_group:
                param_group['max_norm'] *= clip_modifier.item()
                
            # Modify momentum (if applicable)
            if 'momentum' in param_group:
                base_momentum = param_group['momentum']
                param_group['momentum'] = base_momentum * momentum_modifier.item()
                
    def _adapt_meta_parameters(self) -> None:
        """Adapt meta-parameters based on recent performance."""
        if len(self.loss_history) < self.adaptation_frequency:
            return
            
        # Compute performance trends
        recent_losses = list(self.loss_history)[-self.adaptation_frequency:]
        
        if len(recent_losses) < 2:
            return
            
        # Compute loss improvement rate
        initial_loss = recent_losses[0]
        final_loss = recent_losses[-1]
        improvement_rate = (initial_loss - final_loss) / len(recent_losses)
        
        # Compute gradient norms trend
        recent_grads = list(self.gradient_history)[-self.adaptation_frequency:]
        avg_gradient_norm = np.mean(recent_grads)
        
        # Meta-learning step for meta-parameters
        self.meta_optimizer.zero_grad()
        
        # Learning rate modifier adaptation
        if improvement_rate < 0:  # Loss not improving
            # Increase learning rate slightly
            loss_lr = -improvement_rate * 0.01  # Small positive gradient
        else:
            loss_lr = improvement_rate * 0.001
            
        self.meta_params['learning_rate_modifier'].grad = torch.tensor(loss_lr)
        
        # Gradient clip modifier adaptation
        if avg_gradient_norm < 1e-5:  # Very small gradients
            # Increase gradient clipping threshold
            clip_loss = 0.01 / (avg_gradient_norm + 1e-8)
        elif avg_gradient_norm > 10:  # Large gradients
            # Decrease gradient clipping threshold
            clip_loss = avg_gradient_norm * 0.001
        else:
            clip_loss = 0
            
        if clip_loss != 0:
            self.meta_params['gradient_clip_modifier'].grad = torch.tensor(clip_loss)
            
        # Update meta-parameters
        self.meta_optimizer.step()
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'step_count': self.step_count,
            'current_lr': self.scheduler.get_last_lr(),
            'loss_stats': {
                'current_loss': self.loss_history[-1] if self.loss_history else None,
                'best_loss': min(self.loss_history) if self.loss_history else None,
                'avg_loss': np.mean(self.loss_history) if self.loss_history else None,
                'loss_improvement': (
                    self.loss_history[0] - self.loss_history[-1] 
                    if len(self.loss_history) > 1 else None
                )
            },
            'gradient_stats': {
                'current_norm': self.gradient_history[-1] if self.gradient_history else None,
                'avg_norm': np.mean(self.gradient_history) if self.gradient_history else None,
                'max_norm': max(self.gradient_history) if self.gradient_history else None
            }
        }
        
        # Add meta-learning stats if enabled
        if self.use_meta_learning:
            stats['meta_learning'] = {
                'learning_rate_modifier': self.meta_params['learning_rate_modifier'].item(),
                'gradient_clip_modifier': self.meta_params['gradient_clip_modifier'].item(),
                'momentum_modifier': self.meta_params['momentum_modifier'].item(),
                'weight_decay_modifier': self.meta_params['weight_decay_modifier'].item()
            }
            
        return stats
        
    def save_state(self) -> Dict[str, Any]:
        """Save optimizer state."""
        return {
            'optimizer_state': self.optimizer.state_dict(),
            'meta_optimizer_state': (
                self.meta_optimizer.state_dict() if self.use_meta_learning else None
            ),
            'scheduler_state': self.scheduler.state_dict() if hasattr(self.scheduler, 'state_dict') else {},
            'meta_params': (
                {k: v.item() for k, v in self.meta_params.items()} 
                if self.use_meta_learning else {}
            ),
            'optimization_stats': {
                'step_count': self.step_count,
                'loss_history': list(self.loss_history),
                'gradient_history': list(self.gradient_history),
                'performance_metrics': dict(self.performance_metrics)
            }
        }
        
    def load_state(self, state: Dict[str, Any]) -> None:
        """Load optimizer state."""
        self.optimizer.load_state_dict(state['optimizer_state'])
        
        if self.use_meta_learning and state.get('meta_optimizer_state'):
            self.meta_optimizer.load_state_dict(state['meta_optimizer_state'])
            
        if hasattr(self.scheduler, 'load_state_dict') and state.get('scheduler_state'):
            self.scheduler.load_state_dict(state['scheduler_state'])
            
        # Load meta-parameters
        if self.use_meta_learning and state.get('meta_params'):
            for name, value in state['meta_params'].items():
                self.meta_params[name].data = torch.tensor(value)
                
        # Load optimization stats
        if state.get('optimization_stats'):
            stats = state['optimization_stats']
            self.step_count = stats.get('step_count', 0)
            self.loss_history = deque(stats.get('loss_history', []), maxlen=1000)
            self.gradient_history = deque(stats.get('gradient_history', []), maxlen=1000)
            self.performance_metrics = defaultdict(list, stats.get('performance_metrics', {}))


def hyperparameter_optimization_example():
    """Example of hyperparameter optimization."""
    
    # Define parameter space
    parameter_space = {
        'learning_rate': (1e-5, 1e-2),
        'batch_size': (16, 128),
        'weight_decay': (1e-6, 1e-3),
        'dropout_rate': (0.0, 0.5)
    }
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        parameter_space=parameter_space,
        acquisition_function='ei',
        n_initial_points=5
    )
    
    # Define objective function (example)
    def objective(params):
        # Simulate training with given parameters
        # In practice, this would be your actual training function
        lr = params['learning_rate']
        bs = params['batch_size']
        wd = params['weight_decay']
        dr = params['dropout_rate']
        
        # Simulate validation accuracy
        accuracy = -np.random.normal(
            loc=0.8 - lr*10 - dr*2, 
            scale=0.05
        )
        return accuracy  # Higher is better
    
    # Run optimization
    best_params, best_score = optimizer.optimize(
        objective_function=objective,
        n_iterations=20,
        verbose=True
    )
    
    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score}")
    
    return optimizer


def create_adaptive_training_pipeline(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    num_epochs: int,
    use_bayesian_optimization: bool = True,
    optimization_iterations: int = 50
) -> Dict[str, Any]:
    """
    Create complete adaptive training pipeline.
    
    Args:
        model: Neural network model
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        num_epochs: Number of training epochs
        use_bayesian_optimization: Whether to use Bayesian optimization
        optimization_iterations: Number of optimization iterations
        
    Returns:
        Training results and best hyperparameters
    """
    # Define parameter space for optimization
    parameter_space = {
        'learning_rate': (1e-5, 1e-2),
        'weight_decay': (1e-6, 1e-3),
        'beta1': (0.8, 0.99),
        'beta2': (0.9, 0.999)
    }
    
    best_params = None
    best_score = -float('inf')
    
    if use_bayesian_optimization:
        # Bayesian optimization for hyperparameter tuning
        def objective(params):
            optimizer = AdaptiveOptimizer(
                model=model,
                base_optimizer='adam',
                lr=params['learning_rate'],
                betas=(params['beta1'], params['beta2']),
                weight_decay=params['weight_decay']
            )
            
            # Train for a few epochs and return validation score
            for epoch in range(3):  # Short training for optimization
                for batch in train_dataloader:
                    x, y = batch
                    optimizer.zero_grad()
                    output = model(x)
                    loss = F.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step(loss.item())
                    
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    x, y = batch
                    output = model(x)
                    _, predicted = torch.max(output.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
            model.train()
            
            return correct / total  # Accuracy as objective
        
        optimizer = BayesianOptimizer(
            parameter_space=parameter_space,
            acquisition_function='ei'
        )
        
        best_params, best_score = optimizer.optimize(
            objective_function=objective,
            n_iterations=optimization_iterations
        )
    
    # Use best parameters for final training
    final_params = best_params if best_params else {
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'beta1': 0.9,
        'beta2': 0.999
    }
    
    # Create final optimizer
    final_optimizer = AdaptiveOptimizer(
        model=model,
        base_optimizer='adam',
        lr=final_params['learning_rate'],
        betas=(final_params['beta1'], final_params['beta2']),
        weight_decay=final_params['weight_decay'],
        use_meta_learning=True
    )
    
    # Train model
    training_history = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_dataloader:
            x, y = batch
            final_optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            final_optimizer.step(loss.item())
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        model.train()
        
        training_history.append({
            'epoch': epoch,
            'train_loss': epoch_loss / len(train_dataloader),
            'val_accuracy': val_correct / val_total,
            'optimization_stats': final_optimizer.get_optimization_stats()
        })
        
    return {
        'training_history': training_history,
        'best_hyperparameters': best_params,
        'best_validation_accuracy': best_score,
        'final_optimization_stats': final_optimizer.get_optimization_stats()
    }