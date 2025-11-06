"""
Model-Agnostic Meta-Learning (MAML) Implementation
Advanced MAML with first-order and second-order gradient support.

This module implements the foundational MAML algorithm for fast adaptation
across different tasks and domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import copy
from collections import defaultdict


class MetaBatch:
    """Container for meta-learning batch data."""
    
    def __init__(self, tasks: List[Dict[str, torch.Tensor]]):
        """
        Initialize meta batch.
        
        Args:
            tasks: List of task dictionaries with 'support' and 'query' sets
        """
        self.tasks = tasks
        
    def __len__(self):
        return len(self.tasks)
        
    def __getitem__(self, idx):
        return self.tasks[idx]


class MAML(nn.Module):
    """Base MAML implementation with support for different gradient orders."""
    
    def __init__(
        self,
        model: nn.Module,
        lr_inner: float = 0.01,
        lr_meta: float = 0.001,
        num_inner_steps: int = 5,
        order: str = 'first',  # 'first' or 'second'
        inner_optimizer: str = 'sgd',  # 'sgd' or 'adam'
        gradient_clip: Optional[float] = None
    ):
        """
        Initialize MAML.
        
        Args:
            model: Base model to meta-learn
            lr_inner: Learning rate for inner loop updates
            lr_meta: Learning rate for meta-optimizer
            num_inner_steps: Number of inner loop optimization steps
            order: Order of meta-gradients ('first' or 'second')
            inner_optimizer: Inner loop optimizer type
            gradient_clip: Gradient clipping value
        """
        super().__init__()
        self.model = model
        self.lr_inner = lr_inner
        self.lr_meta = lr_meta
        self.num_inner_steps = num_inner_steps
        self.order = order
        self.gradient_clip = gradient_clip
        
        # Initialize meta-optimizer
        self.meta_optimizer = Adam(self.model.parameters(), lr=lr_meta)
        
        # Inner loop optimizer
        self.inner_optimizer_name = inner_optimizer
        
        # State tracking
        self.meta_losses = []
        self.adaptation_metrics = defaultdict(list)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
        
    def inner_loop_step(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform inner loop optimization step.
        
        Args:
            support_x: Support set input features
            support_y: Support set target labels
            fast_weights: Pre-computed fast weights (for second order)
            
        Returns:
            Tuple of (loss, updated_fast_weights)
        """
        if fast_weights is None:
            fast_weights = dict(self.named_parameters())
        
        # Forward pass with fast weights
        output = self._forward_with_weights(support_x, fast_weights)
        loss = F.mse_loss(output, support_y)
        
        # Compute gradients w.r.t. fast weights
        grads = torch.autograd.grad(
            loss, fast_weights.values(), 
            create_graph=(self.order == 'second'),
            retain_graph=True
        )
        
        # Update fast weights
        updated_weights = {}
        for (name, weight), grad in zip(fast_weights.items(), grads):
            if self.inner_optimizer_name == 'sgd':
                updated_weights[name] = weight - self.lr_inner * grad
            else:  # adam
                # Simplified Adam update for inner loop
                updated_weights[name] = weight - self.lr_inner * grad
                
        return loss, updated_weights
        
    def _forward_with_weights(
        self, 
        x: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using provided weights."""
        # Store original parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
            
        # Replace parameters with fast weights
        for name, weight in weights.items():
            for param_name, param in self.model.named_parameters():
                if param_name == name:
                    param.data = weight
                    break
                    
        # Forward pass
        output = self.model(x)
        
        # Restore original parameters
        for name, param in original_params.items():
            for param_name, orig_param in self.model.named_parameters():
                if param_name == name:
                    orig_param.data = param
                    break
                    
        return output
        
    def adapt_to_task(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt model parameters to a specific task.
        
        Args:
            support_x: Support set input features
            support_y: Support set target labels
            
        Returns:
            Fast weights adapted to the task
        """
        fast_weights = dict(self.named_parameters())
        
        for step in range(self.num_inner_steps):
            loss, fast_weights = self.inner_loop_step(
                support_x, support_y, fast_weights
            )
            
        return fast_weights
        
    def meta_train_step(
        self, 
        meta_batch: MetaBatch
    ) -> Dict[str, float]:
        """
        Perform one meta-training step.
        
        Args:
            meta_batch: Batch of tasks for meta-learning
            
        Returns:
            Dictionary containing training metrics
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        meta_accuracy = 0.0
        
        task_losses = []
        task_accuracies = []
        
        for task in meta_batch.tasks:
            support_x = task['support_x']
            support_y = task['support_y']
            query_x = task['query_x']
            query_y = task['query_y']
            
            # Inner loop: adapt to task
            fast_weights = self.adapt_to_task(support_x, support_y)
            
            # Outer loop: compute meta-loss on query set
            if self.order == 'first':
                # First-order approximation
                output = self._forward_with_weights(query_x, fast_weights)
                loss = F.mse_loss(output, query_y)
                meta_loss += loss
                
                # Compute accuracy
                pred = output.round()
                accuracy = (pred == query_y).float().mean()
                task_accuracies.append(accuracy.item())
                
            else:
                # Second-order gradients
                output = self._forward_with_weights(query_x, fast_weights)
                loss = F.mse_loss(output, query_y)
                meta_loss += loss
                
                pred = output.round()
                accuracy = (pred == query_y).float().mean()
                task_accuracies.append(accuracy.item())
                
            task_losses.append(loss.item())
            
        # Average meta-loss
        meta_loss = meta_loss / len(meta_batch.tasks)
        meta_accuracy = np.mean(task_accuracies)
        
        # Backpropagation
        meta_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            
        # Update meta-parameters
        self.meta_optimizer.step()
        
        # Update state
        self.meta_losses.append(meta_loss.item())
        self.adaptation_metrics['task_losses'].append(task_losses)
        self.adaptation_metrics['task_accuracies'].append(task_accuracies)
        
        return {
            'meta_loss': meta_loss.item(),
            'meta_accuracy': meta_accuracy,
            'avg_task_loss': np.mean(task_losses),
            'avg_task_accuracy': meta_accuracy
        }
        
    def predict(
        self, 
        x: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> torch.Tensor:
        """
        Make predictions after adaptation to a task.
        
        Args:
            x: Input to predict
            support_x: Support set for adaptation
            support_y: Support set labels
            
        Returns:
            Predictions after adaptation
        """
        # Adapt to task
        fast_weights = self.adapt_to_task(support_x, support_y)
        
        # Make predictions
        return self._forward_with_weights(x, fast_weights)
        
    def get_meta_state(self) -> Dict[str, Any]:
        """Get current meta-learning state."""
        return {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'meta_loss_history': self.meta_losses.copy(),
            'adaptation_metrics': dict(self.adaptation_metrics),
            'meta_optimizer_state': self.meta_optimizer.state_dict()
        }
        
    def load_meta_state(self, state: Dict[str, Any]) -> None:
        """Load meta-learning state."""
        self.model.load_state_dict(state['model_state'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer_state'])
        self.meta_losses = state.get('meta_loss_history', [])
        self.adaptation_metrics = defaultdict(
            list, state.get('adaptation_metrics', {})
        )


class MAMLFirstOrder(MAML):
    """First-order MAML implementation (Reptile approximation)."""
    
    def __init__(self, *args, **kwargs):
        kwargs['order'] = 'first'
        super().__init__(*args, **kwargs)
        
    def inner_loop_step(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        First-order inner loop step without gradient storage.
        """
        if fast_weights is None:
            fast_weights = dict(self.named_parameters())
        
        # Forward pass
        output = self._forward_with_weights(support_x, fast_weights)
        loss = F.mse_loss(output, support_y)
        
        # Compute gradients w.r.t. fast weights (no higher-order)
        grads = torch.autograd.grad(
            loss, fast_weights.values(), 
            create_graph=False,
            retain_graph=True
        )
        
        # Update fast weights
        updated_weights = {}
        for (name, weight), grad in zip(fast_weights.items(), grads):
            updated_weights[name] = weight - self.lr_inner * grad
                
        return loss, updated_weights


class MAMLSecondOrder(MAML):
    """Second-order MAML with exact gradient computation."""
    
    def __init__(self, *args, **kwargs):
        kwargs['order'] = 'second'
        super().__init__(*args, **kwargs)
        
    def inner_loop_step(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        fast_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Second-order inner loop step with gradient storage.
        """
        if fast_weights is None:
            fast_weights = dict(self.named_parameters())
        
        # Forward pass
        output = self._forward_with_weights(support_x, fast_weights)
        loss = F.mse_loss(output, support_y)
        
        # Compute gradients w.r.t. fast weights (with higher-order)
        grads = torch.autograd.grad(
            loss, fast_weights.values(), 
            create_graph=True,
            retain_graph=True
        )
        
        # Update fast weights
        updated_weights = {}
        for (name, weight), grad in zip(fast_weights.items(), grads):
            updated_weights[name] = weight - self.lr_inner * grad
                
        return loss, updated_weights


class MultiTaskMAML(MAML):
    """Multi-task MAML for handling diverse task distributions."""
    
    def __init__(
        self, 
        *args, 
        task_weights: Optional[torch.Tensor] = None,
        adaptive_weights: bool = True,
        **kwargs
    ):
        """
        Initialize multi-task MAML.
        
        Args:
            task_weights: Initial task weights
            adaptive_weights: Whether to adapt task weights during training
        """
        super().__init__(*args, **kwargs)
        self.adaptive_weights = adaptive_weights
        
        if task_weights is None:
            num_tasks = len(self.adaptation_metrics) if self.adaptation_metrics else 1
            self.task_weights = nn.Parameter(torch.ones(1) / num_tasks)
        else:
            self.task_weights = nn.Parameter(task_weights)
            
    def meta_train_step(
        self, 
        meta_batch: MetaBatch,
        task_types: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Meta training step with adaptive task weighting.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0
        task_losses = []
        task_accuracies = []
        
        for i, task in enumerate(meta_batch.tasks):
            support_x = task['support_x']
            support_y = task['support_y']
            query_x = task['query_x']
            query_y = task['query_y']
            
            # Inner loop adaptation
            fast_weights = self.adapt_to_task(support_x, support_y)
            
            # Compute task-specific loss
            output = self._forward_with_weights(query_x, fast_weights)
            task_loss = F.mse_loss(output, query_y)
            task_losses.append(task_loss.item())
            
            # Compute accuracy
            pred = output.round()
            accuracy = (pred == query_y).float().mean()
            task_accuracies.append(accuracy.item())
            
            # Apply task weight
            if self.adaptive_weights:
                weight = self.task_weights.abs()  # Ensure positive weights
                weight = weight / weight.sum()  # Normalize
                meta_loss += weight * task_loss
            else:
                meta_loss += task_loss
                
        # Average meta-loss
        meta_loss = meta_loss / len(meta_batch.tasks)
        meta_accuracy = np.mean(task_accuracies)
        
        # Backpropagation
        meta_loss.backward()
        
        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip
            )
            
        self.meta_optimizer.step()
        
        # Update state
        self.meta_losses.append(meta_loss.item())
        self.adaptation_metrics['task_losses'].append(task_losses)
        self.adaptation_metrics['task_accuracies'].append(task_accuracies)
        
        return {
            'meta_loss': meta_loss.item(),
            'meta_accuracy': meta_accuracy,
            'task_weights': self.task_weights.detach().cpu().numpy(),
            'avg_task_loss': np.mean(task_losses),
            'avg_task_accuracy': meta_accuracy
        }


def create_meta_batch(
    tasks: List[Dict[str, torch.Tensor]],
    batch_size: int = 4
) -> MetaBatch:
    """
    Create a meta-batch for MAML training.
    
    Args:
        tasks: List of task dictionaries
        batch_size: Number of tasks in each meta-batch
        
    Returns:
        MetaBatch object
    """
    if len(tasks) < batch_size:
        # Replicate tasks if needed
        tasks = tasks * ((batch_size // len(tasks)) + 1)
        
    # Randomly sample tasks
    selected_tasks = np.random.choice(
        tasks, size=batch_size, replace=True
    ).tolist()
    
    return MetaBatch(selected_tasks)


def maml_evaluate(
    model: MAML,
    test_tasks: List[Dict[str, torch.Tensor]],
    num_adapt_steps: int = 5
) -> Dict[str, float]:
    """
    Evaluate MAML on test tasks.
    
    Args:
        model: Trained MAML model
        test_tasks: List of test task dictionaries
        num_adapt_steps: Number of adaptation steps for evaluation
        
    Returns:
        Evaluation metrics
    """
    original_steps = model.num_inner_steps
    model.num_inner_steps = num_adapt_steps
    
    accuracies = []
    losses = []
    
    for task in test_tasks:
        support_x = task['support_x']
        support_y = task['support_y']
        query_x = task['query_x']
        query_y = task['query_y']
        
        # Adapt to task
        fast_weights = model.adapt_to_task(support_x, support_y)
        
        # Evaluate on query set
        output = model._forward_with_weights(query_x, fast_weights)
        loss = F.mse_loss(output, query_y)
        accuracy = (output.round() == query_y).float().mean()
        
        accuracies.append(accuracy.item())
        losses.append(loss.item())
        
    # Restore original steps
    model.num_inner_steps = original_steps
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'num_tasks': len(test_tasks)
    }