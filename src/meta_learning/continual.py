"""
Continual Learning Implementation
Prevents catastrophic forgetting in lifelong learning scenarios.

This module implements state-of-the-art continual learning algorithms
including Elastic Weight Consolidation (EWC), Progressive Networks,
and memory replay mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import copy
from collections import defaultdict, deque
import math
from abc import ABC, abstractmethod


class MemoryBuffer:
    """Memory buffer for experience replay in continual learning."""
    
    def __init__(
        self, 
        max_size: int = 1000,
        buffer_type: str = 'reservoir',  # 'reservoir', 'ring', 'priority'
        priority_function: Optional[callable] = None
    ):
        """
        Initialize memory buffer.
        
        Args:
            max_size: Maximum buffer size
            buffer_type: Type of buffer ('reservoir', 'ring', 'priority')
            priority_function: Function to compute priority for samples
        """
        self.max_size = max_size
        self.buffer_type = buffer_type
        self.priority_function = priority_function
        
        self.buffer = []
        self.priorities = []
        self.task_labels = []
        self.sample_count = 0
        
    def add(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        task_label: Optional[int] = None,
        priority: Optional[float] = None
    ) -> None:
        """Add sample to buffer."""
        if priority is None and self.priority_function:
            priority = self.priority_function(x, y)
        elif priority is None:
            priority = 1.0
            
        sample = (x, y, task_label, priority)
        
        if self.buffer_type == 'reservoir':
            self._add_reservoir(sample)
        elif self.buffer_type == 'ring':
            self._add_ring(sample)
        elif self.buffer_type == 'priority':
            self._add_priority(sample)
        else:
            raise ValueError(f"Unknown buffer type: {self.buffer_type}")
            
    def _add_reservoir(self, sample: Tuple) -> None:
        """Add sample using reservoir sampling."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
            self.priorities.append(sample[3])
            if sample[2] is not None:
                self.task_labels.append(sample[2])
        else:
            # Reservoir sampling: replace with probability
            j = np.random.randint(0, self.sample_count + 1)
            if j < self.max_size:
                self.buffer[j] = sample
                self.priorities[j] = sample[3]
                if sample[2] is not None:
                    self.task_labels[j] = sample[2]
                    
        self.sample_count += 1
        
    def _add_ring(self, sample: Tuple) -> None:
        """Add sample using ring buffer."""
        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
            self.priorities.append(sample[3])
            if sample[2] is not None:
                self.task_labels.append(sample[2])
        else:
            # Replace oldest sample
            idx = self.sample_count % self.max_size
            self.buffer[idx] = sample
            self.priorities[idx] = sample[3]
            if sample[2] is not None:
                self.task_labels[idx] = sample[2]
                
        self.sample_count += 1
        
    def _add_priority(self, sample: Tuple) -> None:
        """Add sample using priority sampling."""
        self.buffer.append(sample)
        self.priorities.append(sample[3])
        if sample[2] is not None:
            self.task_labels.append(sample[2])
            
        # Keep only top max_size samples
        if len(self.buffer) > self.max_size:
            # Sort by priority and keep top max_size
            sorted_indices = np.argsort(self.priorities)[::-1]
            keep_indices = sorted_indices[:self.max_size]
            
            self.buffer = [self.buffer[i] for i in keep_indices]
            self.priorities = [self.priorities[i] for i in keep_indices]
            if self.task_labels:
                self.task_labels = [self.task_labels[i] for i in keep_indices]
                
        self.sample_count += 1
        
    def sample(
        self, 
        batch_size: int,
        task_label: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Sample batch from buffer."""
        if not self.buffer:
            return torch.empty(0), torch.empty(0), []
            
        # Filter by task label if specified
        if task_label is not None and self.task_labels:
            valid_indices = [
                i for i, (_, _, label, _) in enumerate(self.buffer) 
                if label == task_label
            ]
            if not valid_indices:
                # If no samples for specific task, sample from all
                valid_indices = list(range(len(self.buffer)))
        else:
            valid_indices = list(range(len(self.buffer)))
            
        # Sample indices
        if len(valid_indices) < batch_size:
            # Oversample with replacement
            sample_indices = np.random.choice(
                valid_indices, size=batch_size, replace=True
            ).tolist()
        else:
            sample_indices = np.random.choice(
                valid_indices, size=batch_size, replace=False
            ).tolist()
            
        # Extract samples
        sampled_x, sampled_y, sampled_tasks = [], [], []
        for idx in sample_indices:
            x, y, task, _ = self.buffer[idx]
            sampled_x.append(x)
            sampled_y.append(y)
            sampled_tasks.append(task)
            
        return torch.stack(sampled_x), torch.stack(sampled_y), sampled_tasks
        
    def __len__(self) -> int:
        return len(self.buffer)
        
    def clear(self) -> None:
        """Clear buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.task_labels.clear()
        self.sample_count = 0


class EWC:
    """
    Elastic Weight Consolidation for continual learning.
    
    Prevents catastrophic forgetting by penalizing changes to
    important parameters from previous tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 100,
        diagonal_approx: bool = True
    ):
        """
        Initialize EWC.
        
        Args:
            model: Neural network model
            ewc_lambda: EWC regularization coefficient
            fisher_samples: Number of samples to estimate Fisher information
            diagonal_approx: Use diagonal approximation of Fisher matrix
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.diagonal_approx = diagonal_approx
        
        # Store Fisher information and optimal parameters
        self.fisher_info = {}
        self.optimal_params = {}
        self.task_count = 0
        
    def compute_fisher_information(
        self,
        dataloader: DataLoader,
        parameter_subset: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information matrix for current task.
        
        Args:
            dataloader: DataLoader for Fisher computation
            parameter_subset: List of parameter names to compute Fisher for
            
        Returns:
            Dictionary of Fisher information for each parameter
        """
        # Initialize Fisher information
        if parameter_subset is None:
            parameter_subset = [name for name, _ in self.model.named_parameters()]
            
        fisher_info = {}
        for name in parameter_subset:
            fisher_info[name] = torch.zeros_like(
                list(self.model.parameters())[list(self.model.named_parameters())[0][0].find(name.split('.')[0])][1]
                if any(name in pname for pname, _ in self.model.named_parameters())
                else torch.tensor(0.0)
            )
            
        # Collect samples
        samples = []
        for batch in dataloader:
            if len(samples) >= self.fisher_samples:
                break
            samples.append(batch)
            
        if not samples:
            return fisher_info
            
        # Compute Fisher information for each parameter
        for name, param in self.model.named_parameters():
            if name in parameter_subset and param.requires_grad:
                fisher_sum = torch.zeros_like(param)
                
                for batch in samples:
                    # Zero gradients
                    self.model.zero_grad()
                    
                    # Forward pass
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        x, y = batch[0], batch[1]
                    else:
                        continue
                        
                    output = self.model(x)
                    loss = F.cross_entropy(output, y, reduction='sum')
                    
                    # Backward pass
                    loss.backward(retain_graph=True)
                    
                    # Accumulate squared gradients
                    if param.grad is not None:
                        fisher_sum += param.grad.data.clone() ** 2
                        
                # Average over samples
                fisher_info[name] = fisher_sum / len(samples)
                
        return fisher_info
        
    def update_weights(
        self,
        fisher_info: Dict[str, torch.Tensor],
        optimal_params: Dict[str, torch.Tensor]
    ) -> None:
        """Update stored Fisher information and optimal parameters."""
        if self.task_count == 0:
            self.fisher_info = fisher_info
            self.optimal_params = optimal_params
        else:
            # Average with previous tasks
            alpha = 1.0 / (self.task_count + 1)
            for name in fisher_info:
                if name in self.fisher_info:
                    self.fisher_info[name] = (
                        alpha * fisher_info[name] + 
                        (1 - alpha) * self.fisher_info[name]
                    )
                else:
                    self.fisher_info[name] = fisher_info[name]
                    
                if name in self.optimal_params:
                    self.optimal_params[name] = (
                        alpha * optimal_params[name] + 
                        (1 - alpha) * self.optimal_params[name]
                    )
                else:
                    self.optimal_params[name] = optimal_params[name]
                    
        self.task_count += 1
        
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_info:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        ewc_loss = 0.0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info and name in self.optimal_params:
                fisher = self.fisher_info[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
                
        return self.ewc_lambda * ewc_loss
        
    def fit_task(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 1
    ) -> Dict[str, List[float]]:
        """Fit model to current task with EWC regularization."""
        optimizer = torch.optim.Adam(self.model.parameters())
        train_losses = []
        val_losses = []
        
        # Store current optimal parameters
        optimal_params = {
            name: param.clone().detach() 
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        
        for epoch in range(epochs):
            epoch_train_loss = 0.0
            self.model.train()
            
            for batch in train_dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    continue
                    
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(x)
                task_loss = F.cross_entropy(output, y)
                
                # Add EWC loss
                ewc_loss = self.compute_ewc_loss()
                total_loss = task_loss + ewc_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_train_loss += total_loss.item()
                
            train_losses.append(epoch_train_loss / len(train_dataloader))
            
            # Validation
            if val_dataloader:
                val_loss = self.evaluate(val_dataloader)
                val_losses.append(val_loss)
            else:
                val_losses.append(val_losses[-1] if val_losses else 0.0)
                
        # Update Fisher information
        if self.task_count == 0:
            fisher_info = self.compute_fisher_information(train_dataloader)
            self.update_weights(fisher_info, optimal_params)
        else:
            # Accumulate with previous tasks
            fisher_info = self.compute_fisher_information(train_dataloader)
            for name in fisher_info:
                if name in self.fisher_info:
                    self.fisher_info[name] += fisher_info[name]
                else:
                    self.fisher_info[name] = fisher_info[name]
                    
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    continue
                    
                output = self.model(x)
                loss = F.cross_entropy(output, y, reduction='sum')
                total_loss += loss.item()
                total_samples += y.shape[0]
                
        return total_loss / total_samples if total_samples > 0 else 0.0


class ProgressiveNetwork(nn.Module):
    """
    Progressive Neural Networks for continual learning.
    
    Creates new columns for each task while preserving previous columns.
    """
    
    def __init__(
        self,
        base_architecture: nn.Module,
        num_tasks: int,
        freeze_prev_columns: bool = True,
        lateral_connections: bool = True
    ):
        """
        Initialize Progressive Network.
        
        Args:
            base_architecture: Base network architecture
            num_tasks: Number of tasks to support
            freeze_prev_columns: Freeze previous task columns
            lateral_connections: Use lateral connections between columns
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.freeze_prev_columns = freeze_prev_columns
        self.lateral_connections = lateral_connections
        
        # Create columns for each task
        self.columns = nn.ModuleList()
        
        for task_id in range(num_tasks):
            if task_id == 0:
                # First column is the base architecture
                self.columns.append(base_architecture)
            else:
                # Create new column based on base architecture
                new_column = copy.deepcopy(base_architecture)
                self.columns.append(new_column)
                
        # Lateral connections
        if lateral_connections:
            self.lateral_connections = nn.ModuleList()
            for task_id in range(1, num_tasks):
                lateral_conn = nn.ModuleList()
                for prev_task in range(task_id):
                    # Add lateral connections from previous columns
                    lateral_conn.append(
                        nn.Linear(
                            self._get_column_output_size(prev_task),
                            self._get_column_input_size(task_id)
                        )
                    )
                self.lateral_connections.append(lateral_conn)
        else:
            self.lateral_connections = None
            
        self.current_task = 0
        
    def _get_column_output_size(self, task_id: int) -> int:
        """Get output size of a column."""
        with torch.no_grad():
            dummy_input = torch.randn(1, *self._get_input_shape())
            dummy_output = self.columns[task_id](dummy_input)
            return dummy_output.numel()
            
    def _get_column_input_size(self, task_id: int) -> int:
        """Get input size of a column."""
        with torch.no_grad():
            dummy_input = torch.randn(1, *self._get_input_shape())
            dummy_output = self.columns[task_id](dummy_input)
            return dummy_input.numel()
            
    def _get_input_shape(self) -> Tuple:
        """Get input shape (override in subclasses)."""
        return (3, 32, 32)  # Default image input shape
        
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            task_id: Task ID (0-indexed)
            
        Returns:
            Output for the specified task
        """
        if task_id >= self.num_tasks:
            raise ValueError(f"Task ID {task_id} exceeds num_tasks {self.num_tasks}")
            
        # Freeze previous columns if specified
        if self.freeze_prev_columns and task_id > 0:
            for prev_task in range(task_id):
                for param in self.columns[prev_task].parameters():
                    param.requires_grad = False
                    
        # Forward through current column
        output = self.columns[task_id](x)
        
        # Add lateral connections if enabled
        if self.lateral_connections and task_id > 0:
            for prev_task in range(task_id):
                # Get output from previous column
                prev_output = self.columns[prev_task](x)
                
                # Apply lateral connection
                lateral_input = self.lateral_connections[task_id - 1][prev_task](prev_output)
                
                # Add to current output
                if output.shape == lateral_input.shape:
                    output = output + lateral_input
                    
        return output
        
    def set_current_task(self, task_id: int) -> None:
        """Set the current task."""
        self.current_task = task_id
        
    def add_new_column(self, new_column: nn.Module) -> None:
        """Add a new column for additional task."""
        self.columns.append(new_column)
        self.num_tasks += 1
        
        if self.lateral_connections:
            # Add lateral connections for new column
            new_lateral = nn.ModuleList()
            for prev_task in range(self.num_tasks - 1):
                new_lateral.append(
                    nn.Linear(
                        self._get_column_output_size(prev_task),
                        self._get_column_input_size(self.num_tasks - 1)
                    )
                )
            self.lateral_connections.append(new_lateral)


class ContinualLearner:
    """
    Unified continual learning interface supporting multiple strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategy: str = 'ewc',  # 'ewc', 'progressive', 'replay', 'lwf'
        buffer_size: int = 1000,
        **kwargs
    ):
        """
        Initialize Continual Learner.
        
        Args:
            model: Neural network model
            strategy: Learning strategy
            buffer_size: Size of memory buffer (for replay)
            **kwargs: Strategy-specific parameters
        """
        self.model = model
        self.strategy = strategy
        
        # Initialize strategy
        if strategy == 'ewc':
            self.ewc = EWC(model, **kwargs)
        elif strategy == 'progressive':
            if 'num_tasks' not in kwargs:
                raise ValueError("Progressive network requires num_tasks")
            self.progressive_net = ProgressiveNetwork(model, **kwargs)
        elif strategy == 'replay':
            self.memory_buffer = MemoryBuffer(max_size=buffer_size, **kwargs)
        elif strategy == 'lwf':
            self.old_model = None
            self.lwf_temperature = kwargs.get('temperature', 2.0)
            self.lwf_alpha = kwargs.get('alpha', 0.1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        self.task_history = []
        self.performance_history = defaultdict(list)
        
    def train_task(
        self,
        task_data: Dict[str, DataLoader],
        task_id: int,
        num_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Train on a new task.
        
        Args:
            task_data: Dictionary with 'train' and 'val' dataloaders
            task_id: Task ID
            num_epochs: Number of training epochs
            
        Returns:
            Training metrics
        """
        if self.strategy == 'ewc':
            return self._train_ewc(task_data, task_id, num_epochs)
        elif self.strategy == 'progressive':
            return self._train_progressive(task_data, task_id, num_epochs)
        elif self.strategy == 'replay':
            return self._train_replay(task_data, task_id, num_epochs)
        elif self.strategy == 'lwf':
            return self._train_lwf(task_data, task_id, num_epochs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
    def _train_ewc(self, task_data, task_id, num_epochs):
        """Train with EWC strategy."""
        train_losses = self.ewc.fit_task(
            task_data['train'], 
            task_data.get('val'), 
            num_epochs
        )
        
        self.task_history.append({
            'strategy': self.strategy,
            'task_id': task_id,
            'metrics': train_losses
        })
        
        return {
            'final_train_loss': train_losses['train_losses'][-1],
            'final_val_loss': train_losses['val_losses'][-1],
            'task_id': task_id
        }
        
    def _train_progressive(self, task_data, task_id, num_epochs):
        """Train with Progressive Networks strategy."""
        if task_id > 0:
            self.progressive_net.set_current_task(task_id)
            
        # Training loop
        optimizer = torch.optim.Adam(self.progressive_net.parameters())
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            self.progressive_net.train()
            epoch_loss = 0.0
            
            for batch in task_data['train']:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    continue
                    
                optimizer.zero_grad()
                output = self.progressive_net(x, task_id)
                loss = F.cross_entropy(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            train_losses.append(epoch_loss / len(task_data['train']))
            
            # Validation
            if 'val' in task_data:
                val_loss = self._evaluate_progressive(task_data['val'], task_id)
                val_losses.append(val_loss)
            else:
                val_losses.append(train_losses[-1])
                
        self.task_history.append({
            'strategy': self.strategy,
            'task_id': task_id,
            'metrics': {'train_losses': train_losses, 'val_losses': val_losses}
        })
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'task_id': task_id
        }
        
    def _evaluate_progressive(self, dataloader, task_id):
        """Evaluate progressive network."""
        self.progressive_net.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    continue
                    
                output = self.progressive_net(x, task_id)
                loss = F.cross_entropy(output, y, reduction='sum')
                total_loss += loss.item()
                total_samples += y.shape[0]
                
        return total_loss / total_samples if total_samples > 0 else 0.0
        
    def _train_replay(self, task_data, task_id, num_epochs):
        """Train with replay strategy."""
        # Add current task data to memory buffer
        for batch in task_data['train']:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
                for i in range(x.shape[0]):
                    self.memory_buffer.add(
                        x[i], y[i], task_id=task_id
                    )
                    
        # Training with replay
        optimizer = torch.optim.Adam(self.model.parameters())
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.model.train()
            
            # Sample replay data
            replay_size = min(len(self.memory_buffer) // 4, 32)
            if replay_size > 0:
                replay_x, replay_y, _ = self.memory_buffer.sample(
                    replay_size, task_label=task_id
                )
            else:
                replay_x, replay_y = torch.empty(0), torch.empty(0)
                
            for batch in task_data['train']:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    batch_x, batch_y = batch[0], batch[1]
                else:
                    continue
                    
                optimizer.zero_grad()
                
                # Current task loss
                output = self.model(batch_x)
                task_loss = F.cross_entropy(output, batch_y)
                
                # Replay loss
                replay_loss = 0.0
                if len(replay_x) > 0:
                    replay_output = self.model(replay_x)
                    replay_loss = F.cross_entropy(replay_output, replay_y)
                    
                # Combined loss
                total_loss = task_loss + 0.1 * replay_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                
            train_losses.append(epoch_loss / len(task_data['train']))
            
            # Validation
            if 'val' in task_data:
                val_loss = self._evaluate(dataloader=task_data['val'])
                val_losses.append(val_loss)
            else:
                val_losses.append(train_losses[-1])
                
        self.task_history.append({
            'strategy': self.strategy,
            'task_id': task_id,
            'metrics': {'train_losses': train_losses, 'val_losses': val_losses}
        })
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'task_id': task_id,
            'buffer_size': len(self.memory_buffer)
        }
        
    def _train_lwf(self, task_data, task_id, num_epochs):
        """Train with Learning Without Forgetting strategy."""
        if self.old_model is None:
            self.old_model = copy.deepcopy(self.model)
            
        optimizer = torch.optim.Adam(self.model.parameters())
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            self.model.train()
            
            for batch in task_data['train']:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    continue
                    
                optimizer.zero_grad()
                
                # Current task loss
                output = self.model(x)
                task_loss = F.cross_entropy(output, y)
                
                # Distillation loss from old model
                with torch.no_grad():
                    old_output = self.old_model(x)
                    
                distill_loss = F.kl_div(
                    F.log_softmax(output / self.lwf_temperature, dim=1),
                    F.softmax(old_output / self.lwf_temperature, dim=1),
                    reduction='batchmean'
                ) * (self.lwf_temperature ** 2)
                
                # Combined loss
                total_loss = (1 - self.lwf_alpha) * task_loss + self.lwf_alpha * distill_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item()
                
            train_losses.append(epoch_loss / len(task_data['train']))
            
            # Validation
            if 'val' in task_data:
                val_loss = self._evaluate(dataloader=task_data['val'])
                val_losses.append(val_loss)
            else:
                val_losses.append(train_losses[-1])
                
        # Update old model
        self.old_model = copy.deepcopy(self.model)
        
        self.task_history.append({
            'strategy': self.strategy,
            'task_id': task_id,
            'metrics': {'train_losses': train_losses, 'val_losses': val_losses}
        })
        
        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'task_id': task_id
        }
        
    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on dataset."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    continue
                    
                output = self.model(x)
                loss = F.cross_entropy(output, y, reduction='sum')
                total_loss += loss.item()
                total_samples += y.shape[0]
                
        return total_loss / total_samples if total_samples > 0 else 0.0
        
    def evaluate_all_tasks(self, test_data: Dict[int, DataLoader]) -> Dict[str, float]:
        """
        Evaluate model on all tasks seen so far.
        
        Args:
            test_data: Dictionary mapping task_id to test dataloader
            
        Returns:
            Evaluation results for each task
        """
        results = {}
        
        for task_id, dataloader in test_data.items():
            if self.strategy == 'progressive':
                loss = self._evaluate_progressive(dataloader, task_id)
            else:
                loss = self._evaluate(dataloader)
                
            results[f'task_{task_id}'] = loss
            self.performance_history[f'task_{task_id}'].append(loss)
            
        # Compute average performance
        results['average_loss'] = np.mean(list(results.values()))
        results['forgetting_measure'] = self._compute_forgetting_measure()
        
        return results
        
    def _compute_forgetting_measure(self) -> float:
        """Compute forgetting measure (difference between first and last performance)."""
        forgetting_scores = []
        
        for task_id, losses in self.performance_history.items():
            if len(losses) > 1:
                forgetting = losses[0] - losses[-1]  # First performance - last performance
                forgetting_scores.append(forgetting)
                
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
        
    def get_continual_learning_state(self) -> Dict[str, Any]:
        """Get current continual learning state."""
        state = {
            'strategy': self.strategy,
            'model_state': self.model.state_dict(),
            'task_history': self.task_history,
            'performance_history': dict(self.performance_history)
        }
        
        if hasattr(self, 'ewc'):
            state['ewc_state'] = {
                'fisher_info': self.ewc.fisher_info,
                'optimal_params': self.ewc.optimal_params,
                'task_count': self.ewc.task_count
            }
        elif hasattr(self, 'progressive_net'):
            state['progressive_state'] = {
                'num_tasks': self.progressive_net.num_tasks,
                'current_task': self.progressive_net.current_task
            }
        elif hasattr(self, 'memory_buffer'):
            state['buffer_state'] = {
                'buffer': self.memory_buffer.buffer,
                'sample_count': self.memory_buffer.sample_count
            }
        elif hasattr(self, 'old_model'):
            state['old_model_state'] = self.old_model.state_dict()
            
        return state
        
    def load_continual_learning_state(self, state: Dict[str, Any]) -> None:
        """Load continual learning state."""
        self.model.load_state_dict(state['model_state'])
        self.task_history = state.get('task_history', [])
        self.performance_history = defaultdict(list, state.get('performance_history', {}))
        
        if 'ewc_state' in state and hasattr(self, 'ewc'):
            self.ewc.fisher_info = state['ewc_state']['fisher_info']
            self.ewc.optimal_params = state['ewc_state']['optimal_params']
            self.ewc.task_count = state['ewc_state']['task_count']
        elif 'progressive_state' in state and hasattr(self, 'progressive_net'):
            self.progressive_net.num_tasks = state['progressive_state']['num_tasks']
            self.progressive_net.current_task = state['progressive_state']['current_task']
        elif 'buffer_state' in state and hasattr(self, 'memory_buffer'):
            self.memory_buffer.buffer = state['buffer_state']['buffer']
            self.memory_buffer.sample_count = state['buffer_state']['sample_count']
        elif 'old_model_state' in state and hasattr(self, 'old_model'):
            self.old_model.load_state_dict(state['old_model_state'])


def create_continual_learning_benchmark(
    tasks: List[Dict[str, Any]],
    strategy: str = 'ewc',
    buffer_size: int = 1000
) -> Dict[str, Any]:
    """
    Create a continual learning benchmark.
    
    Args:
        tasks: List of task dictionaries
        strategy: Continual learning strategy
        buffer_size: Size of memory buffer
        
    Returns:
        Benchmark configuration
    """
    return {
        'tasks': tasks,
        'strategy': strategy,
        'buffer_size': buffer_size,
        'evaluation_metrics': [
            'accuracy',
            'forgetting_measure',
            'transfer_measure',
            'backward_transfer'
        ],
        'expected_behavior': {
            'ewc': 'Prevents catastrophic forgetting through parameter regularization',
            'progressive': 'Learns new tasks without forgetting via column expansion',
            'replay': 'Maintains memory of previous tasks through data replay',
            'lwf': 'Preserves old task performance through knowledge distillation'
        }
    }