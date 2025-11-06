"""
Few-Shot Learning Implementation
Prototypical networks and relation networks for classification with limited examples.

This module provides state-of-the-art few-shot learning algorithms suitable
for rapid adaptation with minimal training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod
import math


class FewShotDataset(Dataset):
    """Dataset for few-shot learning experiments."""
    
    def __init__(
        self, 
        data: torch.Tensor, 
        labels: torch.Tensor,
        n_way: int,
        n_support: int,
        n_query: int,
        episodes: int,
        transform=None
    ):
        """
        Initialize few-shot dataset.
        
        Args:
            data: Input data of shape (num_samples, ...)
            labels: Labels of shape (num_samples,)
            n_way: Number of classes per episode
            n_support: Number of support examples per class
            n_query: Number of query examples per class
            episodes: Number of episodes to generate
            transform: Data transformations
        """
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.episodes = episodes
        self.transform = transform
        
        self.classes = torch.unique(labels).tolist()
        
    def __len__(self):
        return self.episodes
        
    def __getitem__(self, idx):
        """Generate one episode."""
        # Sample classes
        selected_classes = np.random.choice(
            self.classes, size=self.n_way, replace=False
        )
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for i, cls in enumerate(selected_classes):
            # Get all examples of this class
            cls_indices = torch.where(self.labels == cls)[0]
            
            # Sample support and query sets
            support_indices = np.random.choice(
                cls_indices, size=self.n_support, replace=False
            )
            remaining_indices = np.setdiff1d(cls_indices, support_indices)
            
            query_indices = np.random.choice(
                remaining_indices, size=self.n_query, replace=False
            )
            
            # Add to episode
            support_x.extend(self.data[support_indices])
            support_y.extend([i] * self.n_support)
            query_x.extend(self.data[query_indices])
            query_y.extend([i] * self.n_query)
            
        # Convert to tensors
        support_x = torch.stack(support_x)
        support_y = torch.tensor(support_y)
        query_x = torch.stack(query_x)
        query_y = torch.tensor(query_y)
        
        # Apply transformations
        if self.transform:
            support_x = self.transform(support_x)
            query_x = self.transform(query_x)
            
        return {
            'support_x': support_x,
            'support_y': support_y,
            'query_x': query_x,
            'query_y': query_y
        }


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for Few-Shot Learning.
    
    Implementation of the classical Prototypical Networks algorithm
    for few-shot classification.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        distance_metric: str = 'euclidean',  # 'euclidean', 'cosine'
        learn_distance: bool = False,
        temperature: float = 1.0
    ):
        """
        Initialize Prototypical Network.
        
        Args:
            encoder: Feature encoder network
            distance_metric: Distance metric to use
            learn_distance: Whether to learn distance metric
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.encoder = encoder
        self.distance_metric = distance_metric
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Learnable distance scaling
        self.learn_distance = learn_distance
        if learn_distance:
            self.distance_scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('distance_scale', torch.ones(1))
            
    def encode_support(self, support_x: torch.Tensor) -> torch.Tensor:
        """Encode support set to feature space."""
        return self.encoder(support_x)
        
    def encode_query(self, query_x: torch.Tensor) -> torch.Tensor:
        """Encode query set to feature space."""
        return self.encoder(query_x)
        
    def compute_prototypes(
        self, 
        support_x: torch.Tensor, 
        support_y: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            n_way: Number of classes
            
        Returns:
            Class prototypes of shape (n_way, feature_dim)
        """
        # Encode support set
        features = self.encode_support(support_x)
        
        # Compute prototypes
        prototypes = []
        for class_idx in range(n_way):
            class_mask = (support_y == class_idx)
            class_features = features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
            
        return torch.stack(prototypes)
        
    def forward(
        self, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels  
            query_x: Query set inputs
            query_y: Query set labels (for loss computation)
            
        Returns:
            Logits for query predictions (and loss if query_y provided)
        """
        n_way = len(torch.unique(support_y))
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_x, support_y, n_way)
        
        # Encode query set
        query_features = self.encode_query(query_x)
        
        # Compute distances
        distances = self.compute_distances(query_features, prototypes)
        
        # Convert to logits
        logits = -distances * self.distance_scale / self.temperature
        
        if query_y is not None:
            loss = F.cross_entropy(logits, query_y)
            return logits, loss
        else:
            return logits
            
    def compute_distances(
        self, 
        query_features: torch.Tensor, 
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances between query features and prototypes."""
        if self.distance_metric == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_features, prototypes, p=2)
        elif self.distance_metric == 'cosine':
            # Cosine distance
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            cos_sim = torch.mm(query_norm, proto_norm.t())
            distances = 1 - cos_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        return distances
        
    def predict(self, support_x: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        """Make predictions for query set."""
        return self.forward(support_x, None, query_x)


class RelationNetwork(nn.Module):
    """
    Relation Networks for Few-Shot Learning.
    
    Implementation that learns a relation function to compare
    query examples with support examples.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        relation_head: Optional[nn.Module] = None,
        temperature: float = 1.0
    ):
        """
        Initialize Relation Network.
        
        Args:
            encoder: Feature encoder for input images
            relation_head: Relation function network
            temperature: Temperature parameter
        """
        super().__init__()
        self.encoder = encoder
        
        # Default relation head if not provided
        if relation_head is None:
            relation_head = nn.Sequential(
                nn.Linear(encoder.out_features * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
        self.relation_head = relation_head
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space."""
        return self.encoder(x)
        
    def compute_relations(
        self,
        support_x: torch.Tensor,
        query_x: torch.Tensor,
        support_y: torch.Tensor,
        n_way: int,
        n_support: int
    ) -> torch.Tensor:
        """
        Compute relations between query and support sets.
        
        Args:
            support_x: Support set inputs
            query_x: Query set inputs
            support_y: Support set labels
            n_way: Number of classes
            n_support: Number of support examples per class
            
        Returns:
            Relation scores for each query-class pair
        """
        # Encode all inputs
        support_features = self.encode(support_x)
        query_features = self.encode(query_x)
        
        # Reshape for relation computation
        n_query = query_features.shape[0]
        
        # Replicate support features for each query
        support_repeated = support_features.unsqueeze(0).repeat(n_query, 1, 1)
        query_repeated = query_features.unsqueeze(1).repeat(1, support_repeated.shape[1], 1)
        
        # Concatenate features for relation computation
        relations_input = torch.cat([support_repeated, query_repeated], dim=2)
        
        # Compute relations
        relations = self.relation_head(relations_input).squeeze(-1)
        
        # Reshape to group by query examples
        relations = relations.view(n_query, n_way, n_support)
        
        # Average relations across support examples
        relation_scores = relations.mean(dim=2)
        
        return relation_scores / self.temperature
        
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: Optional[torch.Tensor] = None,
        n_way: int = 5,
        n_support: int = 1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs  
            query_y: Query set labels (for loss computation)
            n_way: Number of classes
            n_support: Number of support examples per class
            
        Returns:
            Relation scores (and loss if query_y provided)
        """
        # Compute relations
        relation_scores = self.compute_relations(
            support_x, query_x, support_y, n_way, n_support
        )
        
        if query_y is not None:
            loss = F.cross_entropy(relation_scores, query_y)
            return relation_scores, loss
        else:
            return relation_scores
            
    def predict(
        self, 
        support_x: torch.Tensor, 
        query_x: torch.Tensor,
        n_way: int = 5,
        n_support: int = 1
    ) -> torch.Tensor:
        """Make predictions for query set."""
        return self.forward(support_x, None, query_x, None, n_way, n_support)


class MatchingNetwork(nn.Module):
    """
    Matching Networks for Few-Shot Learning.
    
    Attention-based few-shot learning with memory components.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        attention_function: str = 'cosine'  # 'cosine', 'euclidean', 'learned'
    ):
        """
        Initialize Matching Network.
        
        Args:
            encoder: Feature encoder
            attention_function: Attention function type
        """
        super().__init__()
        self.encoder = encoder
        self.attention_function = attention_function
        
        # Learnable attention parameters
        if attention_function == 'learned':
            self.attention_net = nn.Sequential(
                nn.Linear(encoder.out_features * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            
    def encode_support(self, support_x: torch.Tensor) -> torch.Tensor:
        """Encode support set."""
        return self.encoder(support_x)
        
    def encode_query(self, query_x: torch.Tensor) -> torch.Tensor:
        """Encode query set."""
        return self.encoder(query_x)
        
    def compute_attention(
        self, 
        query_features: torch.Tensor, 
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention weights using cosine similarity.
        
        Args:
            query_features: Query set features
            support_features: Support set features
            support_labels: Support set labels (one-hot encoded)
            
        Returns:
            Attention-weighted class predictions
        """
        if self.attention_function == 'cosine':
            # Cosine similarity
            query_norm = F.normalize(query_features, p=2, dim=1)
            support_norm = F.normalize(support_features, p=2, dim=1)
            attention = torch.mm(query_norm, support_norm.t())
            
        elif self.attention_function == 'euclidean':
            # Euclidean similarity (negative distance)
            distances = torch.cdist(query_features, support_features, p=2)
            attention = -distances
            
        elif self.attention_function == 'learned':
            # Learnable attention
            n_query = query_features.shape[0]
            n_support = support_features.shape[0]
            
            query_repeated = query_features.unsqueeze(1).repeat(1, n_support, 1)
            support_repeated = support_features.unsqueeze(0).repeat(n_query, 1, 1)
            
            concat_features = torch.cat([query_repeated, support_repeated], dim=2)
            attention = self.attention_net(concat_features).squeeze(-1)
            
        else:
            raise ValueError(f"Unknown attention function: {self.attention_function}")
            
        # Convert attention to probabilities
        attention = F.softmax(attention, dim=1)
        
        return attention
        
    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the network.
        """
        # Encode features
        support_features = self.encode_support(support_x)
        query_features = self.encode_query(query_x)
        
        # One-hot encode support labels
        n_way = len(torch.unique(support_y))
        support_labels_onehot = F.one_hot(support_y, n_way).float()
        
        # Compute attention
        attention = self.compute_attention(
            query_features, support_features, support_labels_onehot
        )
        
        # Weight support labels by attention
        predictions = torch.mm(attention, support_labels_onehot)
        
        if query_y is not None:
            loss = F.binary_cross_entropy(predictions, support_labels_onehot[query_y])
            return predictions, loss
        else:
            return predictions
            
    def predict(self, support_x: torch.Tensor, query_x: torch.Tensor) -> torch.Tensor:
        """Make predictions for query set."""
        return self.forward(support_x, None, query_x)


class FewShotLearner:
    """
    Unified few-shot learning interface supporting multiple algorithms.
    """
    
    def __init__(
        self,
        algorithm: str = 'prototypical',  # 'prototypical', 'relation', 'matching'
        encoder: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Initialize Few-Shot Learner.
        
        Args:
            algorithm: Learning algorithm to use
            encoder: Feature encoder network
            **kwargs: Algorithm-specific parameters
        """
        self.algorithm = algorithm
        self.encoder = encoder
        
        # Initialize algorithm
        if algorithm == 'prototypical':
            self.model = PrototypicalNetwork(encoder, **kwargs)
        elif algorithm == 'relation':
            self.model = RelationNetwork(encoder, **kwargs)
        elif algorithm == 'matching':
            self.model = MatchingNetwork(encoder, **kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.training_history = {'loss': [], 'accuracy': []}
        
    def train_episode(
        self, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train on one episode.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            
        Returns:
            Training metrics
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        if self.algorithm in ['prototypical', 'relation']:
            logits, loss = self.model(
                support_x, support_y, query_x, query_y
            )
            predictions = torch.argmax(logits, dim=1)
        else:  # matching
            predictions, loss = self.model(
                support_x, support_y, query_x, query_y
            )
            
        # Compute accuracy
        accuracy = (predictions == query_y).float().mean()
        
        # Backpropagation
        loss.backward()
        self.optimizer.step()
        
        # Update history
        self.training_history['loss'].append(loss.item())
        self.training_history['accuracy'].append(accuracy.item())
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
        
    def evaluate_episode(
        self, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate on one episode.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            query_y: Query set labels
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            if self.algorithm in ['prototypical', 'relation']:
                logits = self.model(support_x, support_y, query_x)
                predictions = torch.argmax(logits, dim=1)
                loss = F.cross_entropy(logits, query_y)
            else:  # matching
                predictions = self.model(support_x, support_y, query_x)
                # Convert to class predictions
                class_predictions = torch.argmax(predictions, dim=1)
                loss = F.cross_entropy(predictions, query_y)
                
            accuracy = (class_predictions if self.algorithm == 'matching' else predictions == query_y).float().mean()
            
        self.model.train()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
        
    def predict(
        self, 
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        n_way: int = 5,
        n_support: int = 1
    ) -> torch.Tensor:
        """
        Make predictions for query set.
        
        Args:
            support_x: Support set inputs
            support_y: Support set labels
            query_x: Query set inputs
            n_way: Number of classes
            n_support: Number of support examples
            
        Returns:
            Class predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            if self.algorithm == 'prototypical':
                logits = self.model(support_x, support_y, query_x)
                predictions = torch.argmax(logits, dim=1)
            elif self.algorithm == 'relation':
                predictions = self.model.predict(
                    support_x, query_x, n_way, n_support
                )
                predictions = torch.argmax(predictions, dim=1)
            else:  # matching
                predictions = self.model.predict(support_x, query_x)
                predictions = torch.argmax(predictions, dim=1)
                
        self.model.train()
        
        return predictions
        
    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_history': self.training_history.copy(),
            'algorithm': self.algorithm
        }
        
    def load_training_state(self, state: Dict[str, Any]) -> None:
        """Load training state."""
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.training_history = state.get('training_history', {'loss': [], 'accuracy': []})


def create_cosine_distance_matrix(
    features: torch.Tensor, 
    prototypes: torch.Tensor
) -> torch.Tensor:
    """Create cosine distance matrix between features and prototypes."""
    features_norm = F.normalize(features, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    cos_sim = torch.mm(features_norm, prototypes_norm.t())
    return 1 - cos_sim


def create_euclidean_distance_matrix(
    features: torch.Tensor, 
    prototypes: torch.Tensor
) -> torch.Tensor:
    """Create euclidean distance matrix between features and prototypes."""
    return torch.cdist(features, prototypes, p=2)


def evaluate_few_shot_learner(
    learner: FewShotLearner,
    test_episodes: List[Dict[str, torch.Tensor]],
    n_way: int,
    n_support: int,
    n_query: int
) -> Dict[str, float]:
    """
    Evaluate few-shot learner on multiple test episodes.
    
    Args:
        learner: Trained few-shot learner
        test_episodes: List of test episode dictionaries
        n_way: Number of classes per episode
        n_support: Number of support examples per class
        n_query: Number of query examples per class
        
    Returns:
        Evaluation metrics
    """
    accuracies = []
    losses = []
    
    for episode in test_episodes:
        metrics = learner.evaluate_episode(
            episode['support_x'],
            episode['support_y'],
            episode['query_x'],
            episode['query_y']
        )
        accuracies.append(metrics['accuracy'])
        losses.append(metrics['loss'])
        
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_loss': np.mean(losses),
        'std_loss': np.std(losses),
        'num_episodes': len(test_episodes),
        'confidence_interval_95': 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
    }