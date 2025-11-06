import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import random
from collections import namedtuple


class NASOperation(Enum):
    """Neural Architecture Search operation types."""
    
    # Basic operations
    IDENTITY = "identity"
    SKIP_CONNECT = "skip_connect"
    
    # Convolutional operations
    CONV_1x1 = "conv_1x1"
    CONV_3x3 = "conv_3x3"
    CONV_5x5 = "conv_5x5"
    CONV_7x7 = "conv_7x7"
    
    # Pooling operations
    AVG_POOL_3x3 = "avg_pool_3x3"
    MAX_POOL_3x3 = "max_pool_3x3"
    
    # Activation operations
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    ELU = "elu"
    LEAKY_RELU = "leaky_relu"


class SearchSpace:
    """Defines the search space for neural architecture search."""
    
    def __init__(self, num_nodes: int = 4, operations: List[NASOperation] = None):
        """
        Initialize search space.
        
        Args:
            num_nodes: Number of nodes in the architecture
            operations: List of allowed operations
        """
        self.num_nodes = num_nodes
        self.operations = operations or [
            NASOperation.CONV_1x1,
            NASOperation.CONV_3x3,
            NASOperation.CONV_5x5,
            NASOperation.AVG_POOL_3x3,
            NASOperation.MAX_POOL_3x3,
            NASOperation.RELU,
            NASOperation.IDENTITY,
            NASOperation.SKIP_CONNECT
        ]
        
    def get_possible_edges(self) -> List[Tuple[int, int]]:
        """Get all possible edges in the DAG."""
        edges = []
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                edges.append((i, j))
        return edges
    
    def random_architecture(self) -> Dict[str, Any]:
        """Generate a random architecture."""
        edges = self.get_possible_edges()
        operations = {}
        
        for edge in edges:
            operations[edge] = random.choice(self.operations)
        
        return {
            'num_nodes': self.num_nodes,
            'operations': operations
        }


class DARTSCell(nn.Module):
    """Differentiable Architecture Search (DARTS) cell."""
    
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 4):
        """
        Initialize DARTS cell.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels  
            num_nodes: Number of intermediate nodes
        """
        super(DARTSCell, self).__init__()
        
        self.num_nodes = num_nodes
        self.out_channels = out_channels
        
        # Edge operations - each edge has a set of possible operations
        self.edge_ops = nn.ModuleDict()
        
        # Operation candidates
        ops = [
            lambda: nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            lambda: nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 5, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            lambda: nn.AvgPool2d(3, padding=1),
            lambda: nn.MaxPool2d(3, padding=1),
            lambda: nn.Identity()
        ]
        
        # Initialize edge operations for each possible edge
        for i in range(num_nodes + 2):  # +2 for input and output nodes
            for j in range(i + 1, num_nodes + 2):
                edge_key = f"{i}_{j}"
                edge_operations = nn.ModuleList([op() for op in ops])
                self.edge_ops[edge_key] = edge_operations
        
        # Architecture parameters (softmax weights for each edge)
        self.alphas = nn.Parameter(torch.randn(num_nodes + 2, num_nodes + 2, len(ops)))
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the cell."""
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        
        # Start with input nodes
        nodes = [0] * (self.num_nodes + 2)
        nodes[0] = inputs[0]  # First input
        if len(inputs) > 1:
            nodes[1] = inputs[1]  # Second input
        
        # Compute node values
        for i in range(2, self.num_nodes + 2):
            # Sum contributions from all previous nodes
            node_sum = 0
            for j in range(i):
                edge_key = f"{j}_{i}"
                if edge_key in self.edge_ops:
                    # Apply softmax to get weights
                    alpha = F.softmax(self.alphas[j, i], dim=0)
                    
                    # Apply operations and weight them
                    edge_ops = self.edge_ops[edge_key]
                    for k, op in enumerate(edge_ops):
                        if k < len(alpha):
                            edge_output = op(nodes[j])
                            node_sum += alpha[k] * edge_output
            
            nodes[i] = node_sum
        
        # Concatenate output nodes
        return torch.cat([nodes[i] for i in range(2, self.num_nodes + 2)], dim=1)


class NASOptimizer:
    """Neural Architecture Search optimizer with multiple strategies."""
    
    def __init__(self, 
                 search_space: SearchSpace,
                 dataset: Any,
                 evaluation_fn: callable,
                 strategy: str = "darts"):
        """
        Initialize NAS optimizer.
        
        Args:
            search_space: Search space definition
            dataset: Training dataset
            evaluation_fn: Function to evaluate architecture performance
            strategy: Search strategy ("darts", "ea", "random")
        """
        self.search_space = search_space
        self.dataset = dataset
        self.evaluation_fn = evaluation_fn
        self.strategy = strategy
        
        self.population = []
        self.generation = 0
        
    def search(self, 
               max_generations: int = 50,
               population_size: int = 20,
               mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Perform neural architecture search.
        
        Args:
            max_generations: Maximum number of generations
            population_size: Size of population for evolutionary search
            mutation_rate: Rate of mutation
            
        Returns:
            Best architecture found
        """
        if self.strategy == "random":
            return self._random_search(max_generations)
        elif self.strategy == "ea":
            return self._evolutionary_search(max_generations, population_size, mutation_rate)
        elif self.strategy == "darts":
            return self._darts_search(max_generations)
        else:
            raise ValueError(f"Unknown search strategy: {self.strategy}")
    
    def _random_search(self, max_iterations: int) -> Dict[str, Any]:
        """Perform random search."""
        best_architecture = None
        best_score = float('-inf')
        
        for _ in range(max_iterations):
            architecture = self.search_space.random_architecture()
            score = self.evaluation_fn(architecture)
            
            if score > best_score:
                best_score = score
                best_architecture = architecture
        
        return best_architecture
    
    def _evolutionary_search(self, 
                           max_generations: int,
                           population_size: int,
                           mutation_rate: float) -> Dict[str, Any]:
        """Perform evolutionary search."""
        # Initialize population
        self.population = [
            {
                'architecture': self.search_space.random_architecture(),
                'fitness': 0.0
            }
            for _ in range(population_size)
        ]
        
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(max_generations):
            # Evaluate fitness
            for individual in self.population:
                if individual['fitness'] == 0.0:  # Not evaluated yet
                    individual['fitness'] = self.evaluation_fn(individual['architecture'])
            
            # Find best individual
            current_best = max(self.population, key=lambda x: x['fitness'])
            if current_best['fitness'] > best_fitness:
                best_fitness = current_best['fitness']
                best_individual = current_best['architecture'].copy()
            
            # Selection and reproduction
            self.population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Keep top 50% and create offspring
            survivors = self.population[:population_size // 2]
            new_population = survivors.copy()
            
            while len(new_population) < population_size:
                # Selection
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Crossover
                child = self._crossover(parent1['architecture'], parent2['architecture'])
                
                # Mutation
                child = self._mutate(child, mutation_rate)
                
                new_population.append({
                    'architecture': child,
                    'fitness': 0.0  # Will be evaluated next generation
                })
            
            self.population = new_population
            self.generation = generation
        
        return best_individual
    
    def _darts_search(self, max_epochs: int) -> Dict[str, Any]:
        """Perform DARTS search."""
        # DARTS implementation would be more complex
        # This is a simplified version
        best_architecture = None
        best_score = float('-inf')
        
        # Initialize DARTS cell
        darts_cell = DARTSCell(32, 64)  # Example dimensions
        optimizer = torch.optim.Adam(darts_cell.parameters(), lr=0.001)
        
        for epoch in range(max_epochs):
            # Train DARTS cell
            for batch in self.dataset:
                optimizer.zero_grad()
                
                outputs = darts_cell(batch)
                loss = F.mse_loss(outputs, targets(batch))  # Simplified loss
                
                loss.backward()
                optimizer.step()
            
            # Extract architecture from trained cell
            architecture = self._extract_architecture(darts_cell)
            score = self.evaluation_fn(architecture)
            
            if score > best_score:
                best_score = score
                best_architecture = architecture
        
        return best_architecture
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two architectures."""
        child = parent1.copy()
        operations1 = parent1['operations']
        operations2 = parent2['operations']
        
        for edge, op in operations2.items():
            if random.random() < 0.5:
                child['operations'][edge] = op
        
        return child
    
    def _mutate(self, architecture: Dict[str, Any], mutation_rate: float) -> Dict[str, Any]:
        """Mutate an architecture."""
        mutated = architecture.copy()
        
        for edge in mutated['operations']:
            if random.random() < mutation_rate:
                mutated['operations'][edge] = random.choice(self.search_space.operations)
        
        return mutated
    
    def _extract_architecture(self, darts_cell: DARTSCell) -> Dict[str, Any]:
        """Extract architecture from trained DARTS cell."""
        operations = {}
        
        for i in range(darts_cell.num_nodes + 2):
            for j in range(i + 1, darts_cell.num_nodes + 2):
                edge_key = f"{i}_{j}"
                if edge_key in darts_cell.alphas:
                    # Get most likely operation
                    alpha = F.softmax(darts_cell.alphas[i, j], dim=0)
                    max_op_idx = torch.argmax(alpha).item()
                    
                    # Map back to operation enum
                    operation_mapping = {
                        0: NASOperation.CONV_3x3,
                        1: NASOperation.CONV_5x5,
                        2: NASOperation.AVG_POOL_3x3,
                        3: NASOperation.MAX_POOL_3x3,
                        4: NASOperation.IDENTITY
                    }
                    
                    operations[(i, j)] = operation_mapping.get(max_op_idx, NASOperation.IDENTITY)
        
        return {
            'num_nodes': darts_cell.num_nodes,
            'operations': operations
        }


class NeuralArchitectureSearch:
    """Main Neural Architecture Search framework."""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],
                 num_classes: int,
                 search_space: SearchSpace = None):
        """
        Initialize NAS framework.
        
        Args:
            input_shape: Input tensor shape (C, H, W)
            num_classes: Number of output classes
            search_space: Custom search space
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.search_space = search_space or SearchSpace()
        
    def build_model(self, architecture: Dict[str, Any]) -> nn.Module:
        """
        Build model from architecture specification.
        
        Args:
            architecture: Architecture specification
            
        Returns:
            PyTorch model
        """
        class NASModel(nn.Module):
            def __init__(self, input_shape, num_classes, architecture):
                super(NASModel, self).__init__()
                self.input_shape = input_shape
                self.num_classes = num_classes
                self.architecture = architecture
                
                # Build cells based on architecture
                self.cells = nn.ModuleList()
                
                channels = input_shape[0]
                for _ in range(3):  # Example: 3 cells
                    cell = DARTSCell(channels, channels * 2, architecture['num_nodes'])
                    self.cells.append(cell)
                    channels *= 2
                
                # Global pooling and classification head
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                self.classifier = nn.Linear(channels, num_classes)
            
            def forward(self, x):
                # Process through cells
                for cell in self.cells:
                    x = cell([x, x])  # DARTS expects list of inputs
                
                # Global pooling and classification
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                
                return x
        
        return NASModel(self.input_shape, self.num_classes, architecture)
    
    def search(self, 
               dataset: Any,
               strategy: str = "darts",
               max_generations: int = 50,
               evaluation_fn: callable = None) -> Dict[str, Any]:
        """
        Perform complete NAS process.
        
        Args:
            dataset: Training dataset
            strategy: Search strategy
            max_generations: Maximum generations/iterations
            evaluation_fn: Custom evaluation function
            
        Returns:
            Best architecture found
        """
        if evaluation_fn is None:
            evaluation_fn = self._default_evaluation
        
        optimizer = NASOptimizer(
            search_space=self.search_space,
            dataset=dataset,
            evaluation_fn=evaluation_fn,
            strategy=strategy
        )
        
        best_architecture = optimizer.search(
            max_generations=max_generations
        )
        
        return best_architecture
    
    def _default_evaluation(self, architecture: Dict[str, Any]) -> float:
        """
        Default architecture evaluation function.
        
        Args:
            architecture: Architecture to evaluate
            
        Returns:
            Performance score
        """
        try:
            # Build and quickly train the model
            model = self.build_model(architecture)
            
            # Simple training for quick evaluation
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Train for a few epochs (simplified)
            for _ in range(5):
                # Assume we have a way to get training data
                # This is a simplified version - in practice you'd use real data
                pass
            
            # Return a dummy score for now
            # In practice, you'd evaluate on validation set
            return 0.5 + random.random() * 0.5
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0


# Utility functions
def targets(batch):
    """Generate dummy targets for testing."""
    return torch.randn(batch.size(0), 64)  # Simplified target generation


def accuracy_score(predictions, targets):
    """Calculate accuracy score."""
    if len(predictions.shape) > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    if len(targets.shape) > 1:
        targets = torch.argmax(targets, dim=1)
    
    correct = (predictions == targets).float()
    return correct.mean().item()


def visualize_architecture(architecture: Dict[str, Any], save_path: str = None):
    """
    Visualize architecture as a directed acyclic graph.
    
    Args:
        architecture: Architecture specification
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add nodes
    num_nodes = architecture['num_nodes']
    for i in range(num_nodes + 2):  # +2 for input and output
        G.add_node(i, label=f'Node {i}' if i < num_nodes else f'Input/Output')
    
    # Add edges with operations
    operations = architecture['operations']
    for (i, j), op in operations.items():
        G.add_edge(i, j, label=op.value)
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=1000, alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20)
    
    # Draw labels
    labels = {node: f'Node {node}' for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    # Draw edge labels (operations)
    edge_labels = {(i, j): op.value for (i, j), op in operations.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Neural Architecture Graph")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compare_architectures(archs: List[Dict[str, Any]], 
                         metrics: List[str] = None) -> pd.DataFrame:
    """
    Compare multiple architectures.
    
    Args:
        archs: List of architectures to compare
        metrics: List of metrics to calculate
        
    Returns:
        DataFrame with comparison results
    """
    import pandas as pd
    
    if metrics is None:
        metrics = ['accuracy', 'flops', 'parameters', 'latency']
    
    results = []
    
    for i, arch in enumerate(archs):
        row = {'architecture_id': i}
        
        for metric in metrics:
            if metric == 'accuracy':
                # Dummy accuracy calculation
                row[metric] = 0.5 + random.random() * 0.5
            elif metric == 'flops':
                # Estimate FLOPs based on architecture
                row[metric] = 1e6 + random.random() * 9e6
            elif metric == 'parameters':
                # Estimate parameters
                row[metric] = 1e5 + random.random() * 9e5
            elif metric == 'latency':
                # Estimate inference time
                row[metric] = 0.01 + random.random() * 0.09
        
        results.append(row)
    
    return pd.DataFrame(results)


# Example usage and testing
if __name__ == "__main__":
    # Create search space
    search_space = SearchSpace(num_nodes=4)
    
    # Create NAS instance
    nas = NeuralArchitectureSearch(
        input_shape=(3, 32, 32),
        num_classes=10
    )
    
    # Perform search
    print("Starting Neural Architecture Search...")
    best_arch = nas.search(
        dataset=None,  # Would be real dataset
        strategy="random",
        max_generations=10
    )
    
    print(f"Best architecture found: {best_arch}")
    
    # Build model from architecture
    model = nas.build_model(best_arch)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test model forward pass
    x = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Visualize architecture
    visualize_architecture(best_arch)
    
    # Compare multiple architectures
    random_architectures = [search_space.random_architecture() for _ in range(5)]
    comparison_df = compare_architectures(random_architectures)
    print("Architecture Comparison:")
    print(comparison_df)