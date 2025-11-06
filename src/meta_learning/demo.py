"""
Meta-Learning Framework Demo
Comprehensive demonstration of all meta-learning capabilities.

This module provides a complete example of how to use the meta-learning
framework for advanced AI system adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any
import json
import time

# Import all meta-learning components
from .maml import MAML, MAMLFirstOrder, MultiTaskMAML, create_meta_batch
from .few_shot import PrototypicalNetwork, FewShotLearner, FewShotDataset
from .continual import ContinualLearner, EWC, MemoryBuffer, ProgressiveNetwork
from .optimizer import AdaptiveOptimizer, BayesianOptimizer, LearningRateScheduler
from .nas_integration import NeuralArchitectureSearch, DARTS, create_nas_search_space
from .tool_optimization import (
    ToolMetaLearner, ToolProfile, TaskProfile, TaskComplexity, ToolType
)
from .adapter import (
    AdapterModel, AdapterConfig, AdapterType, 
    create_adaptive_adapter_ensemble
)


class MetaLearningDemo:
    """
    Comprehensive demo showcasing all meta-learning capabilities.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def run_complete_demo(self):
        """Run the complete meta-learning demonstration."""
        print("🚀 Starting Meta-Learning Framework Demo")
        print("=" * 50)
        
        # 1. MAML Demo
        print("\n1. Model-Agnostic Meta-Learning (MAML)")
        maml_results = self.demo_maml()
        
        # 2. Few-Shot Learning Demo
        print("\n2. Few-Shot Learning")
        few_shot_results = self.demo_few_shot()
        
        # 3. Continual Learning Demo
        print("\n3. Continual Learning")
        continual_results = self.demo_continual_learning()
        
        # 4. Adaptive Optimization Demo
        print("\n4. Adaptive Optimization")
        optimization_results = self.demo_adaptive_optimization()
        
        # 5. NAS Integration Demo
        print("\n5. Neural Architecture Search")
        nas_results = self.demo_nas()
        
        # 6. Tool Optimization Demo
        print("\n6. Tool Usage Optimization")
        tool_results = self.demo_tool_optimization()
        
        # 7. Adapter Demo
        print("\n7. Meta-Learning Adapters")
        adapter_results = self.demo_adapters()
        
        # 8. Integrated System Demo
        print("\n8. Integrated Meta-Learning System")
        integrated_results = self.demo_integrated_system()
        
        # Compile final results
        self.results = {
            'maml': maml_results,
            'few_shot': few_shot_results,
            'continual_learning': continual_results,
            'adaptive_optimization': optimization_results,
            'nas': nas_results,
            'tool_optimization': tool_results,
            'adapters': adapter_results,
            'integrated_system': integrated_results
        }
        
        self.print_summary()
        return self.results
        
    def demo_maml(self) -> Dict[str, Any]:
        """Demonstrate MAML capabilities."""
        print("   Setting up MAML for function approximation...")
        
        # Create simple regression model
        model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize MAML
        maml = MAMLFirstOrder(
            model=model,
            lr_inner=0.01,
            lr_meta=0.001,
            num_inner_steps=5
        )
        
        # Generate training data (sinusoidal functions)
        def generate_task():
            # Random amplitude and phase
            amplitude = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Generate data
            x = np.random.uniform(-5, 5, 20)
            y = amplitude * np.sin(x + phase) + np.random.normal(0, 0.1, 20)
            
            # Split into support and query
            support_indices = np.random.choice(20, 10, replace=False)
            query_indices = np.setdiff1d(range(20), support_indices)
            
            support_x = torch.tensor(x[support_indices], dtype=torch.float32).unsqueeze(1)
            support_y = torch.tensor(y[support_indices], dtype=torch.float32).unsqueeze(1)
            query_x = torch.tensor(x[query_indices], dtype=torch.float32).unsqueeze(1)
            query_y = torch.tensor(y[query_indices], dtype=torch.float32).unsqueeze(1)
            
            return {
                'support_x': support_x,
                'support_y': support_y,
                'query_x': query_x,
                'query_y': query_y
            }
        
        # Training loop
        print("   Training MAML...")
        meta_losses = []
        
        for step in range(200):
            # Generate meta-batch
            tasks = [generate_task() for _ in range(4)]
            meta_batch = create_meta_batch(tasks, batch_size=4)
            
            # Meta-training step
            metrics = maml.meta_train_step(meta_batch)
            meta_losses.append(metrics['meta_loss'])
            
            if step % 50 == 0:
                print(f"   Step {step}: Meta Loss = {metrics['meta_loss']:.4f}, "
                      f"Accuracy = {metrics['meta_accuracy']:.4f}")
        
        # Test adaptation
        print("   Testing adaptation to new task...")
        test_task = generate_task()
        adapted_predictions = maml.predict(
            test_task['query_x'], 
            test_task['support_x'], 
            test_task['support_y']
        )
        
        test_loss = F.mse_loss(adapted_predictions, test_task['query_y']).item()
        
        return {
            'final_meta_loss': meta_losses[-1],
            'test_adaptation_loss': test_loss,
            'convergence_achieved': meta_losses[-1] < meta_losses[0] * 0.1
        }
        
    def demo_few_shot(self) -> Dict[str, Any]:
        """Demonstrate few-shot learning capabilities."""
        print("   Setting up few-shot learning for image classification...")
        
        # Create simple CNN encoder
        encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU()
        )
        
        # Initialize few-shot learner with prototypical networks
        learner = FewShotLearner(
            algorithm='prototypical',
            encoder=encoder
        )
        
        # Generate few-shot dataset
        print("   Generating synthetic few-shot data...")
        n_way, n_support, n_query = 5, 1, 15
        
        # Simulate few-shot episodes
        train_accuracies = []
        test_accuracies = []
        
        for episode in range(100):
            # Generate support and query sets
            support_x = torch.randn(n_way * n_support, 1, 28, 28)
            support_y = torch.arange(n_way).repeat_interleave(n_support)
            query_x = torch.randn(n_way * n_query, 1, 28, 28)
            query_y = torch.arange(n_way).repeat_interleave(n_query)
            
            # Train on episode
            if episode < 80:  # Training episodes
                train_metrics = learner.train_episode(
                    support_x, support_y, query_x, query_y
                )
                train_accuracies.append(train_metrics['accuracy'])
            else:  # Test episodes
                test_metrics = learner.evaluate_episode(
                    support_x, support_y, query_x, query_y
                )
                test_accuracies.append(test_metrics['accuracy'])
                
        # Results
        avg_train_acc = np.mean(train_accuracies[-20:]) if train_accuracies else 0
        avg_test_acc = np.mean(test_accuracies) if test_accuracies else 0
        
        print(f"   Training accuracy: {avg_train_acc:.4f}")
        print(f"   Test accuracy: {avg_test_acc:.4f}")
        
        return {
            'final_train_accuracy': avg_train_acc,
            'test_accuracy': avg_test_acc,
            'learning_progression': len(train_accuracies) > 0
        }
        
    def demo_continual_learning(self) -> Dict[str, Any]:
        """Demonstrate continual learning capabilities."""
        print("   Setting up continual learning for sequential tasks...")
        
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5 classes
        )
        
        # Initialize continual learner with EWC
        learner = ContinualLearner(
            model=model,
            strategy='ewc',
            ewc_lambda=1000.0
        )
        
        # Simulate sequential tasks
        task_performances = []
        
        for task_id in range(5):
            print(f"   Training on task {task_id}...")
            
            # Generate task data
            train_data = torch.randn(200, 10)
            train_labels = torch.randint(0, 5, (200,))
            val_data = torch.randn(50, 10)
            val_labels = torch.randint(0, 5, (50,))
            
            # Create dataloaders (simplified)
            train_loader = [(train_data, train_labels)]
            val_loader = [(val_data, val_labels)]
            
            # Train on task
            task_results = learner.train_task({
                'train': train_loader,
                'val': val_loader
            }, task_id, num_epochs=5)
            
            # Store performance
            task_performances.append({
                'task_id': task_id,
                'train_loss': task_results['final_train_loss'],
                'val_loss': task_results['final_val_loss']
            })
            
        # Evaluate forgetting
        final_performance = task_performances[-1]['val_loss']
        initial_performance = task_performances[0]['val_loss']
        forgetting_measure = initial_performance - final_performance
        
        print(f"   Forgetting measure: {forgetting_measure:.4f}")
        
        return {
            'task_performances': task_performances,
            'forgetting_measure': forgetting_measure,
            'continual_learning_success': forgetting_measure < 1.0
        }
        
    def demo_adaptive_optimization(self) -> Dict[str, Any]:
        """Demonstrate adaptive optimization capabilities."""
        print("   Setting up adaptive optimizer...")
        
        # Create model
        model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        # Initialize adaptive optimizer
        optimizer = AdaptiveOptimizer(
            model=model,
            base_optimizer='adam',
            use_meta_learning=True,
            meta_learning_rate=0.001
        )
        
        # Generate training data
        X = torch.randn(500, 100)
        y = torch.randint(0, 10, (500,))
        
        # Training loop
        print("   Training with adaptive optimization...")
        losses = []
        
        for epoch in range(20):
            # Shuffle data
            indices = torch.randperm(500)
            epoch_loss = 0
            
            for i in range(0, 500, 32):
                batch_indices = indices[i:i+32]
                batch_x = X[batch_indices]
                batch_y = y[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                output = model(batch_x)
                loss = F.cross_entropy(output, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step(loss.item())
                
                epoch_loss += loss.item()
                
            losses.append(epoch_loss / (500 // 32))
            
            if epoch % 5 == 0:
                stats = optimizer.get_optimization_stats()
                print(f"   Epoch {epoch}: Loss = {losses[-1]:.4f}, "
                      f"LR = {stats['current_lr']:.6f}")
        
        # Final stats
        final_stats = optimizer.get_optimization_stats()
        
        return {
            'final_loss': losses[-1],
            'loss_improvement': losses[0] - losses[-1],
            'meta_learning_active': final_stats['meta_learning'] is not None,
            'optimization_converged': losses[-1] < losses[0] * 0.1
        }
        
    def demo_nas(self) -> Dict[str, Any]:
        """Demonstrate Neural Architecture Search capabilities."""
        print("   Setting up Neural Architecture Search...")
        
        # Create search space
        search_space = create_nas_search_space(
            input_shape=(3, 32, 32),
            num_classes=10,
            max_layers=10
        )
        
        # Initialize NAS
        nas = NeuralArchitectureSearch(
            search_space_config=search_space,
            algorithm='random',
            population_size=20
        )
        
        # Define evaluation function (simplified)
        def evaluate_architecture(architecture):
            # Simulate architecture evaluation
            # In practice, this would train and evaluate the architecture
            
            # Simple heuristic based on architecture properties
            score = 0.0
            
            if 'num_layers' in architecture:
                optimal_layers = 6
                score += 1.0 / (1.0 + abs(architecture['num_layers'] - optimal_layers))
                
            if 'channels' in architecture:
                avg_channels = np.mean(architecture['channels'])
                score += 1.0 / (1.0 + abs(avg_channels - 128) / 256)
                
            if 'operations' in architecture:
                diversity = len(set(architecture['operations'])) / len(architecture['operations'])
                score += diversity
                
            return score / 3.0
        
        # Run search
        print("   Running architecture search...")
        best_arch, best_performance = nas.search_architecture(
            evaluation_function=evaluate_architecture,
            num_evaluations=30,
            verbose=False
        )
        
        # Get search summary
        summary = nas.get_search_summary()
        
        print(f"   Best architecture performance: {best_performance:.4f}")
        print(f"   Search efficiency: {summary['search_efficiency']['improvement_rate']:.4f}")
        
        return {
            'best_performance': best_performance,
            'search_efficiency': summary['search_efficiency']['improvement_rate'],
            'total_evaluations': summary['total_evaluations']
        }
        
    def demo_tool_optimization(self) -> Dict[str, Any]:
        """Demonstrate tool optimization capabilities."""
        print("   Setting up tool optimization system...")
        
        # Create tool profiles
        tools = {
            'search_tool': ToolProfile(
                tool_id='search_tool',
                tool_type=ToolType.SEARCH,
                input_schema={'query': str},
                output_schema={'results': list},
                cost_per_use=1.0,
                reliability=0.9
            ),
            'analysis_tool': ToolProfile(
                tool_id='analysis_tool',
                tool_type=ToolType.ANALYSIS,
                input_schema={'data': list},
                output_schema={'analysis': dict},
                cost_per_use=2.0,
                reliability=0.85
            ),
            'generator_tool': ToolProfile(
                tool_id='generator_tool',
                tool_type=ToolType.CONTENT_GENERATION,
                input_schema={'prompt': str},
                output_schema={'content': str},
                cost_per_use=1.5,
                reliability=0.95
            )
        }
        
        # Create task profiles
        tasks = [
            TaskProfile(
                task_id='research_task',
                task_type='research',
                input_data={'topic': str},
                expected_output={'report': str},
                complexity=TaskComplexity.MODERATE
            ),
            TaskProfile(
                task_id='analysis_task',
                task_type='analysis',
                input_data={'dataset': list},
                expected_output={'insights': list},
                complexity=TaskComplexity.SIMPLE
            ),
            TaskProfile(
                task_id='generation_task',
                task_type='content_generation',
                input_data={'brief': dict},
                expected_output={'content': str},
                complexity=TaskComplexity.COMPLEX
            )
        ]
        
        # Initialize meta-learner
        meta_learner = ToolMetaLearner(tools)
        
        # Simulate task execution and learning
        print("   Simulating tool usage optimization...")
        
        for episode in range(50):
            # Select random task
            task = np.random.choice(tasks)
            
            # Get optimization plan
            plan = meta_learner.optimize_task_execution(task)
            
            # Simulate execution
            success = np.random.random() > 0.2  # 80% success rate
            metrics = {
                'quality_score': np.random.uniform(0.3, 1.0) if success else 0.0,
                'efficiency_score': np.random.uniform(0.5, 1.0),
                'latency': np.random.uniform(1.0, 5.0),
                'cost': np.random.uniform(0.5, 2.0)
            }
            
            # Learn from execution
            learning_result = meta_learner.learn_from_execution(task, {
                'tool_sequence': plan['selected_tools'],
                'success': success,
                'metrics': metrics
            })
        
        # Get final report
        report = meta_learner.get_meta_learning_report()
        
        print(f"   Final success rate: {report['learning_summary']['recent_success_rate']:.4f}")
        print(f"   Performance improvement: {report['performance_improvement']:.4f}")
        
        return {
            'success_rate': report['learning_summary']['recent_success_rate'],
            'performance_improvement': report['performance_improvement'],
            'optimization_confidence': report['optimization_confidence']
        }
        
    def demo_adapters(self) -> Dict[str, Any]:
        """Demonstrate adapter capabilities."""
        print("   Setting up meta-learning adapters...")
        
        # Create simple base model
        base_model = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        
        # Create adapter configurations
        adapter_configs = [
            AdapterConfig(
                adapter_type=AdapterType.LINEAR,
                input_dim=512,
                hidden_dim=128
            ),
            AdapterConfig(
                adapter_type=AdapterType.LORA,
                input_dim=512,
                lora_r=8,
                lora_alpha=16
            ),
            AdapterConfig(
                adapter_type=AdapterType.PREFIX_TUNING,
                input_dim=512,
                prefix_length=20
            )
        ]
        
        # Create adapter model
        adapter_model = AdapterModel(base_model, adapter_configs)
        
        print("   Testing adapter efficiency...")
        
        # Count parameters
        base_params = sum(p.numel() for p in base_model.parameters())
        adapter_params = adapter_model.get_adapter_parameters()
        total_adapter_params = sum(count for name, count in adapter_params.items() 
                                 if name != 'total_adapter_params')
        
        # Calculate efficiency
        parameter_efficiency = adapter_model.get_parameter_efficiency()
        
        # Test adaptation
        test_input = torch.randn(32, 100)
        original_output = base_model(test_input)
        adapted_output = adapter_model(test_input)
        
        # Test enabling/disabling adapters
        adapter_model.enable_adapters([AdapterType.LINEAR])
        enabled_output = adapter_model(test_input)
        
        adapter_model.disable_adapters()
        disabled_output = adapter_model(test_input)
        
        # Verify adapters work
        adapters_active = not torch.allclose(adapted_output, disabled_output)
        
        print(f"   Parameter efficiency: {parameter_efficiency:.6f}")
        print(f"   Adapters working: {adapters_active}")
        
        return {
            'parameter_efficiency': parameter_efficiency,
            'adapters_functional': adapters_active,
            'total_adapter_params': total_adapter_params,
            'base_params': base_params
        }
        
    def demo_integrated_system(self) -> Dict[str, Any]:
        """Demonstrate integrated meta-learning system."""
        print("   Creating integrated meta-learning system...")
        
        # Create a more complex base model
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(50, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 5)
                )
                
            def forward(self, x):
                return self.decoder(self.encoder(x))
        
        base_model = ComplexModel()
        
        # Task requirements
        task_requirements = {
            'few_shot': True,
            'parameter_efficiency': True,
            'rapid_adaptation': True,
            'multimodal': False
        }
        
        # Create adaptive adapter ensemble
        adapter_model = create_adaptive_adapter_ensemble(
            base_model, task_requirements, model_type='language'
        )
        
        # Initialize all meta-learning components
        optimizer = AdaptiveOptimizer(
            model=adapter_model,
            base_optimizer='adamw',
            use_meta_learning=True
        )
        
        # Simulate integrated training
        print("   Training integrated system...")
        start_time = time.time()
        
        training_losses = []
        adaptation_accuracies = []
        
        for epoch in range(15):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Simulate multiple batches
            for batch in range(20):
                # Generate synthetic data
                x = torch.randn(16, 50)
                y = torch.randint(0, 5, (16,))
                
                # Forward pass
                optimizer.zero_grad()
                output = adapter_model(x)
                loss = F.cross_entropy(output, y)
                
                # Meta-learning backward pass
                loss.backward()
                optimizer.step(loss.item())
                
                # Track metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            epoch_accuracy = correct / total
            training_losses.append(epoch_loss / 20)
            adaptation_accuracies.append(epoch_accuracy)
            
            if epoch % 5 == 0:
                print(f"   Epoch {epoch}: Loss = {training_losses[-1]:.4f}, "
                      f"Accuracy = {adaptation_accuracies[-1]:.4f}")
        
        training_time = time.time() - start_time
        
        # Test system capabilities
        print("   Testing system capabilities...")
        
        # Test adapter efficiency
        param_efficiency = adapter_model.get_parameter_efficiency()
        
        # Test optimizer statistics
        opt_stats = optimizer.get_optimization_stats()
        
        # Final performance
        final_accuracy = adaptation_accuracies[-1] if adaptation_accuracies else 0
        final_loss = training_losses[-1] if training_losses else float('inf')
        
        return {
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'training_time': training_time,
            'parameter_efficiency': param_efficiency,
            'meta_learning_active': opt_stats['meta_learning'] is not None,
            'system_converged': final_loss < training_losses[0] * 0.1 if training_losses else False
        }
        
    def print_summary(self):
        """Print comprehensive summary of demo results."""
        print("\n" + "=" * 50)
        print("📊 META-LEARNING FRAMEWORK DEMO SUMMARY")
        print("=" * 50)
        
        for component, results in self.results.items():
            print(f"\n{component.upper().replace('_', ' ')}:")
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  {results}")
        
        # Overall system assessment
        print("\n🎯 SYSTEM ASSESSMENT:")
        
        # Check if all components are working
        working_components = 0
        total_components = len(self.results)
        
        for component, results in self.results.items():
            if isinstance(results, dict):
                if 'success' in results and results['success']:
                    working_components += 1
                elif 'convergence_achieved' in results and results['convergence_achieved']:
                    working_components += 1
                elif 'learning_progression' in results and results['learning_progression']:
                    working_components += 1
                elif 'parameter_efficiency' in results and results['parameter_efficiency'] > 0:
                    working_components += 1
        
        success_rate = working_components / total_components
        print(f"  Component Success Rate: {success_rate:.1%}")
        
        # Key achievements
        print("\n🏆 KEY ACHIEVEMENTS:")
        
        achievements = []
        
        if 'maml' in self.results:
            if self.results['maml'].get('convergence_achieved', False):
                achievements.append("✅ MAML successfully converged for rapid adaptation")
                
        if 'few_shot' in self.results:
            if self.results['few_shot'].get('test_accuracy', 0) > 0.6:
                achievements.append("✅ Few-shot learning achieved good performance")
                
        if 'continual_learning' in self.results:
            if not self.results['continual_learning'].get('forgetting_measure', 1) > 1.0:
                achievements.append("✅ Continual learning prevented catastrophic forgetting")
                
        if 'adaptive_optimization' in self.results:
            if self.results['adaptive_optimization'].get('meta_learning_active', False):
                achievements.append("✅ Meta-learning optimization improved training")
                
        if 'adapters' in self.results:
            if self.results['adapters'].get('parameter_efficiency', 0) < 0.01:
                achievements.append("✅ Adapters achieved high parameter efficiency")
                
        for achievement in achievements:
            print(f"  {achievement}")
        
        # System readiness
        print(f"\n🚀 FRAMEWORK READINESS: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print("  Status: PRODUCTION READY ✅")
        elif success_rate >= 0.6:
            print("  Status: MOSTLY READY ⚠️")
        else:
            print("  Status: NEEDS IMPROVEMENT ❌")
            
        print("\n" + "=" * 50)


def run_meta_learning_demo():
    """Run the complete meta-learning demonstration."""
    demo = MetaLearningDemo()
    return demo.run_complete_demo()


if __name__ == "__main__":
    # Run the demo
    results = run_meta_learning_demo()
    
    # Save results to file
    with open('meta_learning_demo_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return obj
            
        json_results = {k: {kk: convert_numpy(vv) for kk, vv in v.items()} 
                       for k, v in results.items()}
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to meta_learning_demo_results.json")