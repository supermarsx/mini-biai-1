# Comprehensive Evaluation Suite

A complete evaluation framework for assessing Mini BAI model capabilities across multiple dimensions including reasoning, comprehension, safety, and performance.

## Features

### Core Evaluation Modules
- **MMLU-lite**: Language understanding and reasoning
- **GSM8K-lite**: Mathematical problem solving
- **VQA-lite**: Visual question answering
- **Toxicity Testing**: Content safety evaluation
- **A/B Testing**: Comparative model analysis
- **Energy Monitoring**: Performance and efficiency metrics

### Advanced Capabilities
- Async/await support for concurrent evaluations
- Statistical significance testing
- Real-time progress tracking
- Comprehensive reporting and visualization
- Configurable test parameters
- Multi-worker parallel execution

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from comprehensive_evaluation_suite import ComprehensiveEvaluationSuite

# Initialize evaluation suite
evaluator = ComprehensiveEvaluationSuite()

# Run basic evaluation
results = await evaluator.run_full_evaluation()

# Print summary
print(results.summary())
```

### Configuration
```python
# Custom configuration
evaluator = ComprehensiveEvaluationSuite(
    config={
        'mmlu_lite': {'pass_at_k': 1},
        'gsm8k_lite': {'difficulty': 'easy'},
        'toxicity': {'threshold': 0.5},
        'energy_monitoring': {'enabled': True}
    }
)
```

## Detailed Documentation

### MMLU-lite Evaluation

Multiple-choice language understanding with reduced question set.

**Configuration Options:**
- `pass_at_k`: Number of attempts per question (default: 1)
- `subjects`: List of subjects to test (default: all)
- `max_samples`: Maximum questions per subject (default: 100)

**Usage:**
```python
from evaluation_modules import MMLULiteEvaluator

evaluator = MMLULiteEvaluator(
    pass_at_k=1,
    subjects=['math', 'science', 'history'],
    max_samples=50
)

results = await evaluator.evaluate(model)
print(f"Accuracy: {results['accuracy']:.3f}")
```

**Target Metrics:**
- **Accuracy**: ≥70% (5-shot), ≥65% (0-shot)
- **Pass@1**: Single attempt accuracy
- **Coverage**: Subject-wise performance breakdown

### GSM8K-lite Evaluation

Mathematical reasoning and problem-solving assessment.

**Configuration Options:**
- `difficulty`: Question difficulty level ('easy', 'medium', 'hard')
- `show_reasoning`: Whether to include reasoning steps
- `max_samples`: Maximum questions to evaluate

**Usage:**
```python
from evaluation_modules import GSM8KLiteEvaluator

evaluator = GSM8KLiteEvaluator(
    difficulty='medium',
    show_reasoning=True,
    max_samples=100
)

results = await evaluator.evaluate(model)
print(f"Math Accuracy: {results['accuracy']:.3f}")
```

**Target Metrics:**
- **Accuracy**: ≥80% (easy), ≥70% (medium), ≥60% (hard)
- **Reasoning Quality**: Step-by-step solution validation
- **Error Analysis**: Common mistake patterns

### VQA-lite Evaluation

Visual question answering capabilities assessment.

**Configuration Options:**
- `image_size`: Input image dimensions
- `question_types`: Types of questions to include
- `max_samples`: Maximum image-question pairs

**Usage:**
```python
from evaluation_modules import VQALiteEvaluator

evaluator = VQALiteEvaluator(
    image_size=(224, 224),
    question_types=['what', 'where', 'when', 'who', 'why', 'how'],
    max_samples=50
)

results = await evaluator.evaluate(model)
print(f"VQA Accuracy: {results['accuracy']:.3f}")
```

**Target Metrics:**
- **Accuracy**: ≥65% overall
- **Question Type Performance**: Per-category accuracy
- **Image Difficulty**: Performance by image complexity

### Toxicity Testing

Content safety and harmful content detection.

**Configuration Options:**
- `threshold`: Classification threshold for toxicity
- `categories`: Toxicity categories to detect
- `severity_levels`: Toxicity severity classification

**Usage:**
```python
from evaluation_modules import ToxicityEvaluator

evaluator = ToxicityEvaluator(
    threshold=0.5,
    categories=['hate', 'harassment', 'violence', 'self-harm'],
    severity_levels=['low', 'medium', 'high']
)

results = await evaluator.evaluate(model)
print(f"Toxicity Rate: {results['toxicity_rate']:.3f}")
```

**Target Metrics:**
- **Toxicity Rate**: <2% harmful content generation
- **False Positive Rate**: <5% safe content flagged as toxic
- **Category Performance**: Per-category detection accuracy

### A/B Testing Framework

Comparative analysis of different model configurations.

**Configuration Options:**
- `models`: List of models to compare
- `metrics`: Evaluation metrics to compare
- `statistical_test`: Statistical significance test type
- `alpha`: Significance level (default: 0.05)

**Usage:**
```python
from evaluation_modules import ABTestingFramework

framework = ABTestingFramework(
    models=[model_v1, model_v2],
    metrics=['accuracy', 'latency', 'throughput'],
    statistical_test='ttest',
    alpha=0.05
)

results = await framework.compare()
print(f"Significant differences: {results['significant_differences']}")
```

**Features:**
- Statistical significance testing
- Effect size calculations
- Confidence interval estimation
- Multiple comparison correction

### Energy Monitoring

Performance and energy efficiency tracking.

**Configuration Options:**
- `monitoring_interval`: Energy measurement frequency
- `metrics`: Energy metrics to track
- `hardware_support`: Hardware-specific monitoring

**Usage:**
```python
from evaluation_modules import EnergyMonitor

monitor = EnergyMonitor(
    monitoring_interval=1.0,
    metrics=['power', 'energy', 'temperature'],
    hardware_support=True
)

results = await monitor.measure(model)
print(f"Energy per token: {results['energy_per_token']:.2f}J")
```

**Target Metrics:**
- **Energy per Token**: ≤2.0J/token
- **Power Consumption**: Real-time power usage
- **Efficiency Ratio**: Performance per watt
- **Thermal Management**: Temperature monitoring

## Advanced Usage

### Custom Evaluation Pipeline

```python
from comprehensive_evaluation_suite import ComprehensiveEvaluationSuite
import asyncio

# Create custom evaluation suite
suite = ComprehensiveEvaluationSuite()

# Add custom evaluation modules
suite.add_module('custom_test', CustomEvaluator())

# Configure evaluation parameters
suite.configure({
    'mmlu_lite': {'pass_at_k': 3},
    'gsm8k_lite': {'difficulty': 'hard'},
    'parallel_workers': 4
})

# Run evaluation with progress tracking
async def run_with_progress():
    async for progress in suite.run_with_progress():
        print(f"Progress: {progress['percentage']:.1f}% - {progress['current_task']}")

results = await run_with_progress()
```

### Batch Processing

```python
# Process multiple models
models = [model1, model2, model3]
results = []

for model in models:
    result = await suite.evaluate_model(model)
    results.append(result)

# Compare results
comparison = suite.compare_results(results)
```

### Report Generation

```python
# Generate comprehensive report
report = suite.generate_report(results)
report.save('evaluation_report.html')

# Export results
suite.export_results(results, 'results.json')
suite.export_metrics(results, 'metrics.csv')
```

## Configuration Reference

### Global Settings

```yaml
evaluation:
  parallel_workers: 4
  timeout: 300
  retry_attempts: 3
  random_seed: 42
  
  # Logging configuration
  logging:
    level: INFO
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: "evaluation.log"
    
  # Output configuration
  output:
    directory: "results"
    format: ["json", "csv", "html"]
    include_raw_data: true
```

### Module-Specific Settings

```yaml
mmlu_lite:
  pass_at_k: 1
  subjects:
    - mathematics
    - science
    - history
    - literature
  max_samples: 100
  few_shot_examples: 5
  
gsm8k_lite:
  difficulty: "medium"
  show_reasoning: true
  max_samples: 100
  precision_threshold: 0.01
  
vqa_lite:
  image_size: [224, 224]
  question_types:
    - "what"
    - "where"
    - "when"
    - "who"
    - "why"
    - "how"
  max_samples: 50
  
toxicity:
  threshold: 0.5
  categories:
    - "hate"
    - "harassment"
    - "violence"
    - "self-harm"
  severity_levels: ["low", "medium", "high"]
  
energy_monitoring:
  enabled: true
  monitoring_interval: 1.0
  hardware_support: true
  metrics:
    - "power"
    - "energy"
    - "temperature"
```

## API Reference

### ComprehensiveEvaluationSuite

Main evaluation framework class.

#### Methods

**`__init__(config=None)`**
- Initialize evaluation suite with optional configuration

**`configure(config_dict)`**
- Configure evaluation parameters
- Parameters: `config_dict` - Configuration dictionary

**`add_module(name, module)`**
- Add custom evaluation module
- Parameters: `name` - Module name, `module` - Evaluation module instance

**`run_full_evaluation()`**
- Run complete evaluation suite
- Returns: Evaluation results dictionary

**`evaluate_model(model)`**
- Evaluate single model
- Parameters: `model` - Model to evaluate
- Returns: Model evaluation results

**`compare_models(models)`**
- Compare multiple models
- Parameters: `models` - List of models to compare
- Returns: Comparison results

**`generate_report(results)`**
- Generate comprehensive HTML report
- Parameters: `results` - Evaluation results
- Returns: Report object

**`export_results(results, filename)`**
- Export results to JSON file
- Parameters: `results` - Results to export, `filename` - Output filename

### Evaluation Modules

#### Base Module Interface

All evaluation modules inherit from `BaseEvaluator`:

```python
class BaseEvaluator:
    async def evaluate(self, model):
        """Evaluate model and return results"""
        pass
        
    def get_metrics(self):
        """Get evaluation metrics"""
        pass
        
    def validate_config(self, config):
        """Validate configuration"""
        pass
```

## Examples

### Complete Evaluation Example

```python
import asyncio
from comprehensive_evaluation_suite import ComprehensiveEvaluationSuite
from model_interface import MiniBAIModel

async def main():
    # Initialize model and evaluator
    model = MiniBAIModel()
    evaluator = ComprehensiveEvaluationSuite()
    
    # Run complete evaluation
    results = await evaluator.run_full_evaluation()
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"MMLU-lite Accuracy: {results.mmlu_lite.accuracy:.3f}")
    print(f"GSM8K-lite Accuracy: {results.gsm8k_lite.accuracy:.3f}")
    print(f"VQA-lite Accuracy: {results.vqa_lite.accuracy:.3f}")
    print(f"Toxicity Rate: {results.toxicity.toxicity_rate:.3f}")
    print(f"Energy per Token: {results.energy.energy_per_token:.2f}J")
    
    # Generate detailed report
    report = evaluator.generate_report(results)
    report.save('detailed_evaluation.html')
    
    # Export raw data
    evaluator.export_results(results, 'evaluation_data.json')
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
```

### A/B Testing Example

```python
import asyncio
from evaluation_modules import ABTestingFramework
from model_interface import MiniBAIModel

async def compare_models():
    # Load different model versions
    model_v1 = MiniBAIModel(version="1.0")
    model_v2 = MiniBAIModel(version="2.0")
    
    # Setup A/B testing
    framework = ABTestingFramework(
        models=[model_v1, model_v2],
        metrics=['accuracy', 'latency', 'throughput'],
        statistical_test='mann_whitney'
    )
    
    # Run comparison
    results = await framework.compare()
    
    # Print results
    print("\n=== A/B TEST RESULTS ===")
    for metric, result in results.items():
        print(f"{metric}:")
        print(f"  p-value: {result.p_value:.4f}")
        print(f"  significant: {result.significant}")
        print(f"  effect_size: {result.effect_size:.3f}")
        
    return results

if __name__ == "__main__":
    asyncio.run(compare_models())
```

### Custom Evaluation Module

```python
from evaluation_modules import BaseEvaluator
import asyncio

class CustomEvaluator(BaseEvaluator):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
    async def evaluate(self, model):
        # Custom evaluation logic
        results = {
            'custom_metric1': 0.85,
            'custom_metric2': 0.92,
            'summary': 'Custom evaluation completed'
        }
        return results
        
    def get_metrics(self):
        return ['custom_metric1', 'custom_metric2']
        
    def validate_config(self, config):
        # Validate configuration parameters
        return True

# Usage
custom_eval = CustomEvaluator()
results = await custom_eval.evaluate(model)
```

## Troubleshooting

### Common Issues

**Memory Errors**
```python
# Reduce batch sizes
config = {
    'mmlu_lite': {'max_samples': 50},  # Reduced from 100
    'gsm8k_lite': {'max_samples': 50}
}
```

**Timeout Issues**
```python
# Increase timeout and add retries
config = {
    'timeout': 600,  # 10 minutes
    'retry_attempts': 5
}
```

**Model Loading Errors**
```python
# Add model validation
try:
    await model.load()
except Exception as e:
    print(f"Model loading failed: {e}")
    # Fallback to mock model
    model = MockModel()
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
evaluator = ComprehensiveEvaluationSuite(debug=True)
results = await evaluator.run_full_evaluation()
```

## Contributing

### Adding New Evaluation Modules

1. Inherit from `BaseEvaluator`
2. Implement required methods
3. Add configuration validation
4. Include comprehensive tests
5. Update documentation

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Add unit tests for all modules
- Use async/await for I/O operations

## License

This evaluation suite is part of the Mini BAI benchmarking framework. See LICENSE file for details.