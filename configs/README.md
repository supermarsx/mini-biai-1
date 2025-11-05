# Step 2 Configuration System

This directory contains the complete configuration system for mini-biai-1 Step 2, supporting multi-expert routing, affect modulation, auto-learning, and SSM-based language processing.

## Overview

The Step 2 configuration system provides:

- **Multi-Expert Configuration**: Support for Language, Vision, Symbolic, and Affect experts
- **Affect Modulation**: Emotional state tracking and modulation parameters
- **SSM/Linear-Attention**: Configuration options for Mamba and Linear Attention models
- **Auto-Learning System**: STDP and online learning configuration
- **Performance Tuning**: Latency, memory, and throughput optimization
- **Configuration Management**: Validation, migration, and template tools

## Files

### Core Configuration Files

- `step2_base.yaml` - Main Step 2 configuration template with all settings
- `language_expert_template.yaml` - Language expert configuration template
- `vision_expert_template.yaml` - Vision expert configuration template  
- `symbolic_expert_template.yaml` - Symbolic expert configuration template
- `affect_expert_template.yaml` - Affect expert configuration template

### Utility Scripts

- `config_validation_tools.py` - Configuration validation and migration tools
- `generate_step2_config.py` - Configuration generator for different use cases

## Quick Start

### 1. Generate a Quick-Start Configuration

```bash
python configs/generate_step2_config.py quickstart --output configs/my_step2_config.yaml
```

### 2. Generate Performance-Optimized Configuration

```bash
python configs/generate_step2_config.py performance --output configs/performance_config.yaml
```

### 3. Generate Minimal Configuration for Testing

```bash
python configs/generate_step2_config.py minimal --output configs/minimal_config.yaml
```

### 4. Validate Configuration

```bash
python configs/config_validation_tools.py validate configs/my_step2_config.yaml
```

### 5. Migrate from Step 1

```bash
python configs/config_validation_tools.py migrate configs/step1_base.yaml configs/migrated_step2_config.yaml
```

## Configuration Structure

### Top-Level Sections

```yaml
# Core sections in step2_base.yaml
general:                    # Basic project settings
routing:                   # Multi-expert routing configuration
  experts:                # Expert-specific configurations
    language:            # Language expert settings
    vision:              # Vision expert settings  
    symbolic:            # Symbolic expert settings
    affect:              # Affect expert settings
affect:                   # Affect modulation system
language_backbone:        # SSM/Linear attention configuration
auto_learning:            # STDP and online learning
memory:                   # Enhanced memory systems
performance_tuning:       # Performance optimization
training:                 # Multi-task training
evaluation:               # Metrics and evaluation
hardware:                 # Hardware specifications
logging:                  # Logging and monitoring
experimental:             # Research features
```

### Expert Configuration

Each expert has its own configuration section:

#### Language Expert
- Text generation and understanding
- Conversational context processing
- Natural language reasoning
- Affect-aware language generation

#### Vision Expert  
- Image understanding and analysis
- Visual scene processing
- Multimodal alignment
- Image-to-text tasks

#### Symbolic Expert
- Logical reasoning and computation
- Mathematical problem solving
- Structured data processing
- Constraint satisfaction

#### Affect Expert
- Emotion recognition from text
- Sentiment analysis
- Social cue processing
- Affect state modulation (log-only in Step 2)

## Advanced Usage

### Custom Configuration Generation

```python
from generate_step2_config import Step2ConfigGenerator, ConfigRequirements

generator = Step2ConfigGenerator()

requirements = ConfigRequirements(
    project_name="my_project",
    expert_count=4,
    latency_target_ms=150,
    memory_gb=8,
    enable_vision=True,
    enable_affect=True,
    enable_auto_learning=True
)

generator.generate_config(requirements, "my_config.yaml")
```

### Configuration Validation

```python
from config_validation_tools import ConfigValidator

validator = ConfigValidator(strict_mode=False)
result = validator.validate_config("my_config.yaml")

if not result.is_valid:
    print("Validation failed:")
    for error in result.errors:
        print(f"  ERROR: {error}")
    for warning in result.warnings:
        print(f"  WARNING: {warning}")
```

### Template Merging

```python
from config_validation_tools import ConfigTemplateManager

# Load base configuration
with open("step2_base.yaml", 'r') as f:
    base_config = yaml.safe_load(f)

# Merge with expert templates
template_manager = ConfigTemplateManager(".")
merged_config = template_manager.merge_templates(
    base_config, 
    ["language_expert", "vision_expert", "symbolic_expert", "affect_expert"]
)
```

## Key Configuration Parameters

### Multi-Expert Routing

```yaml
routing:
  n_experts: 4                    # Number of experts (Language, Vision, Symbolic, Affect)
  top_k: 2                        # Number of experts to activate simultaneously
  temperature: 0.1               # Gating temperature for routing
  spike_threshold: 0.7           # Minimum spike rate for activation
  target_spike_rate: 0.10        # Target neural firing rate
```

### Affect Modulation

```yaml
affect:
  enabled: true                  # Enable affect detection
  log_only: true                 # Log affect but don't modify routing (Step 2)
  state_representation:
    dimensions: [valence, arousal, dominance]  # VAD model
    emotion_categories: 8        # Number of discrete emotions
```

### SSM Language Backbone

```yaml
language_backbone:
  type: "mamba"                 # mamba, linear_attention, transformer
  d_model: 512                  # Model hidden size
  n_layers: 4                   # Number of layers
  
  # Mamba-specific settings
  mamba:
    d_state: 64                 # SSM state dimension
    d_conv: 4                   # Convolution kernel size
    expand: 2                   # Expansion factor
```

### Auto-Learning System

```yaml
auto_learning:
  enabled: true                 # Enable STDP and online learning
  
  stdp:                        # Spike-Timing-Dependent Plasticity
    enabled: true
    learning_rate: 0.001
    tau_pre: 20.0              # Pre-synaptic trace decay
    tau_post: 20.0             # Post-synaptic trace decay
    a_plus: 0.1                # Potentiation rate
    a_minus: 0.1               # Depression rate
```

### Performance Tuning

```yaml
performance_tuning:
  latency:
    target_latency_ms: 150      # End-to-end latency target
    budget_allocation:          # Budget breakdown
      routing: 10               # ms for routing
      retrieval: 20             # ms for retrieval  
      expert_processing: 100    # ms for expert execution
      affect_detection: 5       # ms for affect
      generation: 15            # ms for generation
  
  memory:
    target_memory_gb: 8         # Memory usage target
    mixed_precision: true       # Use FP16 for efficiency
```

## Expert-Specific Templates

Each expert template provides specialized configuration:

### Language Expert Features
- Text generation with Mamba backbone
- Conversational context management
- Affect-aware language generation
- Multi-task training for language tasks

### Vision Expert Features
- CNN + Transformer architecture
- Image classification and captioning
- Multimodal alignment with text
- Visual affect detection (Step 3)

### Symbolic Expert Features
- Transformer with symbolic bias
- Logical reasoning and mathematical computation
- Constraint satisfaction solving
- Structured data processing

### Affect Expert Features
- VAD (Valence-Arousal-Dominance) emotion model
- Text-based emotion recognition
- Affect state tracking and logging
- Modulatory influence on other experts

## Performance Recommendations

### Development Environment
- Use `quickstart` configuration
- Latency target: 200-300ms
- Memory: 4-8GB
- Batch size: 16-32

### Production Environment
- Use `performance` configuration
- Latency target: 100-150ms
- Memory: 8-16GB  
- Batch size: 32-64
- GPU memory: 4-8GB

### Research Environment
- Use custom configuration
- Enable all experimental features
- Distributed training: true
- Larger memory allocation

## Configuration Validation

The validation tools check:

1. **Structure**: Required sections and fields present
2. **Values**: Parameters within valid ranges
3. **Consistency**: Cross-parameter consistency
4. **Expert Config**: Expert-specific requirements
5. **Performance**: Latency and resource constraints

### Common Validation Issues

- `top_k > n_experts`: Cannot select more experts than available
- `Latency budget overflow`: Sum of allocations exceeds target
- `Missing expert configuration`: Affect enabled but no affect expert
- `Memory hierarchy violation`: STM capacity >= LTM capacity

## Migration Guide

### From Step 1 to Step 2

1. **Use migration tool**:
   ```bash
   python configs/config_validation_tools.py migrate \
       configs/step1_base.yaml \
       configs/step2_migrated.yaml
   ```

2. **Manual review**: Check for required Step 2 additions:
   - Multi-expert routing configuration
   - Affect modulation settings
   - Auto-learning parameters
   - Performance tuning section

3. **Validation**: Run validation on migrated config:
   ```bash
   python configs/config_validation_tools.py validate configs/step2_migrated.yaml
   ```

## Troubleshooting

### Common Issues

1. **Import Error for pydantic**
   ```bash
   pip install pydantic
   ```

2. **Configuration file not found**
   - Check file paths are correct
   - Ensure YAML/JSON syntax is valid

3. **Validation fails**
   - Check required fields are present
   - Verify parameter ranges
   - Review consistency checks

4. **Performance issues**
   - Adjust latency targets
   - Modify batch sizes
   - Check memory allocations

### Debug Mode

Enable debug mode in configuration:

```yaml
general:
  debug: true
  
logging:
  level: "DEBUG"
  
experimental:
  detailed_logging: true
```

## Future Enhancements

The configuration system is designed for extensibility:

- **Step 3**: Visual affect detection, active modulation
- **Step 4**: Distributed training, real-time learning
- **Step 5**: Hierarchical memory, meta-learning

## Support

For configuration issues:
1. Check validation output for specific errors
2. Review this documentation
3. Examine template files for examples
4. Use the generator tools for common use cases
