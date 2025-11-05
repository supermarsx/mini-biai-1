# Examples and Tutorials

This section provides hands-on examples and tutorials to help you learn mini-biai-1 through practical implementations.

## Table of Contents

- [Examples Overview](#examples-overview)
- [Basic Examples](#basic-examples)
- [Intermediate Examples](#intermediate-examples)
- [Advanced Examples](#advanced-examples)
- [Tutorials](#tutorials)
- [Real-World Applications](#real-world-applications)

## Examples Overview

The examples are organized by complexity and use case:

### Basic Examples (Start Here)
- [Basic Usage](#basic-usage) - Simple text generation
- [Memory Operations](#memory-operations) - Store and retrieve data
- [Configuration Demo](#configuration-demo) - Configuration examples
- [CLI Usage](#cli-usage) - Command-line interface examples

### Intermediate Examples
- [Multi-Expert System](#multi-expert-system) - Using multiple experts
- [Training Examples](#training-examples) - Training basic models
- [API Integration](#api-integration) - REST API usage
- [Data Processing](#data-processing) - Working with datasets

### Advanced Examples
- [Custom Experts](#custom-experts) - Creating custom expert modules
- [Performance Optimization](#performance-optimization) - Advanced optimization
- [Distributed Training](#distributed-training) - Multi-GPU training
- [Research Applications](#research-applications) - Research use cases

## Basic Examples

### Basic Usage

The simplest way to use mini-biai-1:

```python
#!/usr/bin/env python3
"""
Basic usage example for mini-biai-1
"""

from mini_biai_1 import create_pipeline
from mini_biai_1.configs import load_config

def basic_usage_example():
    """Demonstrate basic mini-biai-1 usage."""
    
    print("🧠 Mini-Biai-1 Basic Usage Example")
    print("="*50)
    
    try:
        # Method 1: Using create_pipeline (simplest)
        print("1. Creating pipeline...")
        pipeline = create_pipeline("configs/quickstart.yaml")
        
        # Generate text
        print("\n2. Generating text...")
        prompt = "The brain processes information through"
        result = pipeline.generate(prompt, max_length=50)
        
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
        
        # Method 2: Using coordinator directly
        print("\n3. Using coordinator directly...")
        from mini_biai_1.coordinator import MiniBiAiCoordinator
        
        config = load_config("configs/quickstart.yaml")
        coordinator = MiniBiAiCoordinator(config)
        
        # Process multiple inputs
        prompts = [
            "Neural networks are inspired by",
            "Spiking neural networks work by",
            "Memory systems in AI include"
        ]
        
        print("\n4. Processing multiple prompts:")
        for i, prompt in enumerate(prompts, 1):
            result = coordinator.generate(prompt, max_length=30)
            print(f"   {i}. {prompt} → {result[:50]}...")
        
        print("\n✅ Basic usage example completed!")
        
    except FileNotFoundError as e:
        print(f"⚠️  Configuration file not found: {e}")
        print("Using default configuration...")
        
        # Use default pipeline
        pipeline = create_pipeline()
        result = pipeline.generate("Hello, mini-biai-1!")
        print(f"Generated: {result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    basic_usage_example()
```

### Memory Operations

Working with the hierarchical memory system:

```python
#!/usr/bin/env python3
"""
Memory system usage example
"""

from mini_biai_1.memory import HierarchicalMemory
from mini_biai_1.configs import load_config

def memory_example():
    """Demonstrate memory system operations."""
    
    print("💾 Memory System Example")
    print("="*50)
    
    # Initialize memory system
    config = load_config("configs/memory_config.yaml")
    memory = HierarchicalMemory(config)
    
    print("1. Memory system initialized")
    
    # Store different types of information
    print("\n2. Storing information...")
    
    # Store user context
    memory.store("user_id", "user_12345")
    memory.store("session_start", "2024-11-06T10:30:00")
    memory.store("preferences", {"language": "en", "theme": "dark"})
    
    # Store knowledge
    memory.store("fact:brain_structure", 
                "The brain consists of neurons connected by synapses")
    memory.store("concept:machine_learning", 
                "ML is a subset of AI focused on pattern recognition")
    
    print("✅ Stored user context and knowledge")
    
    # Retrieve information
    print("\n3. Retrieving information...")
    
    user_id = memory.retrieve("user_id")
    preferences = memory.retrieve("preferences")
    brain_fact = memory.retrieve("fact:brain_structure")
    
    print(f"User ID: {user_id}")
    print(f"Preferences: {preferences}")
    print(f"Brain fact: {brain_fact}")
    
    # Search for information
    print("\n4. Searching memory...")
    
    # Search for brain-related information
    brain_results = memory.search("brain", top_k=5)
    print(f"Brain search results ({len(brain_results)} items):")
    for key, score in brain_results:
        print(f"  - {key} (score: {score:.3f})")
    
    # Search for AI/ML related information  
    ai_results = memory.search("artificial intelligence", top_k=5)
    print(f"\nAI search results ({len(ai_results)} items):")
    for key, score in ai_results:
        print(f"  - {key} (score: {score:.3f})")
    
    # Memory statistics
    print("\n5. Memory statistics...")
    stats = memory.get_memory_stats()
    print(f"Working memory usage: {stats.working_memory.usage}/{stats.working_memory.capacity}")
    print(f"Long-term memory items: {stats.ltm.item_count}")
    print(f"Search performance: {stats.ltm.avg_search_time:.4f}s")
    
    print("\n✅ Memory operations completed!")

if __name__ == "__main__":
    memory_example()
```

### Configuration Demo

Exploring different configuration options:

```python
#!/usr/bin/env python3
"""
Configuration system demonstration
"""

from mini_biai_1.configs import load_config, create_config
from mini_biai_1.coordinator import MiniBiAiCoordinator

def configuration_demo():
    """Demonstrate configuration system."""
    
    print("⚙️ Configuration System Demo")
    print("="*50)
    
    # 1. Load predefined configuration
    print("1. Loading predefined configuration...")
    try:
        config = load_config("configs/demo_config.yaml")
        print("✅ Loaded demo configuration")
        print(f"   - Model type: {config.get('model.type', 'Not specified')}")
        print(f"   - Max epochs: {config.get('training.max_epochs', 'Not specified')}")
        print(f"   - Batch size: {config.get('training.batch_size', 'Not specified')}")
    except FileNotFoundError:
        print("⚠️ Demo config not found, creating default...")
        config = create_default_config()
    
    # 2. Modify configuration programmatically
    print("\n2. Modifying configuration...")
    
    # Update training parameters
    config["training"]["max_epochs"] = 5
    config["training"]["batch_size"] = 16
    config["training"]["learning_rate"] = 0.001
    
    # Update memory parameters
    config["memory"]["working_memory"]["capacity"] = 500
    config["memory"]["episodic_memory"]["capacity"] = 5000
    
    print("✅ Configuration updated")
    print(f"   - Max epochs: {config['training']['max_epochs']}")
    print(f"   - Batch size: {config['training']['batch_size']}")
    print(f"   - Learning rate: {config['training']['learning_rate']}")
    print(f"   - Working memory capacity: {config['memory']['working_memory']['capacity']}")
    
    # 3. Validate configuration
    print("\n3. Validating configuration...")
    
    try:
        validate_config(config)
        print("✅ Configuration is valid")
    except ValidationError as e:
        print(f"❌ Configuration validation failed: {e}")
        return
    
    # 4. Test with modified configuration
    print("\n4. Testing with modified configuration...")
    
    try:
        coordinator = MiniBiAiCoordinator(config)
        print("✅ Coordinator initialized with modified config")
        
        # Quick generation test
        result = coordinator.generate("Testing configuration", max_length=20)
        print(f"   Test generation: {result}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    print("\n✅ Configuration demo completed!")

def create_default_config():
    """Create a default configuration."""
    return {
        "model": {
            "type": "mini_biai_1",
            "hidden_dim": 512,
            "num_layers": 6
        },
        "training": {
            "max_epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "memory": {
            "working_memory": {
                "capacity": 1000
            },
            "episodic_memory": {
                "capacity": 10000
            }
        },
        "inference": {
            "max_length": 512,
            "temperature": 0.7
        }
    }

def validate_config(config):
    """Validate configuration parameters."""
    required_sections = ["model", "training", "memory"]
    
    for section in required_sections:
        if section not in config:
            raise ValidationError(f"Missing required section: {section}")
    
    # Validate training parameters
    if config["training"]["batch_size"] <= 0:
        raise ValidationError("Batch size must be positive")
    
    if config["training"]["learning_rate"] <= 0:
        raise ValidationError("Learning rate must be positive")

class ValidationError(Exception):
    pass

if __name__ == "__main__":
    configuration_demo()
```

### CLI Usage

Using the command-line interface:

```python
#!/usr/bin/env python3
"""
CLI usage demonstration
"""

import subprocess
import os
import json

def cli_demo():
    """Demonstrate CLI usage."""
    
    print("🖥️  CLI Usage Demo")
    print("="*50)
    
    # Create sample corpus files
    sample_corpus = {
        "artificial_intelligence.txt": """
        Artificial Intelligence (AI) is intelligence demonstrated by machines,
        in contrast to the natural intelligence displayed by humans.
        Leading AI textbooks define the field as the study of "intelligent agents":
        any device that perceives its environment and takes actions that maximize
        its chance of successfully achieving its goals.
        """,
        
        "machine_learning.txt": """
        Machine learning (ML) is a subset of artificial intelligence (AI) that
        provides systems the ability to automatically learn and improve from 
        experience without being explicitly programmed.
        Machine learning focuses on the development of computer programs
        that can access data and use it to learn for themselves.
        """,
        
        "neural_networks.txt": """
        Artificial neural networks (ANN) are computing systems inspired by
        the biological neural networks that constitute animal brains.
        An ANN is based on a collection of connected units or nodes called
        artificial neurons, which loosely model the neurons in a biological brain.
        """
    }
    
    # Create data directory
    os.makedirs("sample_data/corpus", exist_ok=True)
    
    print("1. Creating sample corpus files...")
    for filename, content in sample_corpus.items():
        with open(f"sample_data/corpus/{filename}", "w") as f:
            f.write(content.strip())
    
    print("✅ Created sample corpus")
    
    # Demo CLI commands
    print("\n2. Demonstrating CLI commands...")
    
    # Check CLI help
    print("\n   a) Checking CLI help:")
    result = subprocess.run([
        "python", "src/inference/cli.py", "--help"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ CLI is working")
        print("   Available commands:", result.stdout.split("Commands:")[1].split("\n")[0] if "Commands:" in result.stdout else "Check --help")
    else:
        print(f"   ⚠️  CLI help failed: {result.stderr}")
        return
    
    # Build index
    print("\n   b) Building search index:")
    result = subprocess.run([
        "python", "src/inference/cli.py", "build-index",
        "--corpus", "sample_data/corpus",
        "--output", "sample_data/index"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Index built successfully")
        print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"   ❌ Index build failed: {result.stderr}")
        return
    
    # Query the system
    print("\n   c) Querying the system:")
    
    queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "What are neural networks?"
    ]
    
    for query in queries:
        result = subprocess.run([
            "python", "src/inference/cli.py", "query",
            "--query-text", query,
            "--top-k", "2"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   Query: {query}")
            try:
                # Parse JSON output if available
                output = result.stdout
                if output.strip().startswith("{"):
                    data = json.loads(output)
                    if "results" in data:
                        for i, res in enumerate(data["results"][:2], 1):
                            print(f"      {i}. {res}")
                else:
                    print(f"      Result: {output.strip()}")
            except:
                print(f"      Result: {result.stdout.strip()}")
        else:
            print(f"   ❌ Query failed: {result.stderr}")
    
    # Check system status
    print("\n   d) Checking system status:")
    result = subprocess.run([
        "python", "src/inference/cli.py", "status"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ System status:")
        print(f"   {result.stdout.strip()}")
    else:
        print(f"   ❌ Status check failed: {result.stderr}")
    
    # Run demo script
    print("\n   e) Running quick demo:")
    result = subprocess.run([
        "bash", "scripts/quick_demo.sh"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Demo completed")
        print("   Output preview:", result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout)
    else:
        print(f"   ⚠️  Demo failed: {result.stderr}")
    
    print("\n✅ CLI demonstration completed!")

if __name__ == "__main__":
    cli_demo()
```

## Intermediate Examples

### Multi-Expert System

Using multiple specialized experts:

```python
#!/usr/bin/env python3
"""
Multi-expert system demonstration
"""

from mini_biai_1.experts import LanguageExpert, VisionExpert, SymbolicExpert
from mini_biai_1.coordinator import SpikingRouter
from mini_biai_1.configs import load_config

def multi_expert_demo():
    """Demonstrate multi-expert system."""
    
    print("🧩 Multi-Expert System Demo")
    print("="*50)
    
    # Load configuration
    config = load_config("configs/multi_expert.yaml")
    
    # Initialize router
    router = SpikingRouter(config.coordinator)
    
    # Create specialized experts
    print("1. Creating experts...")
    
    # Language expert
    language_expert = LanguageExpert(config.language)
    print("   ✅ Language Expert created")
    
    # Symbolic expert (if available)
    try:
        symbolic_expert = SymbolicExpert(config.symbolic)
        print("   ✅ Symbolic Expert created")
    except Exception as e:
        print(f"   ⚠️  Symbolic Expert unavailable: {e}")
        symbolic_expert = None
    
    # Add experts to router
    router.add_expert("language", language_expert)
    if symbolic_expert:
        router.add_expert("symbolic", symbolic_expert)
    
    print(f"   📊 Total experts: {len(router.get_experts())}")
    
    # Process different types of inputs
    print("\n2. Processing inputs...")
    
    # Text processing (routes to language expert)
    text_input = "Explain how neural networks process information."
    print(f"\n   Text input: {text_input}")
    
    try:
        text_result = router.forward(text_input)
        print(f"   Result: {text_result[:100]}...")
    except Exception as e:
        print(f"   ❌ Text processing failed: {e}")
    
    # Symbolic reasoning (routes to symbolic expert)
    if symbolic_expert:
        symbolic_input = "If A implies B, and B implies C, what can we conclude about A and C?"
        print(f"\n   Symbolic input: {symbolic_input}")
        
        try:
            symbolic_result = router.forward(symbolic_input)
            print(f"   Result: {symbolic_result[:100]}...")
        except Exception as e:
            print(f"   ❌ Symbolic processing failed: {e}")
    
    # Routing statistics
    print("\n3. Routing statistics...")
    stats = router.get_routing_stats()
    print(f"   Expert usage: {stats.expert_usage}")
    print(f"   Average routing time: {stats.avg_routing_time:.4f}s")
    print(f"   Routing efficiency: {stats.routing_efficiency:.3f}")
    
    # Performance comparison
    print("\n4. Performance comparison...")
    
    import time
    
    # Test single expert vs multi-expert
    single_expert = LanguageExpert(config.language)
    
    test_inputs = [
        "What is machine learning?",
        "Explain neural networks",
        "Describe deep learning"
    ]
    
    print("\n   Single expert performance:")
    for inp in test_inputs:
        start = time.time()
        single_result = single_expert.process(inp)
        single_time = time.time() - start
        print(f"      {inp[:30]}... → {single_time:.4f}s")
    
    print("\n   Multi-expert performance:")
    for inp in test_inputs:
        start = time.time()
        multi_result = router.forward(inp)
        multi_time = time.time() - start
        print(f"      {inp[:30]}... → {multi_time:.4f}s")
    
    print("\n✅ Multi-expert demonstration completed!")

if __name__ == "__main__":
    multi_expert_demo()
```

### Training Examples

Basic training workflows:

```python
#!/usr/bin/env python3
"""
Training workflow examples
"""

from mini_biai_1.training import RoutingTrainer, SyntheticRoutingDataset
from mini_biai_1.coordinator import MiniBiAiCoordinator
from mini_biai_1.configs import load_config

def basic_training_example():
    """Basic training workflow example."""
    
    print("🎯 Basic Training Example")
    print("="*50)
    
    # Load training configuration
    config = load_config("configs/training_demo.yaml")
    
    # Create coordinator and trainer
    coordinator = MiniBiAiCoordinator(config)
    trainer = RoutingTrainer(coordinator, config.training)
    
    print("1. Creating synthetic dataset...")
    
    # Create synthetic routing dataset
    dataset = SyntheticRoutingDataset(
        num_samples=1000,
        input_dim=100,
        num_experts=4,
        sequence_length=32
    )
    
    print(f"   ✅ Dataset created with {len(dataset)} samples")
    print(f"   - Input dimension: {dataset.input_dim}")
    print(f"   - Number of experts: {dataset.num_experts}")
    print(f"   - Sequence length: {dataset.sequence_length}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    print(f"\n2. Dataset split:")
    print(f"   - Training: {len(train_dataset)} samples")
    print(f"   - Validation: {len(val_dataset)} samples")
    print(f"   - Test: {len(test_dataset)} samples")
    
    # Training loop
    print("\n3. Starting training...")
    
    max_epochs = 5
    for epoch in range(max_epochs):
        print(f"\n   Epoch {epoch + 1}/{max_epochs}")
        
        # Train one epoch
        train_results = trainer.train_epoch(train_dataset)
        print(f"   ✅ Training loss: {train_results.loss:.4f}")
        
        # Validate
        val_results = trainer.validate(val_dataset)
        print(f"   ✅ Validation accuracy: {val_results.accuracy:.4f}")
        
        # Early stopping check
        if epoch > 0 and val_results.accuracy > best_val_acc:
            best_val_acc = val_results.accuracy
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            print(f"   💾 Checkpoint saved")
        
        # Learning rate scheduling
        trainer.scheduler.step()
    
    # Final evaluation
    print("\n4. Final evaluation...")
    test_results = trainer.evaluate(test_dataset)
    print(f"   Final test accuracy: {test_results.accuracy:.4f}")
    print(f"   Final test loss: {test_results.loss:.4f}")
    
    # Training summary
    print("\n5. Training summary:")
    summary = trainer.get_training_summary()
    print(f"   Total training time: {summary.total_time:.2f}s")
    print(f"   Best validation accuracy: {summary.best_val_acc:.4f}")
    print(f"   Final test accuracy: {summary.final_test_acc:.4f}")
    
    print("\n✅ Basic training example completed!")

def custom_training_example():
    """Custom training with validation callbacks."""
    
    print("\n🎯 Custom Training Example")
    print("="*50)
    
    from mini_biai_1.training import TrainingCallback
    
    class ValidationCallback(TrainingCallback):
        """Custom validation callback."""
        
        def __init__(self, validation_dataset):
            self.val_dataset = validation_dataset
            self.best_accuracy = 0.0
            
        def on_epoch_end(self, trainer, epoch, results):
            val_results = trainer.validate(self.val_dataset)
            
            if val_results.accuracy > self.best_accuracy:
                self.best_accuracy = val_results.accuracy
                trainer.save_checkpoint(f"best_model_epoch_{epoch}.pt")
                print(f"   💾 New best model saved (accuracy: {val_results.accuracy:.4f})")
            
            print(f"   📊 Epoch {epoch} - Val accuracy: {val_results.accuracy:.4f}")
    
    # Setup training with callback
    config = load_config("configs/custom_training.yaml")
    coordinator = MiniBiAiCoordinator(config)
    trainer = RoutingTrainer(coordinator, config.training)
    
    # Create dataset
    dataset = SyntheticRoutingDataset(num_samples=500, input_dim=50, num_experts=3)
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    # Create callback
    callback = ValidationCallback(val_dataset)
    trainer.add_callback(callback)
    
    # Train with callback
    print("Training with custom callback...")
    results = trainer.train(train_dataset, max_epochs=3, callbacks=[callback])
    
    print(f"\n✅ Custom training completed!")
    print(f"Best validation accuracy: {callback.best_accuracy:.4f}")

if __name__ == "__main__":
    basic_training_example()
    custom_training_example()
```

### API Integration

Creating and using REST APIs:

```python
#!/usr/bin/env python3
"""
API integration example
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from mini_biai_1 import create_pipeline
from mini_biai_1.configs import load_config

# Define request/response models
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9

class MemoryRequest(BaseModel):
    key: str
    value: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class GenerationResponse(BaseModel):
    result: str
    processing_time: float
    tokens_generated: int

class MemoryResponse(BaseModel):
    success: bool
    message: str

class QueryResponse(BaseModel):
    results: list
    processing_time: float

# Initialize FastAPI app
app = FastAPI(
    title="Mini-Biai-1 API",
    description="REST API for mini-biai-1 brain-inspired AI system",
    version="1.0.0"
)

# Global pipeline (initialized on startup)
pipeline = None

@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    try:
        config = load_config("configs/api_config.yaml")
        pipeline = create_pipeline(config)
        print("✅ Pipeline initialized successfully")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        pipeline = None

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Mini-Biai-1 API",
        "version": "1.0.0",
        "status": "running" if pipeline else "initialization_failed"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {"status": "healthy", "pipeline": "ready"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    """Generate text from prompt."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    import time
    start_time = time.time()
    
    try:
        result = pipeline.generate(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        processing_time = time.time() - start_time
        tokens_generated = len(result.split())
        
        return GenerationResponse(
            result=result,
            processing_time=processing_time,
            tokens_generated=tokens_generated
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store", response_model=MemoryResponse)
async def store_memory(request: MemoryRequest):
    """Store information in memory system."""
    # This would integrate with the memory system
    # For demo purposes, just return success
    return MemoryResponse(
        success=True,
        message=f"Stored key '{request.key}' in memory"
    )

@app.post("/memory/query", response_model=QueryResponse)
async def query_memory(request: QueryRequest):
    """Query memory system."""
    import time
    start_time = time.time()
    
    # Mock query results for demo
    mock_results = [
        {"key": "brain_structure", "value": "The brain consists of neurons", "score": 0.95},
        {"key": "neural_networks", "value": "Networks of interconnected nodes", "score": 0.87},
        {"key": "machine_learning", "value": "AI subset for pattern recognition", "score": 0.82}
    ]
    
    processing_time = time.time() - start_time
    
    return QueryResponse(
        results=mock_results[:request.top_k],
        processing_time=processing_time
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Mock statistics
    return {
        "queries_processed": 1250,
        "average_response_time": 0.245,
        "memory_usage": "156 MB",
        "uptime": "2h 15m 30s"
    }

def start_api_server():
    """Start the API server."""
    print("🚀 Starting Mini-Biai-1 API Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    start_api_server()
```

To test the API:

```bash
# Start the server
python api_example.py

# In another terminal, test the endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "The brain processes information", "max_length": 50}'
```

## Advanced Examples

### Custom Experts

Creating custom expert modules:

```python
#!/usr/bin/env python3
"""
Custom expert module example
"""

import torch
import torch.nn as nn
from mini_biai_1.experts import BaseExpert
from mini_biai_1.interfaces import ExpertInput, ExpertOutput

class SentimentExpert(BaseExpert):
    """Custom sentiment analysis expert."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Simple sentiment classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # Negative, Neutral, Positive
            nn.LogSoftmax(dim=-1)
        )
        
        self.expert_type = "sentiment"
        
    def forward(self, input_data: ExpertInput) -> ExpertOutput:
        """Process sentiment analysis."""
        
        # Extract text from input
        text = input_data.text
        embeddings = self.encode_text(text)
        
        # Classify sentiment
        logits = self.classifier(embeddings)
        sentiment_scores = torch.exp(logits)
        
        # Determine sentiment
        sentiment = torch.argmax(sentiment_scores, dim=-1)
        confidence = torch.max(sentiment_scores, dim=-1)[0]
        
        # Map to labels
        labels = ["negative", "neutral", "positive"]
        sentiment_label = labels[sentiment.item()]
        
        return ExpertOutput(
            result={
                "sentiment": sentiment_label,
                "confidence": confidence.item(),
                "scores": {
                    "negative": sentiment_scores[0][0].item(),
                    "neutral": sentiment_scores[0][1].item(),
                    "positive": sentiment_scores[0][2].item()
                }
            },
            confidence=confidence.item(),
            processing_time=0.1  # Mock processing time
        )
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Simple text encoding (placeholder)."""
        # In real implementation, use proper embeddings
        # For demo, create random embeddings
        return torch.randn(1, self.config.input_dim)

class MathExpert(BaseExpert):
    """Custom mathematical reasoning expert."""
    
    def __init__(self, config):
        super().__init__(config)
        self.expert_type = "math"
        
    def forward(self, input_data: ExpertInput) -> ExpertOutput:
        """Process mathematical operations."""
        
        import re
        import math
        
        text = input_data.text
        
        # Simple mathematical pattern matching
        patterns = {
            r'(\d+)\s*\+\s*(\d+)': lambda m: int(m.group(1)) + int(m.group(2)),
            r'(\d+)\s*-\s*(\d+)': lambda m: int(m.group(1)) - int(m.group(2)),
            r'(\d+)\s*\*\s*(\d+)': lambda m: int(m.group(1)) * int(m.group(2)),
            r'(\d+)\s*/\s*(\d+)': lambda m: int(m.group(1)) / int(m.group(2)) if int(m.group(2)) != 0 else "division by zero",
            r'sqrt\((\d+)\)': lambda m: math.sqrt(int(m.group(1)))
        }
        
        result = None
        operation = None
        
        for pattern, func in patterns.items():
            match = re.search(pattern, text)
            if match:
                try:
                    result = func(match)
                    operation = match.group(0)
                    break
                except:
                    continue
        
        if result is None:
            result = "No mathematical operation detected"
            operation = "none"
        
        return ExpertOutput(
            result={
                "operation": operation,
                "result": result,
                "type": "mathematical_reasoning"
            },
            confidence=1.0 if operation != "none" else 0.0,
            processing_time=0.05
        )

def custom_experts_demo():
    """Demonstrate custom experts."""
    
    print("🔧 Custom Experts Demo")
    print("="*50)
    
    # Create expert configurations
    sentiment_config = type('Config', (), {
        'input_dim': 512,
        'hidden_dim': 256
    })()
    
    math_config = type('Config', (), {})()
    
    # Create experts
    sentiment_expert = SentimentExpert(sentiment_config)
    math_expert = MathExpert(math_config)
    
    print("1. Custom experts created")
    print(f"   - Sentiment Expert: {sentiment_expert.expert_type}")
    print(f"   - Math Expert: {math_expert.expert_type}")
    
    # Test sentiment expert
    print("\n2. Testing Sentiment Expert:")
    
    sentiment_inputs = [
        "I love this product!",
        "This is terrible.",
        "It's okay, I guess."
    ]
    
    for text in sentiment_inputs:
        input_data = ExpertInput(text=text)
        output = sentiment_expert(input_data)
        
        print(f"   Text: '{text}'")
        print(f"   Sentiment: {output.result['sentiment']} (confidence: {output.confidence:.3f})")
        print()
    
    # Test math expert
    print("3. Testing Math Expert:")
    
    math_inputs = [
        "What is 15 + 27?",
        "Calculate 100 / 4",
        "What's the square root of 16?",
        "What is 5 times 6?"
    ]
    
    for text in math_inputs:
        input_data = ExpertInput(text=text)
        output = math_expert(input_data)
        
        print(f"   Input: {text}")
        print(f"   Operation: {output.result['operation']}")
        print(f"   Result: {output.result['result']}")
        print()
    
    # Integration with router
    print("4. Integrating with routing system:")
    
    from mini_biai_1.coordinator import SpikingRouter
    
    router = SpikingRouter(type('Config', (), {
        'spike_threshold': 1.0,
        'time_steps': 8
    })())
    
    router.add_expert("sentiment", sentiment_expert)
    router.add_expert("math", math_expert)
    
    # Test routing
    test_inputs = [
        ("I'm feeling great today!", "sentiment"),
        ("Calculate 2 + 2", "math"),
        ("What's 3 * 4?", "math")
    ]
    
    for text, expected_type in test_inputs:
        print(f"\n   Input: '{text}'")
        try:
            result = router.forward(text)
            print(f"   Routed successfully (expected: {expected_type})")
        except Exception as e:
            print(f"   Routing failed: {e}")
    
    print("\n✅ Custom experts demonstration completed!")

if __name__ == "__main__":
    custom_experts_demo()
```

## Tutorials

### Step-by-Step Tutorial: Building a Complete Application

```python
#!/usr/bin/env python3
"""
Complete application tutorial: Building a Q&A system
"""

from mini_biai_1 import create_pipeline
from mini_biai_1.memory import HierarchicalMemory
from mini_biai_1.configs import load_config

class QASystem:
    """Complete Q&A system using mini-biai-1."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.pipeline = create_pipeline(self.config)
        self.memory = HierarchicalMemory(self.config.memory)
        
        # Knowledge base
        self.knowledge_base = {
            "What is AI?": "Artificial Intelligence is the simulation of human intelligence in machines.",
            "What is ML?": "Machine Learning is a subset of AI that enables computers to learn.",
            "What is DL?": "Deep Learning is a subset of ML using neural networks with many layers."
        }
        
        # Load knowledge into memory
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load knowledge base into memory."""
        for question, answer in self.knowledge_base.items():
            self.memory.store(f"qa:{question}", answer)
            self.memory.store(f"knowledge:{question.lower()}", answer)
    
    def answer_question(self, question: str) -> str:
        """Answer a question using the Q&A system."""
        
        # First, try to find exact match in knowledge base
        exact_match = self.memory.retrieve(f"qa:{question}")
        if exact_match:
            return f"From knowledge base: {exact_match}"
        
        # Search for similar questions
        similar_results = self.memory.search(question, top_k=3)
        
        if similar_results:
            # Get the best match
            best_key, score = similar_results[0]
            
            if score > 0.7:  # High confidence match
                answer = self.memory.retrieve(best_key)
                return f"Similar question found: {answer}"
        
        # Generate answer using the language model
        prompt = f"Q: {question}\nA:"
        generated_answer = self.pipeline.generate(prompt, max_length=100)
        
        # Store new question-answer pair
        self.memory.store(f"qa:{question}", generated_answer)
        
        return f"Generated answer: {generated_answer}"
    
    def chat(self):
        """Interactive chat interface."""
        print("🤖 Mini-Biai-1 Q&A System")
        print("="*40)
        print("Type 'quit' to exit, 'help' for commands")
        print()
        
        while True:
            try:
                question = input("You: ").strip()
                
                if question.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif question.lower() == 'help':
                    self._show_help()
                    continue
                elif not question:
                    continue
                
                # Get answer
                answer = self.answer_question(question)
                print(f"Bot: {answer}")
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("\n💡 Q&A System Help:")
        print("- Ask questions about AI, ML, or DL")
        print("- The system will search knowledge base first")
        print("- If no match found, it will generate an answer")
        print("- New Q&A pairs are stored automatically")
        print("- Type 'quit' to exit")
        print()

def tutorial_complete_application():
    """Run the complete application tutorial."""
    
    print("📚 Complete Application Tutorial")
    print("Building a Q&A System with Mini-Biai-1")
    print("="*50)
    
    try:
        # Create Q&A system
        qa_system = QASystem("configs/qa_system.yaml")
        print("✅ Q&A System initialized")
        
        # Test with sample questions
        test_questions = [
            "What is AI?",
            "How does machine learning work?",
            "What is deep learning?",
            "What is neural networks?"
        ]
        
        print("\n🤔 Testing with sample questions:")
        for question in test_questions:
            answer = qa_system.answer_question(question)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
        
        # Start interactive chat (comment out for automated demo)
        # qa_system.chat()
        
        print("\n✅ Tutorial completed successfully!")
        print("\nNext steps:")
        print("1. Run qa_system.chat() for interactive mode")
        print("2. Extend knowledge_base with domain-specific information")
        print("3. Implement more sophisticated question matching")
        print("4. Add conversation history tracking")
        
    except FileNotFoundError:
        print("⚠️  Configuration file not found, using defaults")
        # Create simple demo without full configuration
        print("Creating simplified Q&A demo...")
        
        print("Demo questions:")
        demo_questions = ["What is artificial intelligence?", "How do neural networks work?"]
        
        for question in demo_questions:
            print(f"Q: {question}")
            print("A: This would be answered by the Q&A system...")
            print()

if __name__ == "__main__":
    tutorial_complete_application()
```

---

## Real-World Applications

### Research Applications

1. **Cognitive Modeling**: Model human cognitive processes
2. **Neuroscience Research**: Study brain-inspired algorithms
3. **Adaptive Systems**: Build self-improving AI systems
4. **Multi-Modal AI**: Combine text, vision, and reasoning

### Production Applications

1. **Real-Time Processing**: Low-latency inference systems
2. **Resource-Constrained Environments**: Efficient AI for edge devices
3. **Interactive Systems**: Conversational AI and chatbots
4. **Knowledge Management**: Intelligent information retrieval

## Next Steps

Explore more advanced examples:

- [Architecture Documentation](../architecture/overview.md) - System design
- [Training Guide](../training/index.md) - Advanced training techniques
- [API Reference](../api/index.md) - Complete API documentation
- [Developer Guides](../developer-guides/contributing.md) - Contributing to the project

---

*Have an example to share? Submit it to our [Examples Repository](https://github.com/mini-biai-1/mini-biai-1-examples) or create a pull request!*