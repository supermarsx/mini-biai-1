# Enhanced CLI Quick Reference

## Essential Commands

### Basic Usage
```bash
# Show help
python3 src/inference/cli.py help

# Check status
python3 src/inference/cli.py status
python3 src/inference/cli.py status --detailed

# Run demo
python3 src/inference/cli.py demo
```

### Query Examples

#### Basic Queries
```bash
python3 src/inference/cli.py query "What is AI?"
python3 src/inference/cli.py query "Explain machine learning"
python3 src/inference/cli.py query "How does climate work?"
```

#### With Affect Detection
```bash
python3 src/inference/cli.py query "I'm excited about AI!" --affect-demo
python3 src/inference/cli.py query "I'm worried about technology" --affect-demo
python3 src/inference/cli.py query "This is amazing!" --affect-demo --visualize-routing
```

#### Manual Expert Selection
```bash
# Single expert
python3 src/inference/cli.py query "Write a story" --experts language --expert-mode
python3 src/inference/cli.py query "Solve logic puzzle" --experts symbolic --expert-mode

# Multiple experts
python3 src/inference/cli.py query "Complex analysis" --experts language,symbolic --expert-mode
python3 src/inference/cli.py query "Technical explanation" --experts symbolic,vision --expert-mode
```

#### Online Learning
```bash
python3 src/inference/cli.py query "Complex problem" --auto-learning
python3 src/inference/cli.py query "Deep learning concepts" --auto-learning --visualize-routing
```

#### Comprehensive Features
```bash
# All features enabled
python3 src/inference/cli.py query "How will AI change education?" \
  --affect-demo \
  --auto-learning \
  --visualize-routing \
  --expert-mode \
  --experts language,symbolic \
  --output-format detailed

# Save results
python3 src/inference/cli.py query "Research topic" \
  --affect-demo \
  --output-format json \
  --save-results results.json
```

### Output Formats

#### Text (Default)
```bash
python3 src/inference/cli.py query "AI question"
```

#### JSON Format
```bash
python3 src/inference/cli.py query "AI question" --output-format json
```

#### Detailed Format
```bash
python3 src/inference/cli.py query "AI question" --output-format detailed
```

## Flag Reference

| Flag | Description | Example |
|------|-------------|----------|
| `--experts LIST` | Comma-separated expert names | `--experts language,symbolic` |
| `--expert-mode` | Enable manual expert selection | `--expert-mode` |
| `--affect-demo` | Show affect detection | `--affect-demo` |
| `--auto-learning` | Enable online learning | `--auto-learning` |
| `--visualize-routing` | Show expert selection | `--visualize-routing` |
| `--output-format FORMAT` | Output format (text/json/detailed) | `--output-format json` |
| `--save-results PATH` | Save results to file | `--save-results result.json` |

## Available Experts

- **language**: Natural language processing and generation
- **vision**: Visual scene understanding and image analysis
- **symbolic**: Logical reasoning, mathematics, and structured analysis

## Advanced Usage

### Batch Processing
```bash
# Process multiple queries
for query in "AI benefits" "ML applications" "Climate solutions"; do
    python3 src/inference/cli.py query "$query" --save-results "results/${query// /_}.json"
done
```

### Configuration
```bash
# View current configuration
python3 src/inference/cli.py config-get

# Update chunk size
python3 src/inference/cli.py config-set chunk_size 256

# Update index path
python3 src/inference/cli.py config-set index_path ./custom/index
```

### Debugging
```bash
# Verbose output
python3 src/inference/cli.py query "test" --verbose

# Check system status
python3 src/inference/cli.py status --detailed

# Build/rebuild index
python3 src/inference/cli.py build-index --rebuild --verbose
```

For detailed documentation, see `CLI_README.md`.