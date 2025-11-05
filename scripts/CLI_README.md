# Inference System CLI Documentation

A comprehensive command-line interface for corpus indexing and intelligent query processing, built with Python and the Fire library for easy CLI creation.

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:

```bash
pip install fire
```

3. Make scripts executable:

```bash
chmod +x scripts/quick_demo.sh
```

## Quick Start

Run the demo to see the system in action:

```bash
bash scripts/quick_demo.sh
```

## CLI Commands

### Build Index

Build an index from corpus files:

```bash
python3 src/inference/cli.py build-index --corpus data/corpus --index-path data/index --chunk-size 512
```

**Parameters:**
- `--corpus`: Path to corpus directory (required)
- `--index-path`: Output directory for index (defaults to `./data/index`)
- `--rebuild`: Force rebuild existing index
- `--verbose`: Enable verbose output

**Examples:**

```bash
# Basic index building
python3 src/inference/cli.py build-index --corpus data/corpus

# Rebuild existing index
python3 src/inference/cli.py build-index --corpus data/corpus --rebuild

# Custom chunk size with verbose output
python3 src/inference/cli.py build-index --corpus data/corpus --chunk-size 256 --verbose
```

### Query the System

Search the indexed content:

```bash
python3 src/inference/cli.py query --query-text "machine learning" --k 5 --output-format text
```

**Parameters:**
- `--query-text`: Query string (required)
- `--k`: Number of results to return (default: 5)
- `--index-path`: Path to index directory
- `--corpus-path`: Path to corpus directory
- `--output-format`: Output format (`text`, `json`, `csv`)
- `--save-results`: Optional path to save results

**Examples:**

```bash
# Basic query
python3 src/inference/cli.py query --query-text "What is AI?"

# Query with more results
python3 src/inference/cli.py query --query-text "climate change" --k 10

# JSON output with saved results
python3 src/inference/cli.py query \
    --query-text "renewable energy" \
    --k 5 \
    --output-format json \
    --save-results results/energy_search.json

# CSV output for spreadsheet analysis
python3 src/inference/cli.py query \
    --query-text "quantum computing" \
    --k 10 \
    --output-format csv \
    --save-results results/quantum_results.csv
```

### Check System Status

View system information and health:

```bash
python3 src/inference/cli.py status
```

Shows:
- Corpus path and index path
- File existence checks
- Number of corpus files
- Sample file names

### Configure Settings

Update system configuration:

```bash
python3 src/inference/cli.py config-set chunk_size 256
```

**Available Configuration Options:**
- `chunk_size`: Size of text chunks for indexing (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `index_path`: Default index location
- `corpus_path`: Default corpus location

### Run Demo

Execute the full demonstration:

```bash
python3 src/inference/cli.py demo
```

## File Structure

```
project/
├── src/
│   └── inference/
│       └── cli.py              # Main CLI interface
├── scripts/
│   ├── build_index.py          # Corpus indexing script
│   └── quick_demo.sh          # End-to-end demo script
├── data/
│   ├── corpus/                 # Sample corpus files
│   │   ├── artificial_intelligence.txt
│   │   ├── machine_learning.txt
│   │   ├── climate_change.txt
│   │   ├── quantum_computing.txt
│   │   └── blockchain_technology.txt
│   └── index/                  # Generated index files
├── config.json                 # System configuration
└── CLI_README.md              # This file
```

## Indexing Process

The indexing system:

1. **File Discovery**: Scans corpus directory for `.txt` files
2. **Text Processing**: Reads and validates file content
3. **Chunking**: Splits text into overlapping chunks (configurable size)
4. **Index Building**: Creates keyword and file indexes
5. **Metadata Generation**: Stores processing statistics and configuration

### Generated Index Files

The indexing process creates several files in the output directory:

- `index.json`: Main index file with metadata and search structures
- `documents.json`: Individual document chunks with content and metadata
- `metadata.json`: Processing metadata and configuration
- `indexing_report.json`: Detailed statistics about the indexing process

## Query Features

### Output Formats

**Text Format** (default):
```
Found 3 results:

1. machine_learning.txt (score: 0.952)
   Machine Learning is a branch of artificial intelligence that focuses on building systems...

2. artificial_intelligence.txt (score: 0.834)
   Artificial Intelligence (AI) represents one of the most significant technological advances...
```

**JSON Format**:
```json
[
  {
    "file": "machine_learning.txt",
    "score": 0.952,
    "content": "Machine Learning is a branch of artificial intelligence..."
  }
]
```

**CSV Format**:
```csv
file,score,content
machine_learning.txt,0.952,"Machine Learning is a branch of artificial intelligence..."
```

### Search Capabilities

- **Keyword Search**: Finds documents containing specific terms
- **Relevance Scoring**: Ranks results by relevance
- **Multiple Results**: Returns top K most relevant documents
- **Content Extraction**: Provides relevant text snippets

## Configuration

The system uses a `config.json` file for configuration:

```json
{
  "index_path": "./data/index",
  "corpus_path": "./data/corpus",
  "model_name": "default",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

### Configuration Management

Use the CLI to update configuration:

```bash
# Update chunk size
python3 src/inference/cli.py config-set chunk_size 256

# Update index path
python3 src/inference/cli.py config-set index_path ./custom/index/path

# Update overlap
python3 src/inference/cli.py config-set chunk_overlap 100
```

## Error Handling

The system provides comprehensive error handling:

### Input Validation

- Required parameter checking
- File path validation
- Range validation for numeric parameters
- Format validation for output options

### File System Errors

- Missing corpus directory
- Inaccessible index files
- Permission errors
- Corrupted index files

### Processing Errors

- Invalid text encoding
- Empty or malformed files
- Memory limitations
- Index corruption

### Error Messages

All errors include:
- Clear description of the problem
- Suggested solutions
- Debug information when verbose mode is enabled

## Reproducibility

The system ensures reproducible results through:

### Deterministic Processing

- Consistent chunking algorithms
- Stable keyword extraction
- Predictable document ID generation
- Deterministic scoring methods

### Configuration Tracking

- Saved configuration files
- Processing parameters in metadata
- Index version tracking
- Creation timestamp recording

### Content Integrity

- Document hash verification
- File content checksums
- Index consistency checks
- Data validation rules

## Logging

The system provides multiple logging levels:

### Log Levels

- **ERROR**: Critical errors that prevent operation
- **WARNING**: Non-critical issues and potential problems
- **INFO**: General operational information
- **DEBUG**: Detailed processing information

### Log Output

Logs are displayed on stderr with timestamps and levels:

```
2024-11-06 00:43:41,123 - INFO - Processing corpus: ./data/corpus
2024-11-06 00:43:41,124 - INFO - Found 5 text files
2024-11-06 00:43:41,125 - DEBUG - Processing file: artificial_intelligence.txt
```

### Silent Mode

Suppress all output except errors:

```bash
python3 src/inference/cli.py build-index --corpus data/corpus --silent
```

## Advanced Usage

### Custom Corpus Creation

1. Create directory: `mkdir data/my_corpus`
2. Add `.txt` files with relevant content
3. Build index: `python3 src/inference/cli.py build-index --corpus data/my_corpus`
4. Query content: `python3 src/inference/cli.py query --query-text "your search"`

### Batch Processing

Process multiple queries:

```bash
#!/bin/bash
queries=("AI applications" "climate solutions" "quantum computing")
for query in "${queries[@]}"; do
    python3 src/inference/cli.py query --query-text "$query" --save-results "results/${query// /_}.json"
done
```

### Integration with Other Tools

```python
# Example Python script
import subprocess
import json

def search_corpus(query):
    result = subprocess.run([
        'python3', 'src/inference/cli.py', 'query',
        '--query-text', query,
        '--output-format', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        return json.loads(result.stdout)
    else:
        raise RuntimeError(f"Query failed: {result.stderr}")

# Use the function
results = search_corpus("machine learning algorithms")
for result in results:
    print(f"Found: {result['file']} (score: {result['score']:.3f})")
```

## Troubleshooting

### Common Issues

**Index not found**:
```
ERROR - Index not found: ./data/index. Run 'build-index' first.
```
Solution: Run `python3 src/inference/cli.py build-index --corpus data/corpus`

**No results found**:
```
No results found.
```
Solution: 
- Verify corpus files exist in the specified directory
- Check that index was built successfully
- Try different query terms

**Permission errors**:
```
ERROR - Could not read file: permission denied
```
Solution: Check file permissions and ensure read access to corpus files

**Memory issues**:
```
ERROR - Memory allocation failed
```
Solution: Reduce chunk size for large corpora

### Debug Mode

Enable verbose output for detailed information:

```bash
python3 src/inference/cli.py build-index --corpus data/corpus --verbose
python3 src/inference/cli.py query --query-text "AI" --verbose
```

### System Diagnostics

Check system status:

```bash
python3 src/inference/cli.py status
```

Review index reports:

```bash
cat data/index/indexing_report.json
```

## Performance

### Optimization Tips

- **Chunk Size**: Larger chunks capture more context but use more memory
- **Chunk Overlap**: Higher overlap improves recall but increases index size
- **File Size**: Smaller files process faster but may reduce context
- **Memory**: Monitor memory usage during large corpus processing

### Scalability

The system is designed to handle:
- **Corpus Size**: Thousands of text files
- **File Size**: Individual files up to several MB
- **Index Size**: Indexes can grow to hundreds of MB
- **Query Load**: Hundreds of queries per minute

## Development

### Adding New Features

The CLI is built with the Fire library, making it easy to add new commands:

```python
def new_command(self, parameter: str) -> bool:
    """New CLI command"""
    try:
        # Implementation here
        return True
    except Exception as e:
        logger.error(f"New command failed: {e}")
        return False
```

### Extending Search

Modify `scripts/build_index.py` to add new search algorithms or indexing strategies.

### Custom Output Formats

Add new format handlers in the `_format_output` method of `src/inference/cli.py`.

## Examples

### Complete Workflow

```bash
# 1. Build index from corpus
python3 src/inference/cli.py build-index --corpus data/corpus --verbose

# 2. Check system status
python3 src/inference/cli.py status

# 3. Run a few example queries
python3 src/inference/cli.py query --query-text "artificial intelligence" --k 3
python3 src/inference/cli.py query --query-text "climate change solutions" --k 5 --output-format json
python3 src/inference/cli.py query --query-text "quantum applications" --save-results results/quantum.json

# 4. Configure for different use case
python3 src/inference/cli.py config-set chunk_size 256
python3 src/inference/cli.py config-set chunk_overlap 100

# 5. Rebuild with new settings
python3 src/inference/cli.py build-index --rebuild

# 6. Final verification
python3 src/inference/cli.py status
```

### Research Use Case

```bash
# Academic paper analysis
python3 src/inference/cli.py build-index --corpus papers/ --chunk-size 1024

# Literature review queries
python3 src/inference/cli.py query --query-text "machine learning methods" --k 10 --output-format json > results/ml_methods.json
python3 src/inference/cli.py query --query-text "neural networks" --k 10 --output-format json > results/neural_networks.json

# Comparative analysis
python3 src/inference/cli.py query --query-text "optimization algorithms" --k 5 --save-results results/optimization.csv
```

---

For questions or issues, check the system status and logs for detailed diagnostic information.