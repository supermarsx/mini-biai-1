# Data Gatherer System

A comprehensive data collection system for web scraping, file processing, and data validation with batch processing capabilities.

## Features

### 🕷️ Web Crawling
- **Rate-limited concurrent crawling** with configurable delays
- **Respectful crawling** with robots.txt support
- **Content filtering** by file type and domain
- **Robust error handling** and retry mechanisms
- **Multiple output formats** (JSON, text files)

### 📄 File Processing
- **Multi-format support**: PDF, DOC, DOCX, TXT, MD, CSV, JSON, XML, HTML
- **Text extraction** with metadata preservation
- **Intelligent chunking** with configurable overlap
- **Batch processing** of directories
- **Comprehensive file format detection**

### ✅ Data Validation
- **Content cleaning** (HTML removal, whitespace normalization)
- **Quality filtering** (length, repetition, character patterns)
- **Duplicate detection** and removal
- **Language detection** (optional)
- **Validation scoring** and reporting

### 🚀 Batch Processing
- **Job orchestration** with progress tracking
- **Checkpoint system** for recovery
- **Combined workflows** (crawl → process → validate)
- **Statistics and monitoring**
- **Error recovery** and logging

## Quick Start

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt

# Or install core dependencies only
pip install aiohttp beautifulsoup4 PyPDF2 python-docx langdetect tqdm
```

### Basic Usage

```python
from src.data_gatherer import Config, WebCrawler, FileProcessor, DataValidator, BatchIngestor

# 1. Configure the system
config = Config()
config.crawler.max_pages = 100
config.crawler.delay_between_requests = 1.0

# 2. Web crawling
async def crawl_example():
    async with WebCrawler(config.crawler) as crawler:
        results = await crawler.crawl_urls([
            "https://example.com",
            "https://httpbin.org/html"
        ])
        crawler.save_results(results, "my_crawl_results.json")

# 3. File processing
processor = FileProcessor(config.file_processor)
results = processor.process_directory("./documents")

# 4. Data validation
validator = DataValidator(config.validator)
documents = [{"text": "Sample text to validate", "metadata": {"source": "example"}}]
validation_results = validator.validate_documents(documents)

# 5. Batch processing
ingestor = BatchIngestor(config)
job_id = ingestor.create_crawl_job(["https://example.com"])
ingestor.run_job(job_id)
```

## Configuration

The system uses a hierarchical configuration system with four main components:

### Crawler Configuration
```python
config = Config()
config.crawler.max_pages = 100
config.crawler.delay_between_requests = 1.0
config.crawler.concurrent_requests = 5
config.crawler.timeout = 30
config.crawler.user_agent = "YourApp/1.0"
config.crawler.respect_robots_txt = True
config.crawler.allowed_domains = ["example.com", "docs.example.com"]
```

### File Processor Configuration
```python
config.file_processor.supported_formats = ['.pdf', '.docx', '.txt']
config.file_processor.chunk_size = 1000
config.file_processor.overlap_size = 100
config.file_processor.extract_metadata = True
```

### Data Validator Configuration
```python
config.validator.min_text_length = 10
config.validator.max_text_length = 100000
config.validator.remove_html_tags = True
config.validator.normalize_whitespace = True
config.validator.filter_duplicates = True
config.validator.language_detection = False
```

### Batch Processing Configuration
```python
config.batch.batch_size = 10
config.batch.max_workers = 4
config.batch.checkpoint_interval = 5
config.batch.resume_from_checkpoint = True
```

## Component Details

### Web Crawler (`crawler.py`)

Advanced web crawler with rate limiting and concurrent processing:

```python
# Basic usage
async with WebCrawler() as crawler:
    results = await crawler.crawl_urls(urls, max_pages=50)

# With custom configuration
config = CrawlerConfig(
    max_pages=100,
    delay_between_requests=2.0,
    concurrent_requests=3,
    respect_robots_txt=True
)
async with WebCrawler(config) as crawler:
    results = await crawler.crawl_sitemap("https://example.com/sitemap.xml")
```

**Features:**
- Rate limiting with configurable delays
- Concurrent request handling
- robots.txt compliance
- Domain and path filtering
- Content type filtering
- Comprehensive error handling
- Progress tracking

### File Processor (`file_processor.py`)

Handles text extraction from various document formats:

```python
# Process single file
result = processor.process_file("document.pdf")

# Process directory
results = processor.process_directory("./documents", recursive=True)

# Custom configuration
config = FileProcessorConfig(
    chunk_size=500,
    extract_metadata=True,
    supported_formats=['.pdf', '.docx', '.txt']
)
```

**Supported Formats:**
- **PDF**: PyPDF2, pdfplumber
- **Word Documents**: DOC, DOCX (python-docx)
- **Text Files**: TXT, MD, RTF, ODT
- **Data Files**: CSV, JSON, XML
- **Web Files**: HTML

**Features:**
- Automatic format detection
- Metadata extraction
- Text chunking with overlap
- Batch processing
- Error recovery

### Data Validator (`data_validator.py`)

Content filtering and cleaning with validation scoring:

```python
# Validate single text
result = validator.validate_text("Sample text to validate")

# Validate multiple documents
documents = [{"text": "Doc 1", "metadata": {}}, {"text": "Doc 2", "metadata": {}}]
results = validator.validate_documents(documents)

# Filter valid documents
valid_results = validator.filter_documents(results, min_score=0.5)
```

**Validation Filters:**
- **Length Filter**: Min/max text length
- **HTML Tag Filter**: Remove HTML tags
- **Whitespace Filter**: Normalize whitespace
- **Special Character Filter**: Clean special characters
- **Duplicate Filter**: Remove duplicate content
- **Language Filter**: Detect and filter by language
- **Quality Filter**: Assess content quality

**Features:**
- Configurable validation rules
- Validation scoring (0.0-1.0)
- Filter statistics
- Issue tracking
- Report generation

### Batch Ingestor (`batch_ingest.py`)

Orchestrates complex data processing workflows:

```python
# Create individual jobs
crawl_job_id = ingestor.create_crawl_job(urls, job_name="Web Crawl")
file_job_id = ingestor.create_file_processing_job(file_paths, job_name="File Processing")
validation_job_id = ingestor.create_validation_job(documents, job_name="Data Validation")

# Create combined job
stages = [
    {"type": "crawl", "urls": ["https://example.com"], "max_pages": 10},
    {"type": "validate", "documents": [{"text": "Sample", "metadata": {}}]}
]
combined_job_id = ingestor.create_combined_job(stages, job_name="Complete Workflow")

# Run jobs
ingestor.run_job(crawl_job_id)
```

**Job Types:**
- **Crawl Jobs**: Web scraping with rate limiting
- **File Processing Jobs**: Document text extraction
- **Validation Jobs**: Data cleaning and filtering
- **Combined Jobs**: Multi-stage workflows

**Features:**
- Job management and tracking
- Progress monitoring
- Checkpoint system
- Error recovery
- Statistics collection
- Custom workflow stages

## Examples

See `examples.py` for comprehensive usage examples:

```bash
python src/data_gatherer/examples.py
```

This will demonstrate:
- Basic configuration
- Web crawling with rate limiting
- File processing and text extraction
- Data validation and cleaning
- Batch processing workflows
- Error handling
- Custom configurations

## File Structure

```
src/data_gatherer/
├── __init__.py              # Package initialization
├── config.py               # Configuration management
├── crawler.py              # Web crawler with rate limiting
├── file_processor.py       # Multi-format file processor
├── data_validator.py       # Content validation and cleaning
├── batch_ingest.py         # Batch processing orchestration
├── examples.py             # Usage examples
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Advanced Usage

### Custom Workflows

```python
# Create a custom data pipeline
async def custom_pipeline(urls, file_paths):
    # Stage 1: Crawl web content
    async with WebCrawler() as crawler:
        web_results = await crawler.crawl_urls(urls)
    
    # Stage 2: Process files
    processor = FileProcessor()
    file_results = processor.process_directory(file_paths)
    
    # Stage 3: Validate all content
    all_documents = []
    for result in web_results + file_results:
        if result.success:
            all_documents.append({
                "text": result.extracted_text,
                "metadata": result.metadata
            })
    
    validator = DataValidator()
    validated_results = validator.validate_documents(all_documents)
    
    # Stage 4: Save final results
    validator.save_validation_report(validated_results)
    return validated_results
```

### Performance Optimization

```python
# High-performance configuration
config = Config()
config.crawler.concurrent_requests = 10  # More concurrent requests
config.crawler.delay_between_requests = 0.5  # Faster rate
config.file_processor.chunk_size = 2000  # Larger chunks
config.batch.max_workers = 8  # More workers
config.batch.batch_size = 50  # Larger batches
```

### Monitoring and Statistics

```python
# Get processing statistics
stats = ingestor.get_statistics()
print(f"Processed {stats['global_stats']['total_items_processed']} items")
print(f"Success rate: {stats['completed_jobs']}/{stats['active_jobs']}")

# Monitor individual jobs
for job in ingestor.list_jobs():
    print(f"Job {job.job_id}: {job.status} ({job.progress*100:.1f}%)")
```

## Error Handling

The system includes comprehensive error handling:

```python
try:
    async with WebCrawler() as crawler:
        results = await crawler.crawl_urls(urls)
except Exception as e:
    logger.error(f"Crawling failed: {e}")
    
# Check individual results
for result in results:
    if not result.success:
        print(f"Failed to process {result.url}: {result.error}")
```

## Logging

Configure logging for debugging and monitoring:

```python
import logging

# Set log level
logging.getLogger('web_crawler').setLevel(logging.DEBUG)
logging.getLogger('file_processor').setLevel(logging.INFO)
logging.getLogger('data_validator').setLevel(logging.INFO)
logging.getLogger('batch_ingestor').setLevel(logging.INFO)

# Custom log format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
```

## Best Practices

### Respectful Web Crawling
- Always respect robots.txt (`respect_robots_txt = True`)
- Use appropriate delays (`delay_between_requests >= 1.0`)
- Set realistic user agents
- Limit concurrent requests (`concurrent_requests <= 5`)

### File Processing
- Check supported formats before processing
- Use appropriate chunk sizes (500-2000 characters)
- Enable metadata extraction for better results
- Handle large files with batch processing

### Data Validation
- Set appropriate length limits
- Use language detection for multilingual content
- Filter duplicates for large datasets
- Monitor validation scores

### Batch Processing
- Use checkpoints for long-running jobs
- Monitor progress regularly
- Handle errors gracefully
- Clean up completed jobs periodically

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **PDF Processing Issues**: Install PDF libraries
   ```bash
   pip install PyPDF2 pdfplumber
   ```

3. **Memory Issues**: Reduce batch size or chunk size
   ```python
   config.batch.batch_size = 5
   config.file_processor.chunk_size = 500
   ```

4. **Crawl Rate Limiting**: Increase delay between requests
   ```python
   config.crawler.delay_between_requests = 2.0
   ```

5. **File Format Not Supported**: Check supported formats
   ```python
   processor = FileProcessor()
   print(processor.get_supported_formats())
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Changelog

### v1.0.0
- Initial release
- Web crawler with rate limiting
- Multi-format file processor
- Data validation and cleaning
- Batch processing orchestration
- Comprehensive examples and documentation