#!/usr/bin/env python3
"""
BrainMod Step 2 - Corpus Indexer

This script builds a comprehensive search index for the BrainMod Step 2 documentation
and codebase. It processes text files, chunks them appropriately, and creates
vector embeddings for semantic search.

Features:
- Automatic file discovery and text extraction
- Intelligent text chunking with overlap
- Vector embedding generation using sentence transformers
- Efficient indexing with FAISS
- Persistent storage of index and metadata
- Search capabilities with relevance scoring

Usage:
    python build_index.py [--source-dir DIR] [--output-dir DIR] [--rebuild]
    
    --source-dir: Directory containing text files to index (default: current directory)
    --output-dir: Directory to store index files (default: ./search_index)
    --rebuild: Force rebuild of existing index
    --quiet: Suppress progress output

Examples:
    # Build index for current directory
    python build_index.py
    
    # Build index for specific directory
    python build_index.py --source-dir ./brainmod --output-dir ./my_index
    
    # Rebuild existing index
    python build_index.py --rebuild
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers is required. Install with: pip install sentence-transformers")
    sys.exit(1)

try:
    import faiss
except ImportError:
    print("Error: faiss is required. Install with: pip install faiss-cpu")
    sys.exit(1)

try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken not available. Token counting will be approximate.")
    tiktoken = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    source_file: str
    chunk_id: str
    start_line: int
    end_line: int
    word_count: int
    token_count: Optional[int] = None
    embedding: Optional[np.ndarray] = None

class TokenCounter:
    """Handles token counting for text chunks."""
    
    def __init__(self):
        self.encoder = None
        if tiktoken:
            try:
                # Use GPT-3.5/GPT-4 encoding for consistent tokenization
                self.encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken: {e}")
                self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        
        # Fallback: rough estimation
        return len(text.split()) * 1.3  # Rough estimate

class FileProcessor:
    """Handles file discovery and text extraction."""
    
    # File extensions to process
    TEXT_EXTENSIONS = {
        '.py', '.md', '.txt', '.rst', '.ipynb', '.json', '.yaml', '.yml',
        '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat',
        '.html', '.htm', '.css', '.js', '.ts', '.jsx', '.tsx',
        '.c', '.cpp', '.h', '.hpp', '.cs', '.java', '.go', '.rs',
        '.php', '.rb', '.swift', '.kt', '.scala', '.clj',
        '.sql', '.r', '.R', '.m', '.pl', '.lua',
        '.tex', '.bib', '.cfg', '.ini', '.toml', '.properties'
    }
    
    # Directories to skip
    SKIP_DIRECTORIES = {
        '__pycache__', '.git', '.svn', '.hg', 'node_modules',
        '.venv', 'venv', '.env', 'env', 'build', 'dist',
        '.pytest_cache', '.mypy_cache', '.tox', '.coverage'
    }
    
    # Files to skip
    SKIP_FILES = {
        '.DS_Store', 'Thumbs.db', '*.pyc', '*.pyo', '*.pyd',
        '.gitignore', '.gitattributes', 'LICENSE', 'README.*'
    }
    
    @classmethod
    def is_text_file(cls, file_path: Path) -> bool:
        """Check if file should be processed."""
        # Check extension
        if file_path.suffix.lower() in cls.TEXT_EXTENSIONS:
            return True
        
        # Check if it's a text file without extension
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first chunk and check if it's mostly text
                chunk = f.read(1024)
                # Simple heuristic: if it contains readable text characters
                text_chars = sum(1 for c in chunk if c.isprintable())
                return text_chars / len(chunk) > 0.7 if chunk else False
        except Exception:
            return False
    
    @classmethod
    def should_skip_path(cls, path: Path) -> bool:
        """Check if path should be skipped."""
        # Skip hidden directories
        if path.name.startswith('.'):
            return True
        
        # Skip specific directories
        if path.name in cls.SKIP_DIRECTORIES:
            return True
        
        # Skip files in skip list
        if path.name in cls.SKIP_FILES:
            return True
        
        return False
    
    @classmethod
    def discover_files(cls, source_dir: Path) -> List[Path]:
        """Discover all text files in directory."""
        files = []
        
        for root, dirs, filenames in os.walk(source_dir):
            root_path = Path(root)
            
            # Skip directories
            dirs[:] = [d for d in dirs if not cls.should_skip_path(root_path / d)]
            
            for filename in filenames:
                file_path = root_path / filename
                
                if cls.should_skip_path(file_path):
                    continue
                
                if cls.is_text_file(file_path):
                    files.append(file_path)
        
        logger.info(f"Discovered {len(files)} text files to process")
        return files
    
    @classmethod
    def extract_text(cls, file_path: Path) -> str:
        """Extract text from file."""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return ""

class TextChunker:
    """Handles intelligent text chunking."""
    
    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.token_counter = TokenCounter()
    
    def chunk_text(self, text: str, source_file: str) -> List[TextChunk]:
        """Split text into chunks."""
        if not text.strip():
            return []
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        current_line = 1
        chunk_id_counter = 0
        
        for line_num, paragraph in enumerate(paragraphs, 1):
            paragraph_with_lines = f"\n[Lines {line_num}]\n{paragraph}"
            
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + paragraph_with_lines if current_chunk else paragraph_with_lines
            token_count = self.token_counter.count_tokens(test_chunk)
            
            if token_count > self.max_tokens and current_chunk:
                # Save current chunk
                chunk = self._create_chunk(
                    current_chunk, source_file, chunk_id_counter,
                    current_line, line_num - 1
                )
                chunks.append(chunk)
                chunk_id_counter += 1
                
                # Start new chunk with overlap
                if self.overlap_tokens > 0:
                    # Take last part of current chunk for overlap
                    words = current_chunk.split()
                    if len(words) > self.overlap_tokens:
                        overlap_text = ' '.join(words[-self.overlap_tokens:])
                        current_chunk = overlap_text + "\n\n" + paragraph_with_lines
                        current_line = line_num
                    else:
                        current_chunk = paragraph_with_lines
                        current_line = line_num
                else:
                    current_chunk = paragraph_with_lines
                    current_line = line_num
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += paragraph_with_lines
                else:
                    current_chunk = paragraph_with_lines
                    current_line = line_num
        
        # Add final chunk
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, source_file, chunk_id_counter,
                current_line, len(paragraphs)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, content: str, source_file: str, chunk_id: int, 
                     start_line: int, end_line: int) -> TextChunk:
        """Create a TextChunk object."""
        word_count = len(content.split())
        token_count = self.token_counter.count_tokens(content)
        
        return TextChunk(
            content=content,
            source_file=source_file,
            chunk_id=f"{Path(source_file).stem}_{chunk_id:04d}",
            start_line=start_line,
            end_line=end_line,
            word_count=word_count,
            token_count=token_count
        )

class EmbeddingGenerator:
    """Handles text embedding generation."""
    
    # Pre-trained models for different use cases
    MODELS = {
        'all-mpnet-base-v2': {
            'description': 'Best general-purpose model',
            'max_seq_length': 384
        },
        'all-MiniLM-L6-v2': {
            'description': 'Fast and lightweight',
            'max_seq_length': 256
        },
        'multi-qa-MiniLM-L6-cos-v1': {
            'description': 'Optimized for question-answering',
            'max_seq_length': 256
        }
    }
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2', device: str = 'cpu'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set max sequence length if supported
            if hasattr(self.model, 'max_seq_length'):
                max_len = self.MODELS[self.model_name]['max_seq_length']
                self.model.max_seq_length = max_len
            
            logger.info(f"Embedding model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a simpler model
            try:
                logger.info("Falling back to all-MiniLM-L6-v2")
                self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.model_name = 'all-MiniLM-L6-v2'
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> np.ndarray:
        """Generate embeddings for text chunks."""
        if not chunks:
            return np.array([])
        
        # Extract text content
        texts = [chunk.content for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks")
        start_time = time.time()
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            # Assign embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]
            
            duration = time.time() - start_time
            logger.info(f"Generated {len(embeddings)} embeddings in {duration:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

class VectorIndexer:
    """Handles vector indexing with FAISS."""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = None
        self.chunk_metadata = []  # Store metadata for each vector
    
    def build_index(self, embeddings: np.ndarray, chunks: List[TextChunk]) -> faiss.Index:
        """Build FAISS index from embeddings."""
        if len(embeddings) == 0:
            logger.warning("No embeddings to index")
            return None
        
        logger.info(f"Building FAISS index for {len(embeddings)} vectors")
        
        # Use IVF index for better performance on large datasets
        nlist = min(int(np.sqrt(len(embeddings))), 100)  # Number of clusters
        
        # Create IVF index
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        
        # Train index
        logger.info("Training index...")
        self.index.train(embeddings)
        
        # Add vectors
        logger.info("Adding vectors to index...")
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        self.chunk_metadata = [asdict(chunk) for chunk in chunks]
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
        return self.index
    
    def save_index(self, index_path: Path, metadata_path: Path):
        """Save index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata as JSON
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Index saved to {index_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def load_index(self, index_path: Path, metadata_path: Path) -> bool:
        """Load index and metadata from disk."""
        try:
            # Load FAISS index
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
            else:
                logger.warning(f"Index file not found: {index_path}")
                return False
            
            # Load metadata
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
                return False
            
            self.dimension = self.index.d
            logger.info(f"Index loaded successfully with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar vectors."""
        if self.index is None or len(self.chunk_metadata) == 0:
            logger.warning("No index available for search")
            return []
        
        # Ensure query embedding has correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Return (index, score) pairs
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
        return results

class CorpusIndexer:
    """Main class for building and managing the search index."""
    
    def __init__(self, source_dir: str, output_dir: str, model_name: str = 'all-mpnet-base-v2',
                 device: str = 'cpu', max_tokens: int = 500, overlap_tokens: int = 50,
                 rebuild: bool = False):
        
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.device = device
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.rebuild = rebuild
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.file_processor = FileProcessor()
        self.text_chunker = TextChunker(max_tokens, overlap_tokens)
        self.embedding_generator = EmbeddingGenerator(model_name, device)
        self.vector_indexer = None
        
        # Index files
        self.index_file = self.output_dir / "corpus.index"
        self.metadata_file = self.output_dir / "corpus_metadata.json"
        self.config_file = self.output_dir / "index_config.json"
    
    def load_existing_index(self) -> bool:
        """Load existing index if available."""
        if not self.rebuild and self.index_file.exists() and self.metadata_file.exists():
            logger.info("Loading existing index...")
            
            # Load configuration
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Check if configuration matches
                if (config.get('model_name') == self.model_name and
                    config.get('max_tokens') == self.max_tokens and
                    config.get('overlap_tokens') == self.overlap_tokens):
                    
                    # Load index
                    self.vector_indexer = VectorIndexer(config['embedding_dimension'])
                    if self.vector_indexer.load_index(self.index_file, self.metadata_file):
                        return True
                else:
                    logger.info("Index configuration mismatch, rebuilding...")
            else:
                # Try to load with old format
                self.vector_indexer = VectorIndexer(768)  # Default dimension
                if self.vector_indexer.load_index(self.index_file, self.metadata_file):
                    return True
        
        return False
    
    def build_index(self) -> bool:
        """Build the search index."""
        logger.info("Starting corpus indexing...")
        start_time = time.time()
        
        try:
            # Discover and process files
            logger.info(f"Discovering files in {self.source_dir}")
            files = self.file_processor.discover_files(self.source_dir)
            
            if not files:
                logger.warning("No text files found to index")
                return False
            
            # Extract text from files
            logger.info("Extracting text from files...")
            file_contents = []
            for file_path in files:
                content = self.file_processor.extract_text(file_path)
                if content.strip():
                    file_contents.append((file_path, content))
            
            if not file_contents:
                logger.warning("No readable text content found")
                return False
            
            # Chunk text
            logger.info("Chunking text...")
            all_chunks = []
            for file_path, content in file_contents:
                chunks = self.text_chunker.chunk_text(content, str(file_path))
                all_chunks.extend(chunks)
            
            logger.info(f"Created {len(all_chunks)} text chunks")
            
            if not all_chunks:
                logger.warning("No text chunks created")
                return False
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(all_chunks)
            
            # Build vector index
            logger.info("Building vector index...")
            self.vector_indexer = VectorIndexer(embeddings.shape[1])
            self.vector_indexer.build_index(embeddings, all_chunks)
            
            # Save index
            self.vector_indexer.save_index(self.index_file, self.metadata_file)
            
            # Save configuration
            config = {
                'model_name': self.model_name,
                'embedding_dimension': embeddings.shape[1],
                'max_tokens': self.max_tokens,
                'overlap_tokens': self.overlap_tokens,
                'source_directory': str(self.source_dir),
                'total_files': len(files),
                'total_chunks': len(all_chunks),
                'total_embeddings': embeddings.shape[0],
                'created_at': time.time()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            duration = time.time() - start_time
            logger.info(f"Indexing completed in {duration:.2f}s")
            logger.info(f"Processed {len(files)} files into {len(all_chunks)} chunks")
            logger.info(f"Created {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search the index."""
        if self.vector_indexer is None:
            logger.error("No index available. Run build_index() first.")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.model.encode([query])
            
            # Search
            results = self.vector_indexer.search(query_embedding, k)
            
            # Format results
            formatted_results = []
            for idx, score in results:
                if idx < len(self.vector_indexer.chunk_metadata):
                    metadata = self.vector_indexer.chunk_metadata[idx]
                    
                    # Remove embedding from metadata to save space
                    result_metadata = {k: v for k, v in metadata.items() if k != 'embedding'}
                    result_metadata['similarity_score'] = score
                    
                    formatted_results.append(result_metadata)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get index statistics."""
        if self.vector_indexer is None:
            return {}
        
        stats = {
            'total_vectors': self.vector_indexer.index.ntotal if self.vector_indexer.index else 0,
            'embedding_dimension': self.vector_indexer.dimension,
            'total_chunks': len(self.vector_indexer.chunk_metadata)
        }
        
        # Load configuration for additional stats
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                stats.update(config)
            except Exception:
                pass
        
        return stats

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Build comprehensive search index for BrainMod Step 2 codebase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for current directory
  python build_index.py
  
  # Build index for specific directory
  python build_index.py --source-dir ./brainmod --output-dir ./my_index
  
  # Rebuild with custom settings
  python build_index.py --rebuild --max-tokens 300 --overlap 30
        """
    )
    
    parser.add_argument(
        '--source-dir', '-s',
        default='.',
        help='Directory containing text files to index (default: current directory)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='./search_index',
        help='Directory to store index files (default: ./search_index)'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=EmbeddingGenerator.MODELS.keys(),
        default='all-mpnet-base-v2',
        help='Embedding model to use (default: all-mpnet-base-v2)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to use for embeddings (default: cpu)'
    )
    
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=500,
        help='Maximum tokens per chunk (default: 500)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=50,
        help='Token overlap between chunks (default: 50)'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of existing index'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    parser.add_argument(
        '--search',
        help='Search query to test the index'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show index statistics'
    )
    
    args = parser.parse_args()
    
    # Set quiet mode
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Create indexer
    indexer = CorpusIndexer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        rebuild=args.rebuild
    )
    
    # Try to load existing index
    if not indexer.load_existing_index():
        # Build new index
        if not indexer.build_index():
            logger.error("Failed to build index")
            sys.exit(1)
    
    # Handle additional commands
    if args.stats:
        stats = indexer.get_statistics()
        print("\nIndex Statistics:")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print()
    
    if args.search:
        results = indexer.search(args.search, k=10)
        print(f"\nSearch Results for: '{args.search}'")
        print("=" * 70)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('chunk_id', 'Unknown')} (Score: {result.get('similarity_score', 0):.3f})")
            print(f"   File: {result.get('source_file', 'Unknown')}")
            print(f"   Lines: {result.get('start_line', 0)}-{result.get('end_line', 0)}")
            print(f"   Words: {result.get('word_count', 0)}")
            print(f"   Preview: {result.get('content', '')[:200]}...")
        
        if not results:
            print("No results found.")
    
    # Show model information
    if not args.quiet:
        model_info = EmbeddingGenerator.MODELS.get(args.model, {})
        print(f"\nUsing model: {args.model}")
        print(f"Description: {model_info.get('description', 'Unknown')}")
        print(f"Max sequence length: {model_info.get('max_seq_length', 'Unknown')}")

if __name__ == "__main__":
    main()