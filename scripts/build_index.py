#!/usr/bin/env python3
"""
Corpus Indexing Script
Processes text files and builds vector index for efficient retrieval
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import hashlib
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorpusIndexer:
    """Handles corpus indexing and vector storage"""

    def __init__(self, corpus_path: str, output_path: str, config: Dict[str, Any]):
        self.corpus_path = Path(corpus_path)
        self.output_path = Path(output_path)
        self.config = config
        self.chunk_size = config.get('chunk_size', 512)
        self.chunk_overlap = config.get('chunk_overlap', 50)
        self.documents = []
        self.index = {}
        
    def process_corpus(self) -> bool:
        """Process all text files in corpus directory"""
        try:
            logger.info(f"Processing corpus: {self.corpus_path}")
            
            # Find all text files
            text_files = list(self.corpus_path.glob('*.txt'))
            if not text_files:
                logger.error(f"No .txt files found in {self.corpus_path}")
                return False
            
            logger.info(f"Found {len(text_files)} text files")
            
            # Process each file
            for file_path in text_files:
                if not self._process_file(file_path):
                    logger.error(f"Failed to process file: {file_path}")
                    return False
            
            logger.info(f"Processed {len(self.documents)} document chunks")
            return True
            
        except Exception as e:
            logger.error(f"Corpus processing failed: {e}")
            return False

    def _process_file(self, file_path: Path) -> bool:
        """Process individual file"""
        try:
            logger.debug(f"Processing file: {file_path.name}")
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                logger.warning(f"Empty file: {file_path.name}")
                return True
            
            # Split into chunks
            chunks = self._split_into_chunks(content)
            
            # Add to documents
            for i, chunk in enumerate(chunks):
                doc_id = self._generate_doc_id(file_path, i)
                self.documents.append({
                    'id': doc_id,
                    'file': str(file_path.relative_to(self.corpus_path)),
                    'chunk_id': i,
                    'content': chunk,
                    'hash': hashlib.md5(chunk.encode()).hexdigest(),
                    'file_hash': hashlib.md5(content.encode()).hexdigest(),
                    'created_at': datetime.now().isoformat()
                })
            
            logger.debug(f"Split {file_path.name} into {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"File processing error for {file_path}: {e}")
            return False

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Find chunk end
            end = start + self.chunk_size
            
            # Adjust end to not split words
            if end < text_length:
                # Try to find a good break point (space or punctuation)
                while end > start and end < text_length:
                    if text[end] in ' \n\t.,;:!?-':
                        end += 1
                        break
                    end -= 1
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks

    def _generate_doc_id(self, file_path: Path, chunk_id: int) -> str:
        """Generate unique document ID"""
        file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
        return f"{file_path.stem}_{chunk_id:04d}_{file_hash}"

    def build_index(self) -> bool:
        """Build searchable index from documents"""
        try:
            logger.info("Building search index...")
            
            # Create simple keyword index
            # In a real implementation, this would use vector embeddings
            self.index = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'corpus_path': str(self.corpus_path),
                    'total_documents': len(self.documents),
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'config': self.config
                },
                'documents': self.documents,
                'keyword_index': self._build_keyword_index(),
                'file_index': self._build_file_index()
            }
            
            logger.info(f"Index built with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return False

    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """Build keyword index for simple text search"""
        keyword_index = {}
        
        for doc in self.documents:
            # Extract keywords (simple approach - could be enhanced)
            words = set(doc['content'].lower().split())
            
            # Filter out common words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = {word for word in words if len(word) > 2 and word not in stop_words}
            
            # Add to index
            for keyword in keywords:
                if keyword not in keyword_index:
                    keyword_index[keyword] = []
                keyword_index[keyword].append(doc['id'])
        
        logger.info(f"Built keyword index with {len(keyword_index)} keywords")
        return keyword_index

    def _build_file_index(self) -> Dict[str, List[str]]:
        """Build index by file for quick file lookups"""
        file_index = {}
        
        for doc in self.documents:
            file_name = doc['file']
            if file_name not in file_index:
                file_index[file_name] = []
            file_index[file_name].append(doc['id'])
        
        return file_index

    def save_index(self) -> bool:
        """Save index to output directory"""
        try:
            logger.info(f"Saving index to {self.output_path}")
            
            # Ensure output directory exists
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main index file
            index_file = self.output_path / 'index.json'
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, ensure_ascii=False)
            
            # Save documents separately for easy access
            docs_file = self.output_path / 'documents.json'
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            metadata_file = self.output_path / 'metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.index['metadata'], f, indent=2)
            
            # Create summary report
            self._create_summary_report()
            
            logger.info("Index saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Index saving failed: {e}")
            return False

    def _create_summary_report(self):
        """Create summary report of indexing process"""
        try:
            report_file = self.output_path / 'indexing_report.json'
            
            # Analyze documents
            file_stats = {}
            for doc in self.documents:
                file_name = doc['file']
                if file_name not in file_stats:
                    file_stats[file_name] = {
                        'chunks': 0,
                        'total_chars': 0,
                        'total_words': 0
                    }
                
                file_stats[file_name]['chunks'] += 1
                file_stats[file_name]['total_chars'] += len(doc['content'])
                file_stats[file_name]['total_words'] += len(doc['content'].split())
            
            # Create report
            report = {
                'summary': {
                    'total_files': len(file_stats),
                    'total_chunks': len(self.documents),
                    'total_characters': sum(stats['total_chars'] for stats in file_stats.values()),
                    'total_words': sum(stats['total_words'] for stats in file_stats.values()),
                    'avg_chunks_per_file': len(self.documents) / len(file_stats) if file_stats else 0
                },
                'files': file_stats,
                'configuration': self.config,
                'created_at': datetime.now().isoformat()
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Could not create summary report: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Build corpus index')
    parser.add_argument('--corpus', required=True, help='Path to corpus directory')
    parser.add_argument('--output', required=True, help='Output directory for index')
    parser.add_argument('--chunk-size', type=int, default=512, help='Chunk size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, default=50, help='Overlap between chunks')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--silent', action='store_true', help='Suppress output except errors')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.silent:
        logging.basicConfig(level=logging.ERROR)
    elif args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Create configuration
    config = {
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'corpus_path': args.corpus,
        'output_path': args.output
    }
    
    # Initialize indexer
    indexer = CorpusIndexer(args.corpus, args.output, config)
    
    # Process corpus
    if not indexer.process_corpus():
        logger.error("Corpus processing failed")
        sys.exit(1)
    
    # Build index
    if not indexer.build_index():
        logger.error("Index building failed")
        sys.exit(1)
    
    # Save index
    if not indexer.save_index():
        logger.error("Index saving failed")
        sys.exit(1)
    
    logger.info("Indexing completed successfully")


if __name__ == '__main__':
    main()