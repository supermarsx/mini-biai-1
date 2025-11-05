"""
Pytest Configuration and Shared Fixtures
=========================================

Shared fixtures and configuration for the mini-biai-1 test suite.
This module provides common test infrastructure that can be used
across all test modules.
"""

import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, Any, List

# Import test utilities
from test_utils import (
    TestDataGenerator, PerformanceMonitor, DataValidator, 
    MockFactory, TestConfig, AsyncPerformanceMonitor
)
from mock_data_generators import ComprehensiveMockDataGenerator, MockDataConfig
from performance_benchmarks import BenchmarkConfig, SystemBenchmark

# Import test utilities
from mini_biai_1_test_utils import (
    Mini-Biai-1TestDataGenerator,
    Mini-Biai-1MockFactory,
    Mini-Biai-1PerformanceMonitor,
    Mini-Biai-1TestHelpers,
    Mini-Biai-1TestData,
    test_data_generator,
    mock_factory,
    performance_monitor,
    test_helpers,
    TestMetrics
)

# Global test configuration
pytest_plugins = []


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment for the entire test session"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Set test-specific environment variables
    import os
    os.environ['TESTING'] = 'true'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU in tests
    
    yield
    
    # Cleanup after all tests
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="mini-biai-1_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_config():
    """Provide sample configuration for testing"""
    return {
        'model': {
            'd_model': 512,
            'router': {
                'top_k': 1,
                'temperature': 0.8,
                'spike_threshold': 0.7,
                'target_spike_rate': 0.10
            },
            'language': {
                'backbone': 'linear_attn_stub',
                'max_new_tokens': 128
            }
        },
        'memory': {
            'stm': {
                'max_tokens': 4096,
                'buffer_size': 1024,
                'scratchpad_size': 128,
                'compression_ratio': 0.8
            },
            'ltm': {
                'backend': 'faiss',
                'dim': 512,
                'index': 'ivf_pq',
                'nlist': 4096,
                'nprobe': 8,
                'top_k': 5
            }
        },
        'serve': {
            'latency_budget_ms': 150
        },
        'logging': {
            'level': 'INFO'
        }
    }


@pytest.fixture
def deterministic_vectors():
    """Generate deterministic vectors for reproducible tests"""
    def _generate_vectors(count: int, dim: int = 512, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        vectors = np.random.normal(0, 1, (count, dim))
        return vectors
    
    return _generate_vectors


@pytest.fixture
def sample_texts():
    """Provide sample texts for testing"""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are computational models inspired by biological brains.",
        "FAISS enables efficient similarity search in high-dimensional space.",
        "Spiking neurons emit discrete signals called spikes.",
        "Short-term memory maintains recent conversational context.",
        "Long-term memory stores permanent knowledge and experiences.",
        "The coordinator routes between different processing modules.",
        "Vector embeddings capture semantic relationships between texts.",
        "Retrieval augmented generation combines search with generation.",
        "Leaky integrate and fire neurons have membrane potentials that decay over time.",
        "The brain uses both short-term and long-term memory systems.",
        "Artificial neural networks can be trained using backpropagation.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Transformers use self-attention to process sequences of data.",
        "Memory consolidation transfers information from short-term to long-term storage.",
        "Spiking neural networks closer approximate biological neuron behavior.",
        "Vector databases index embeddings for fast similarity search.",
        "The hippocampus plays a crucial role in memory formation.",
        "Neural plasticity allows the brain to adapt and learn new information."
    ]


@pytest.fixture
def mock_components():
    """Provide mock components for testing"""
    class MockComponents:
        def __init__(self):
            self.tokenizer = MagicMock()
            self.encoder = MagicMock()
            self.stm = MagicMock()
            self.ltm = MagicMock()
            self.coordinator = MagicMock()
            self.decoder = MagicMock()
            
            # Configure default behaviors
            self._configure_defaults()
        
        def _configure_defaults(self):
            """Configure default mock behaviors"""
            # Tokenizer
            self.tokenizer.encode.return_value = [101, 202, 303, 404, 505]
            self.tokenizer.decode.return_value = "decoded text"
            
            # Encoder
            def encode_side_effect(text):
                np.random.seed(hash(text) % 2**32)
                return np.random.normal(0, 1, 512)
            
            self.encoder.encode.side_effect = encode_side_effect
            
            # STM
            self.stm.get_context.return_value = {
                'tokens': list(range(100)),
                'token_count': 100,
                'scratchpad': {'test_key': 'test_value'},
                'kv_count': 1,
                'total_token_count': 100
            }
            self.stm.get_summary.return_value = {
                'total_tokens': 100,
                'buffer_utilization': 0.5,
                'scratchpad_size': 1,
                'turn_tokens': 5,
                'turn_kv': 1
            }
            self.stm.add_tokens = MagicMock()
            self.stm.add_kv = MagicMock()
            self.stm.clear_turn = MagicMock()
            self.stm.reset = MagicMock()
            
            # LTM
            self.ltm.query.return_value = [
                (0, 0.95, "Retrieved relevant context"),
                (1, 0.87, "Additional relevant information"),
                (2, 0.82, "Supporting context")
            ]
            self.ltm.add.return_value = [0, 1, 2]
            self.ltm.save = MagicMock()
            self.ltm.load = MagicMock()
            
            # Coordinator
            self.coordinator.forward.return_value = (
                torch.tensor([[0.85]]),  # routing weights
                0.12  # spike rate
            )
            
            # Decoder
            self.decoder.generate.return_value = "Generated response based on retrieved context."
    
    return MockComponents()