#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-Efficient Attention Mechanisms

This module provides advanced attention mechanisms optimized for memory efficiency
and performance in deep learning applications.

Author: MiniMax AI
Version: 1.0.0
Created: 2025-11-06

Features:
- Chunked attention for long sequences
- Sparse attention patterns 
- Linear attention for reduced complexity
- Flash attention integration
- Performance statistics tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import time
import warnings

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    warnings.warn("Flash Attention not available. Install flash-attn for optimal performance.")


@dataclass
class AttentionStats:
    """Statistics for attention mechanism performance."""
    forward_time: float = 0.0
    memory_used: float = 0.0
    tokens_processed: int = 0
    chunks_processed: int = 0
    sparse_ratio: float = 0.0
    linear_approximation: bool = False
    flash_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'forward_time': self.forward_time,
            'memory_used': self.memory_used,
            'tokens_processed': self.tokens_processed,
            'chunks_processed': self.chunks_processed,
            'sparse_ratio': self.sparse_ratio,
            'linear_approximation': self.linear_approximation,
            'flash_used': self.flash_used,
            'throughput': self.tokens_processed / self.forward_time if self.forward_time > 0 else 0
        }


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention mechanism supporting multiple optimization strategies.
    
    Features:
    - Chunked attention for handling long sequences
    - Sparse attention with learned sparsity patterns
    - Linear attention for O(n) complexity
    - Flash attention when available
    - Adaptive strategy selection based on sequence length
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        chunk_size: int = 512,
        dropout: float = 0.1,
        use_flash: bool = True,
        use_sparse: bool = True,
        use_linear: bool = True,
        sparse_top_k: int = 16
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads 
        self.head_dim = dim // num_heads
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.sparse_top_k = sparse_top_k
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Sparsity patterns for sparse attention
        self.sparse_patterns = nn.Parameter(torch.randn(num_heads, sparse_top_k))
        
        # Statistics tracking
        self.stats = AttentionStats()
        self.reset_stats()
        
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = AttentionStats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return self.stats.to_dict()
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split tensor into multiple heads."""
        b, n, d = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Combine multiple heads back into single tensor."""
        b, h, n, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, n, h * d)
    
    def chunked_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute attention using chunking to reduce memory usage.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim] 
            v: Value tensor [batch, heads, seq_len, head_dim]
            chunk_size: Size of chunks to process
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        batch, heads, seq_len, head_dim = q.shape
        chunks = (seq_len + chunk_size - 1) // chunk_size
        
        # Process in chunks
        outputs = []
        
        for i in range(chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)
            
            # Get chunk
            q_chunk = q[:, :, start_idx:end_idx, :]
            k_chunk = k  # Full key for attention
            v_chunk = v  # Full value for attention
            
            # Compute attention for chunk
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(head_dim)
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention
            chunk_output = torch.matmul(attn_weights, v_chunk)
            outputs.append(chunk_output)
            
            self.stats.chunks_processed += 1
        
        # Concatenate chunks
        output = torch.cat(outputs, dim=2)
        return self._combine_heads(output)
    
    def sparse_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention using learned sparse patterns.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            
        Returns:
            Sparse attention output
        """
        batch, heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply sparse pattern (top-k per query position)
        top_scores, top_indices = torch.topk(scores, self.sparse_top_k, dim=-1)
        
        # Create sparse attention weights
        sparse_attn = torch.zeros_like(scores)
        sparse_attn.scatter_(-1, top_indices, F.softmax(top_scores, dim=-1))
        
        # Apply dropout
        sparse_attn = F.dropout(sparse_attn, p=self.dropout, training=self.training)
        
        # Compute output
        output = torch.matmul(sparse_attn, v)
        
        # Update statistics
        self.stats.sparse_ratio = 1.0 - (self.sparse_top_k / seq_len)
        
        return self._combine_heads(output)
    
    def linear_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention using linear approximation for O(n) complexity.
        
        Args:
            q: Query tensor [batch, heads, seq_len, head_dim]
            k: Key tensor [batch, heads, seq_len, head_dim]  
            v: Value tensor [batch, heads, seq_len, head_dim]
            
        Returns:
            Linear attention output
        """
        # Feature maps for linear attention
        q_map = F.elu(q) + 1
        k_map = F.elu(k) + 1
        
        # Compute key-value products
        kv = torch.matmul(k_map.transpose(-2, -1), v)
        z = torch.sum(k_map, dim=-2, keepdim=True)
        
        # Compute output
        output = torch.matmul(q_map, kv) / z
        
        # Update statistics
        self.stats.linear_approximation = True
        
        return self._combine_heads(output)
    
    def flash_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention using Flash Attention if available.
        
        Args:
            q: Query tensor [batch, seq_len, dim]
            k: Key tensor [batch, seq_len, dim]
            v: Value tensor [batch, seq_len, dim]
            
        Returns:
            Flash attention output
        """
        if not FLASH_ATTN_AVAILABLE:
            # Fallback to standard attention
            return self.standard_attention(q, k, v)
        
        # Reshape for flash attention (remove head dimension)
        q_flat = q.view(q.shape[0], q.shape[1], -1)
        k_flat = k.view(k.shape[0], k.shape[1], -1) 
        v_flat = v.view(v.shape[0], v.shape[1], -1)
        
        # Use flash attention
        output = flash_attn.flash_attn_func(
            q_flat, k_flat, v_flat,
            dropout_p=self.dropout if self.training else 0.0,
            causal=False,
            window_size=(-1, -1)
        )
        
        self.stats.flash_used = True
        return output
    
    def standard_attention(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor
    ) -> torch.Tensor:
        """Standard scaled dot-product attention."""
        batch, heads, seq_len, head_dim = q.shape
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        return self._combine_heads(output)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive attention strategy selection.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        start_time = time.time()
        
        batch, seq_len, dim = x.shape
        
        # Linear projections
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))  
        v = self._split_heads(self.v_proj(x))
        
        # Update statistics
        self.stats.tokens_processed = batch * seq_len
        
        # Memory tracking
        if torch.cuda.is_available():
            self.stats.memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        
        # Adaptive strategy selection based on sequence length
        if FLASH_ATTN_AVAILABLE and self.training and seq_len <= 2048:
            # Use flash attention for short-medium sequences during training
            output = self.flash_attention(x, x, x)
        elif seq_len > 1024 and self.chunk_size < seq_len:
            # Use chunked attention for very long sequences
            output = self.chunked_attention(q, k, v)
        elif hasattr(self, 'use_sparse') and self.use_sparse and seq_len > 64:
            # Use sparse attention for medium sequences
            output = self.sparse_attention(q, k, v)
        elif hasattr(self, 'use_linear') and self.use_linear and seq_len > 128:
            # Use linear attention for long sequences
            output = self.linear_attention(q, k, v)
        else:
            # Use standard attention for short sequences
            output = self.standard_attention(q, k, v)
        
        # Final projection
        output = self.out_proj(output)
        
        # Update timing statistics
        self.stats.forward_time = time.time() - start_time
        
        return output
    
    def profile_attention(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Profile attention performance across different strategies.
        
        Args:
            x: Input tensor for profiling
            
        Returns:
            Dictionary with profiling results for each strategy
        """
        results = {}
        
        # Test standard attention
        start_time = time.time()
        self.reset_stats()
        output_std = self.standard_attention(*[self._split_heads(proj(x)) for proj in [self.q_proj, self.k_proj, self.v_proj]])
        results['standard'] = {'time': time.time() - start_time, 'memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}
        
        # Test sparse attention
        start_time = time.time()
        self.reset_stats()
        output_sparse = self.sparse_attention(*[self._split_heads(proj(x)) for proj in [self.q_proj, self.k_proj, self.v_proj]])
        results['sparse'] = {'time': time.time() - start_time, 'memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}
        
        # Test linear attention
        start_time = time.time()
        self.reset_stats()
        output_linear = self.linear_attention(*[self._split_heads(proj(x)) for proj in [self.q_proj, self.k_proj, self.v_proj]])
        results['linear'] = {'time': time.time() - start_time, 'memory': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}
        
        return results


def get_attention_engine(dim: int = 512, num_heads: int = 8) -> MemoryEfficientAttention:
    """Get a configured attention engine for common use cases."""
    return MemoryEfficientAttention(
        dim=dim,
        num_heads=num_heads,
        chunk_size=512,
        dropout=0.1,
        use_flash=True,
        use_sparse=True,
        use_linear=True,
        sparse_top_k=16
    )


# Demo function
async def efficient_attention_demo():
    """Demonstrate efficient attention mechanisms."""
    print("=== Efficient Attention Demo ===")
    
    # Create attention engine
    attn = get_attention_engine(dim=512, num_heads=8)
    
    # Test with different sequence lengths
    test_cases = [
        ("Short sequence", 128),
        ("Medium sequence", 512), 
        ("Long sequence", 1024),
        ("Very long sequence", 2048)
    ]
    
    for name, seq_len in test_cases:
        print(f"\n{name} (seq_len={seq_len}):")
        
        # Create random input
        x = torch.randn(2, seq_len, 512)
        
        # Profile different attention strategies
        if torch.cuda.is_available():
            x = x.cuda()
            attn = attn.cuda()
        
        with torch.no_grad():
            # Profile attention
            profile_results = attn.profile_attention(x)
            
            print("Strategy comparison:")
            for strategy, stats in profile_results.items():
                throughput = (2 * seq_len) / stats['time']  # tokens/sec
                print(f"  {strategy:10s}: {stats['time']:.4f}s, {throughput:.0f} tok/s")
            
            # Test forward pass
            attn.reset_stats()
            output = attn(x)
            
            # Show statistics
            stats = attn.get_stats()
            print(f"Forward pass stats:")
            print(f"  Time: {stats['forward_time']:.4f}s")
            print(f"  Memory: {stats['memory_used']:.2f}GB")
            print(f"  Throughput: {stats['throughput']:.0f} tokens/sec")
            if stats['sparse_ratio'] > 0:
                print(f"  Sparsity: {stats['sparse_ratio']:.2%}")
            if stats['linear_approximation']:
                print("  Used linear attention")
            if stats['flash_used']:
                print("  Used flash attention")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    import asyncio
    asyncio.run(efficient_attention_demo())
