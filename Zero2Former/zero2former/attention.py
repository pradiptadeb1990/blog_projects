# Copyright (c) 2025 Pradipta Deb under MIT License (see LICENSE).
# Source for "Zero2Former: Building Transformers from Scratch"
# Source Code: https://github.com/pradiptadeb1990/blog_projects

"""
Attention mechanism implementation from scratch.

This module implements the attention mechanisms described in the 'Attention Is All You Need'
paper (https://arxiv.org/abs/1706.03762), providing a step-by-step educational implementation.

"""


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Implementation of scaled dot-product attention as described in 'Attention Is All You Need'."""

    def __init__(self, dropout: float = 0.0) -> None:
        """Initialize the attention layer.

        Args:
            dropout: Dropout probability. Defaults to 0.0.
        """
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (bs, num_heads, seq_len_q, d_k)
            key: Key tensor of shape (bs, num_heads, seq_len_k, d_k)
            value: Value tensor of shape (bs, num_heads, seq_len_v, d_v)
            mask: Optional mask tensor of shape (bs, num_heads, seq_len_q, seq_len_k)

        Returns:
            tuple containing:
                - Output tensor of shape (bs, num_heads, seq_len_q, d_v)
                - Attention weights of shape (bs, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))

        # Apply mask only if it's provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        scaled_attention_weights = F.softmax(attention_scores, dim=-1)

        # Apply dropout on attention weights
        scaled_attention_weights = self.dropout(scaled_attention_weights)

        # Matrix multiply attention weights with values
        scaled_output = torch.matmul(scaled_attention_weights, value)

        return scaled_output, scaled_attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(
        self, model_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True
    ) -> None:
        """Initialize the multi-head attention layer.

        Args:
            model_dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias in linear transformations
        """
        super(MultiHeadAttention, self).__init__()

        # Validating model dimension
        if model_dim % num_heads != 0:
            raise ValueError(f"Model dimension {model_dim} must be divisible by num_heads {num_heads}")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        self.scaled_attention = ScaledDotProductAttention(dropout)
        self.k_dim = model_dim // num_heads

        # Linear layers (projections) for Q, K, and V
        self.weight_q = nn.Linear(model_dim, model_dim, bias=bias)
        self.weight_k = nn.Linear(model_dim, model_dim, bias=bias)
        self.weight_v = nn.Linear(model_dim, model_dim, bias=bias)
        
        # Output projection
        self.weight_o = nn.Linear(model_dim, model_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply multi-head attention.

        Args:
            query: Query tensor of shape (bs, seq_len_q, model_dim)
            key: Key tensor of shape (bs, seq_len_k, model_dim)
            value: Value tensor of shape (bs, seq_len_v, model_dim)
            mask: Optional mask tensor

        Returns:
            tuple containing:
                - Output tensor of shape (bs, seq_len_q, model_dim)
                - Attention weights from the last head
        """
        # Get batch size
        bs = query.size(0)

        # Apply linear transformations and split into heads
        # Transform query using a linear layer, then reshape:
        # 1. Apply linear projection to query (bs, seq_len_q, model_dim)
        # 2. Reshape to (bs, seq_len_q, num_heads, k_dim)
        # 3. Transpose to (bs, num_heads, seq_len_q, k_dim) for attention
        q = self.weight_q(query).view(bs, -1, self.num_heads, self.k_dim).transpose(1, 2)
        k = self.weight_k(key).view(bs, -1, self.num_heads, self.k_dim).transpose(1, 2)
        v = self.weight_v(value).view(bs, -1, self.num_heads, self.k_dim).transpose(1, 2)

        # Apply mask only if it's provided
        # The mask should be of shape (batch_size, 1, 1, seq_len) or (batch_size, num_heads, seq_len_q, seq_len_k)
        # If it's the first shape, we broadcast it to all heads
        if mask is not None and mask.dim() == 4:
            if mask.size(1) == 1:
                mask = mask.repeat(1, self.num_heads, 1, 1)

        # Apply scaled dot-product attention
        scaled_attention_output, scaled_attention_weights = self.scaled_attention(q, k, v, mask)

        # Concatenate heads and apply output projection
        # 1. Transpose dimensions 1 and 2 to get (bs, seq_len, num_heads, k_dim)
        # 2. Make tensor contiguous in memory for efficient reshaping
        # 3. Combine the num_heads and k_dim dimensions to get original model_dim
        multi_head_output = (
            scaled_attention_output.transpose(1, 2)
            .contiguous()
            .view(bs, -1, self.num_heads * self.k_dim)
        )
        multi_head_output = self.weight_o(multi_head_output)

        # Apply dropout
        multi_head_output = self.dropout(multi_head_output)

        return multi_head_output, scaled_attention_weights
