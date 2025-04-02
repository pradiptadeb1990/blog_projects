# Copyright (c) 2025 Pradipta Deb under MIT License (see LICENSE).
# Source for "Zero2Former: Building Transformers from Scratch"
# Source Code: https://github.com/pradiptadeb1990/blog_projects

"""
Implementation of Transformer blocks from scratch.

This module implements the core transformer blocks as described in the
'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762), including
the feed-forward network, encoder block, and decoder block.
"""

import torch
import torch.nn as nn

from zero2former.attention import MultiHeadAttention


class FeedForwardNetwork(nn.Module):
    """Feed-forward network with ReLU activation.
    
    FFN(x) = act(xW₁ + b₁)W₂ + b₂
    where W₁ is the first linear transformation, W₂ is the second linear transformation,
    and b₁, b₂ are the bias terms.
    """

    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        activation: nn.Module = nn.ReLU(),
    ) -> None:
        """Initialize the feed-forward network.

        Args:
            model_dim: Input and output dimension
            ffn_dim: Hidden dimension of the feed-forward network
            activation: Activation function to use (default: ReLU)
        """
        super(FeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(model_dim, ffn_dim)
        self.activation = activation
        self.linear2 = nn.Linear(ffn_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to the input.

        Args:
            x: Input tensor of shape (batch_size, seq_len, model_dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, model_dim)
        """
        # First linear transformation with activation
        x = self.linear1(x)
        x = self.activation(x)
        
        # Second linear transformation
        x = self.linear2(x)
        
        return x


class EncoderBlock(nn.Module):
    """Transformer encoder block.
    
    Consists of:
    1. Multi-head self-attention
    2. Add & Norm
    3. Feed-forward network
    4. Add & Norm
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder block.

        Args:
            model_dim: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super(EncoderBlock, self).__init__()

        # Multi-head attention
        self.self_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            model_dim=model_dim,
            ffn_dim=ffn_dim
        )
        
        # Layer normalizations
        self.attention_norm = nn.LayerNorm(model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        input: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process input through the encoder block.

        Args:
            input: Input tensor of shape (batch_size, seq_len, model_dim)
            mask: Optional mask tensor for self-attention

        Returns:
            Output tensor of shape (batch_size, seq_len, model_dim)
        """
        # Self attention with residual connection and layer norm
        attn_output, _ = self.self_attention(input, input, input, mask)
        input = self.attention_norm(input + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.feed_forward(input)
        output = self.ffn_norm(input + self.dropout(ffn_output))
        
        return output


class DecoderBlock(nn.Module):
    """Transformer decoder block.
    
    Consists of:
    1. Masked multi-head self-attention
    2. Add & Norm
    3. Multi-head cross-attention with encoder output
    4. Add & Norm
    5. Feed-forward network
    6. Add & Norm
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the decoder block.

        Args:
            model_dim: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension of the feed-forward network
            dropout: Dropout probability
        """
        super(DecoderBlock, self).__init__()

        # Self attention
        self.self_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Cross attention
        self.cross_attention = MultiHeadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            model_dim=model_dim,
            ffn_dim=ffn_dim
        )
        
        # Layer normalizations
        self.attention_norm = nn.LayerNorm(model_dim)
        self.cross_attention_norm = nn.LayerNorm(model_dim)
        self.ffn_norm = nn.LayerNorm(model_dim)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        input: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Process input through the decoder block.

        Args:
            input: Input tensor of shape (batch_size, seq_len, model_dim)
            encoder_output: Output from encoder of shape (batch_size, src_seq_len, model_dim)
            self_attn_mask: Mask for self-attention (usually causal mask)
            cross_attn_mask: Mask for cross-attention with encoder output

        Returns:
            Output tensor of shape (batch_size, seq_len, model_dim)
        """
        # Self attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(input, input, input, self_attn_mask)
        input = self.attention_norm(input + self.dropout(self_attn_output))
        
        # Cross attention with residual connection and layer norm
        cross_attn_output, _ = self.cross_attention(
            input, encoder_output, encoder_output, cross_attn_mask
        )
        
        # Add & Norm
        input = self.cross_attention_norm(input + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ffn_output = self.feed_forward(input)

        # Add & Norm
        output = self.ffn_norm(input + self.dropout(ffn_output))
        
        return output
