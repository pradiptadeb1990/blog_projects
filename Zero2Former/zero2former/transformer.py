# Copyright (c) 2025 Pradipta Deb under MIT License (see LICENSE).
# Source for "Zero2Former: Building Transformers from Scratch"
# Source Code: https://github.com/pradiptadeb1990/blog_projects

"""
Implementation of the complete Transformer architecture.

This module implements the full Transformer model as described in the
'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762).
"""

from typing import Optional

import torch
import torch.nn as nn

from zero2former.positional_encoding import PositionalEncoding
from zero2former.transformer_blocks import DecoderBlock, EncoderBlock


class TransformerEncoder(nn.Module):
    """Stack of N encoder blocks."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the encoder stack.

        Args:
            num_layers: Number of encoder blocks
            model_dim: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension of feed-forward networks
            dropout: Dropout probability
        """
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input through the encoder stack.

        Args:
            x: Input tensor of shape (batch_size, seq_len, model_dim)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch_size, seq_len, model_dim)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """Stack of N decoder blocks."""

    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the decoder stack.

        Args:
            num_layers: Number of decoder blocks
            model_dim: Model dimension
            num_heads: Number of attention heads
            ffn_dim: Hidden dimension of feed-forward networks
            dropout: Dropout probability
        """
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            DecoderBlock(
                model_dim=model_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input through the decoder stack.

        Args:
            x: Input tensor of shape (batch_size, seq_len, model_dim)
            encoder_output: Output from encoder
            self_attn_mask: Mask for self-attention
            cross_attn_mask: Mask for cross-attention

        Returns:
            Output tensor of shape (batch_size, seq_len, model_dim)
        """
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)
        return x


class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks."""

    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 4096,
        device: torch.device = "cpu"
    ) -> None:
        """Initialize the Transformer.

        Args:
            source_vocab_size: Size of source vocabulary
            target_vocab_size: Size of target vocabulary
            model_dim: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            ffn_dim: Hidden dimension of feed-forward networks
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(Transformer, self).__init__()

        self.device = device
        # Embeddings
        self.source_embedding = nn.Embedding(source_vocab_size, model_dim)
        self.target_embedding = nn.Embedding(target_vocab_size, model_dim)
        
        # Scale embeddings by sqrt(model_dim)
        self.embedding_scale = torch.sqrt(torch.tensor(model_dim))
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            model_dim=model_dim,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        # Encoder and decoder stacks
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout
        )
        
        # Final linear layer
        self.output_linear = nn.Linear(model_dim, target_vocab_size)
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                # Initialize weights using Xavier uniform
                nn.init.xavier_uniform_(p)

    def create_padding_mask(
        self,
        seq: torch.Tensor,
        pad_idx: int = 0
    ) -> torch.Tensor:
        """Create mask for padding tokens.

        Args:
            seq: Input sequence tensor of shape (batch_size, seq_len)
            pad_idx: Index used for padding

        Returns:
            Mask tensor of shape (batch_size, 1, 1, seq_len)
        """
        # Ensure seq is on the correct device
        seq = seq.to(self.device)

        # Create mask for padding tokens (1 for non-pad, 0 for pad)
        mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask

    def create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal mask for decoder self-attention.

        Args:
            size: Size of the square mask

        Returns:
            Lower triangular mask of shape (1, 1, size, size)
        """
        # Create causal mask (lower triangular)
        mask = torch.triu(torch.ones(1, 1, size, size), diagonal=1) == 0
        return mask.to(self.device)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_padding_mask: Optional[torch.Tensor] = None,
        target_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process input through the transformer model.

        Args:
            source: Source sequence tensor of shape (batch_size, source_seq_len)
            target: Target sequence tensor of shape (batch_size, target_seq_len)
            source_padding_mask: Mask for source padding tokens
            target_padding_mask: Mask for target padding tokens
            memory_padding_mask: Mask for memory (encoder output) padding tokens

        Returns:
            Output tensor of shape (batch_size, target_seq_len, target_vocab_size)
        """
        # Create causal mask for decoder
        target_seq_len = target.size(1)
        target_mask = self.create_causal_mask(target_seq_len)
        if target_padding_mask is not None:
            # Combine causal mask with padding mask
            target_mask = target_mask & target_padding_mask
        
        # Embed and encode source sequence
        source = self.source_embedding(source) * self.embedding_scale
        source = self.positional_encoding(source)
        memory = self.encoder(source, source_padding_mask)
        
        # Embed and decode target sequence
        target = self.target_embedding(target) * self.embedding_scale
        target = self.positional_encoding(target)
        output = self.decoder(
            target,
            memory,
            self_attn_mask=target_mask,
            cross_attn_mask=memory_padding_mask
        )
        
        # Project to vocabulary size
        output = self.output_linear(output)
        
        return output

    def encode(
        self,
        source: torch.Tensor,
        source_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence.

        Args:
            source: Source sequence tensor
            source_padding_mask: Mask for padding tokens

        Returns:
            Encoder output tensor
        """
        source = self.source_embedding(source) * self.embedding_scale
        source = self.positional_encoding(source)
        return self.encoder(source, source_padding_mask)

    def decode(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence given encoder memory.

        Args:
            target: Target sequence tensor
            memory: Encoder output tensor
            target_padding_mask: Mask for target padding tokens
            memory_padding_mask: Mask for memory padding tokens

        Returns:
            Decoder output tensor
        """
        target_seq_len = target.size(1)
        target_mask = self.create_causal_mask(target_seq_len)
        
        if target_padding_mask is not None:
            target_mask = target_mask & target_padding_mask
        
        target = self.target_embedding(target) * self.embedding_scale
        target = self.positional_encoding(target)
        output = self.decoder(
            target,
            memory,
            self_attn_mask=target_mask,
            cross_attn_mask=memory_padding_mask
        )
        
        return self.output_linear(output)
