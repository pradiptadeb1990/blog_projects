# Copyright (c) 2025 Pradipta Deb under MIT License (see LICENSE).
# Source for "Zero2Former: Building Transformers from Scratch"
# Source Code: https://github.com/pradiptadeb1990/blog_projects

"""
Positional encoding implementation from scratch.

This module implements the sinusoidal positional encoding as described in the
'Attention Is All You Need' paper (https://arxiv.org/abs/1706.03762). The positional
encodings add information about the position of tokens in the sequence.
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding layer.
    
    This implements the fixed sinusoidal positional encoding where the encoding for
    each position is:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    where pos is the position and i is the dimension.
    """

    def __init__(
        self, 
        model_dim: int, 
        max_seq_length: int = 4096,
        dropout: float = 0.1
    ) -> None:
        """Initialize the positional encoding layer.

        Args:
            model_dim: The dimensionality of the model/embeddings
            max_seq_length: Maximum sequence length to pre-compute
            dropout: Dropout probability to apply to the encodings
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Create position vector [0, 1, 2, ..., max_seq_length-1]
        # Shape: (max_seq_length, 1)
        position = torch.arange(max_seq_length).unsqueeze(1)
        
        # Calculate division terms for the sinusoidal formula
        # This creates the frequency bands across dimensions
        # 1/(10000^(2i/d_model)) = 10000^(-2i/d_model) = exp(log(10000) * -2i/d_model) = exp(-log(10000) * 2i/d_model)
        denominator_term = torch.exp(
            torch.arange(0, model_dim, 2) * (-torch.log(torch.tensor(10000.0)) / model_dim)
        )
        
        # Create positional encoding matrix
        # Shape: (max_seq_length, model_dim)
        positional_encodings = torch.zeros(max_seq_length, model_dim)

        # Compute the positional encoding for each position and dimension, sin for even dimensions and cos for odd dimensions
        positional_encodings[:, 0::2] = torch.sin(position * denominator_term)
        positional_encodings[:, 1::2] = torch.cos(position * denominator_term)
        
        # convert 'positional_encodings' as a register buffer to make sure that the positional encoding won't be updated during backpropagation
        self.register_buffer('positional_encodings', positional_encodings)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.

        Args:
            input_data: Input tensor of shape (batch_size, seq_len, model_dim)

        Returns:
            Tensor of same shape as input with positional encodings added
        """
        seq_len = input_data.size(1)
        pos_encoding = self.positional_encodings[:seq_len, :]
        
        # Add positional encoding to input
        # pos_encoding has shape (seq_len, model_dim)
        # input_data has shape (batch_size, seq_len, model_dim)
        # we will handle the batch dimension using broadcasting
        
        input_data = input_data + pos_encoding
        
        # Apply dropout and return
        return self.dropout(input_data)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding layer.
    
    Instead of using fixed sinusoidal encodings, this layer learns the position
    embeddings during training.
    """

    def __init__(
        self,
        model_dim: int,
        max_seq_length: int = 5000,
        dropout: float = 0.1
    ) -> None:
        """Initialize the learned positional encoding layer.

        Args:
            model_dim: The dimensionality of the model/embeddings
            max_seq_length: Maximum sequence length to pre-compute
            dropout: Dropout probability to apply to the encodings
        """
        super(LearnedPositionalEncoding, self).__init__()
        
        # Initialize the position embeddings as a learnable parameter
        # shape: (1, max_seq_length, model_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, model_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Add learned positional encoding to the input tensor.

        Args:
            input_data: Input tensor of shape (batch_size, seq_len, model_dim)

        Returns:
            Tensor of same shape as input with positional encodings added
        """
        seq_len = input_data.size(1)

        # Get positional encoding for the sequence length
        # pos_embedding has shape (1, max_seq_length, model_dim)
        # pos_encoding has shape (1, seq_len, model_dim)
        # we will handle the batch dimension using broadcastingS
        pos_encoding = self.pos_embedding[:, :seq_len, :]
        
        # Add positional encoding to input
        input_data = input_data + pos_encoding
        
        # Apply dropout and return
        return self.dropout(input_data)
