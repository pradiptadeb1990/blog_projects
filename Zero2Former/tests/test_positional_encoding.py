"""Tests for positional encoding implementations."""

import torch

from zero2former.positional_encoding import LearnedPositionalEncoding, PositionalEncoding


def test_sinusoidal_positional_encoding():
    """Test the fixed sinusoidal positional encoding."""
    batch_size = 2
    seq_len = 10
    model_dim = 512

    # Create positional encoding layer
    pe = PositionalEncoding(model_dim=model_dim, max_seq_length=100)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, model_dim)
    
    # Apply positional encoding
    output = pe(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, model_dim)


def test_learned_positional_encoding():
    """Test the learned positional encoding."""
    batch_size = 2
    seq_len = 10
    model_dim = 512

    # Create learned positional encoding layer
    pe = LearnedPositionalEncoding(model_dim=model_dim, max_seq_length=100)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, model_dim)
    
    # Apply positional encoding
    output = pe(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, model_dim)
    
    # Check that the positional embeddings are learnable parameters
    assert isinstance(pe.pos_embedding, torch.nn.Parameter)
    assert pe.pos_embedding.requires_grad
    
    # Check gradient flow
    loss = output.sum()
    loss.backward()
    
    assert pe.pos_embedding.grad is not None


def test_positional_encoding_different_sequence_lengths():
    """Test both encodings with different sequence lengths."""
    model_dim = 512
    max_seq_len = 100
    
    encodings = [
        PositionalEncoding(model_dim, max_seq_len),
        LearnedPositionalEncoding(model_dim, max_seq_len)
    ]
    
    for pe in encodings:
        # Test with different sequence lengths
        for seq_len in [1, 10, 50, max_seq_len]:
            x = torch.randn(2, seq_len, model_dim)
            output = pe(x)
            assert output.shape == (2, seq_len, model_dim)


def test_positional_encoding_dropout():
    """Test dropout in positional encodings."""
    batch_size = 2
    seq_len = 10
    model_dim = 512
    dropout = 0.5
    
    encodings = [
        PositionalEncoding(model_dim, dropout=dropout),
        LearnedPositionalEncoding(model_dim, dropout=dropout)
    ]
    
    for pe in encodings:
        x = torch.randn(batch_size, seq_len, model_dim)

        # Test in eval mode
        pe.eval()
        eval_output = pe(x)
        
        # In eval mode, output should be deterministic
        assert torch.allclose(eval_output, pe(x))
