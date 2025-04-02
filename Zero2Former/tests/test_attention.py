"""Tests for attention mechanisms."""

import torch

from zero2former.attention import MultiHeadAttention, ScaledDotProductAttention


def test_scaled_dot_product_attention():
    """Test the scaled dot-product attention mechanism."""
    batch_size = 2
    num_heads = 4
    seq_len = 10
    d_k = 64

    attention = ScaledDotProductAttention(dropout=0.1)
    
    # Create sample inputs
    query = torch.randn(batch_size, num_heads, seq_len, d_k)
    key = torch.randn(batch_size, num_heads, seq_len, d_k)
    value = torch.randn(batch_size, num_heads, seq_len, d_k)
    
    # Test without mask
    output, weights = attention(query, key, value)
    
    assert output.shape == (batch_size, num_heads, seq_len, d_k)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)
    assert output.shape == query.shape
    
    # Test with mask
    mask = torch.ones(batch_size, num_heads, seq_len, seq_len)
    mask[:, :, :, -1] = 0  # Mask out the last position
    
    output, weights = attention(query, key, value, mask)
    assert torch.all(weights[:, :, :, -1] == 0)  # Masked positions should have zero attention


def test_multi_head_attention():
    """Test the multi-head attention mechanism."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(model_dim=d_model, num_heads=num_heads, dropout=0.1)
    
    # Create sample inputs
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    # Test without mask
    output, weights = mha(query, key, value)
    
    assert output.shape == (batch_size, seq_len, d_model)
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Test with mask (batch_size, 1, 1, seq_len)
    mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[:, :, :, -1] = False  # Mask out the last position
    
    output, weights = mha(query, key, value, mask)
    
    # Verify masking worked
    assert torch.all(weights[:, :, :, -1] == 0)  # Last position should have zero attention
    assert output.shape == (batch_size, seq_len, d_model)
    assert torch.all(weights[:, :, :, -1] == 0)  # Masked positions should have zero attention


def test_multi_head_attention_gradients():
    """Test that gradients flow properly through the multi-head attention."""
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8

    mha = MultiHeadAttention(model_dim=d_model, num_heads=num_heads)
    
    # Create sample inputs that require gradients
    query = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    key = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    value = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    # Forward pass
    output, _ = mha(query, key, value)
    
    # Compute loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert all(p.grad is not None for p in mha.parameters())
