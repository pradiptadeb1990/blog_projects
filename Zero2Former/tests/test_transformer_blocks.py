"""Tests for transformer block implementations."""

import torch

from zero2former.transformer_blocks import DecoderBlock, EncoderBlock, FeedForwardNetwork


def test_feed_forward_network():
    """Test the feed-forward network."""
    batch_size = 2
    seq_len = 10
    model_dim = 512
    ffn_dim = 2048

    # Create feed-forward network
    ffn = FeedForwardNetwork(model_dim=model_dim, ffn_dim=ffn_dim)
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, model_dim)
    
    # Apply feed-forward network
    output = ffn(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, model_dim)
    
    # Test gradient flow
    loss = output.sum()
    loss.backward()
    
    # Check that gradients are computed
    assert ffn.linear1.weight.grad is not None
    assert ffn.linear2.weight.grad is not None


def test_encoder_block():
    """Test the encoder block."""
    batch_size = 2
    seq_len = 10
    model_dim = 512
    num_heads = 8
    ffn_dim = 2048

    # Create encoder block
    encoder = EncoderBlock(
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, model_dim)
    
    # Test without mask
    output = encoder(x)
    assert output.shape == (batch_size, seq_len, model_dim)
    
    # Test with attention mask (batch_size, 1, 1, seq_len)
    mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)

    output = encoder(x, mask)
    assert output.shape == (batch_size, seq_len, model_dim)


def test_decoder_block():
    """Test the decoder block."""
    batch_size = 2
    seq_len = 10
    src_seq_len = 12  # Different source sequence length
    model_dim = 512
    num_heads = 8
    ffn_dim = 2048

    # Create decoder block
    decoder = DecoderBlock(
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim
    )
    
    # Create sample inputs
    x = torch.randn(batch_size, seq_len, model_dim)
    encoder_output = torch.randn(batch_size, src_seq_len, model_dim)
    
    # Test without masks
    output = decoder(x, encoder_output)
    assert output.shape == (batch_size, seq_len, model_dim)
    
    # Test with attention masks
    # Self-attention uses causal mask
    self_attn_mask = torch.triu(torch.ones(1, 1, seq_len, seq_len), diagonal=1) == 0
    
    # Cross-attention uses padding mask
    cross_attn_mask = torch.ones(batch_size, 1, 1, src_seq_len, dtype=torch.bool)
    
    output = decoder(
        x,
        encoder_output,
        self_attn_mask=self_attn_mask,
        cross_attn_mask=cross_attn_mask
    )
    assert output.shape == (batch_size, seq_len, model_dim)


def test_block_gradient_flow():
    """Test gradient flow through encoder and decoder blocks."""
    batch_size = 2
    seq_len = 10
    src_seq_len = 12
    model_dim = 512
    num_heads = 8
    ffn_dim = 2048

    # Create blocks
    encoder = EncoderBlock(model_dim, num_heads, ffn_dim)
    decoder = DecoderBlock(model_dim, num_heads, ffn_dim)
    
    # Create inputs that require gradients
    x = torch.randn(batch_size, seq_len, model_dim, requires_grad=True)
    encoder_output = torch.randn(batch_size, src_seq_len, model_dim, requires_grad=True)
    
    # Forward pass
    enc_output = encoder(x)
    dec_output = decoder(x, encoder_output)
    
    # Backward pass
    loss = (enc_output.sum() + dec_output.sum()) / 2
    loss.backward()
    
    # Check that gradients are computed
    assert x.grad is not None
    assert encoder_output.grad is not None
