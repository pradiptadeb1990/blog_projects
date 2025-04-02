"""Tests for the transformer model."""

import torch

from zero2former.transformer import Transformer, TransformerDecoder, TransformerEncoder


def test_transformer_encoder():
    """Test the transformer encoder stack."""
    batch_size = 2
    seq_len = 10
    model_dim = 512
    num_heads = 8
    ffn_dim = 2048
    num_layers = 6

    # Create encoder
    encoder = TransformerEncoder(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim
    )
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, model_dim)
    
    # Test without mask
    output = encoder(x)
    assert output.shape == (batch_size, seq_len, model_dim)
    
    # Test with padding mask (1 for tokens to keep, 0 for tokens to mask)
    mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)
    mask[:, :, :, -2:] = False  # Mask out last two positions
    output = encoder(x, mask)
    assert output.shape == (batch_size, seq_len, model_dim)


def test_transformer_decoder():
    """Test the transformer decoder stack."""
    batch_size = 2
    seq_len = 10
    source_seq_len = 12
    model_dim = 512
    num_heads = 8
    ffn_dim = 2048
    num_layers = 6

    # Create decoder
    decoder = TransformerDecoder(
        num_layers=num_layers,
        model_dim=model_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim
    )
    
    # Create sample inputs
    x = torch.randn(batch_size, seq_len, model_dim)
    encoder_output = torch.randn(batch_size, source_seq_len, model_dim)
    
    # Test without masks
    output = decoder(x, encoder_output)
    assert output.shape == (batch_size, seq_len, model_dim)
    
    # Test with masks
    # Self-attention uses causal mask
    self_attn_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
    self_attn_mask = self_attn_mask.unsqueeze(0).unsqueeze(0)
    
    # Cross-attention uses padding mask
    cross_attn_mask = torch.ones(batch_size, 1, 1, source_seq_len, dtype=torch.bool)
    cross_attn_mask[:, :, :, -2:] = False  # Mask out last two positions
    
    output = decoder(
        x,
        encoder_output,
        self_attn_mask=self_attn_mask,
        cross_attn_mask=cross_attn_mask
    )
    assert output.shape == (batch_size, seq_len, model_dim)


def test_transformer():
    """Test the complete transformer model."""
    batch_size = 2
    source_seq_len = 10
    target_seq_len = 8
    source_vocab_size = 1000
    target_vocab_size = 1000
    model_dim = 512
    
    # Create transformer
    transformer = Transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        model_dim=model_dim
    )
    
    # Create sample inputs
    source = torch.randint(0, source_vocab_size, (batch_size, source_seq_len))
    target = torch.randint(0, target_vocab_size, (batch_size, target_seq_len))
    
    # Test forward pass
    output = transformer(source, target)
    assert output.shape == (batch_size, target_seq_len, target_vocab_size)
    
    # Test with padding masks (1 for tokens to keep, 0 for tokens to mask)
    source_padding_mask = torch.ones(batch_size, 1, 1, source_seq_len, dtype=torch.bool)
    source_padding_mask[:, :, :, -2:] = False  # Mask out last two positions
    
    target_padding_mask = torch.ones(batch_size, 1, 1, target_seq_len, dtype=torch.bool)
    target_padding_mask[:, :, :, -2:] = False  # Mask out last two positions
    
    memory_padding_mask = source_padding_mask  # Memory mask is same as source mask
    
    output = transformer(
        source,
        target,
        source_padding_mask=source_padding_mask,
        target_padding_mask=target_padding_mask,
        memory_padding_mask=memory_padding_mask
    )
    assert output.shape == (batch_size, target_seq_len, target_vocab_size)


def test_transformer_encode_decode():
    """Test separate encoding and decoding steps."""
    batch_size = 2
    source_seq_len = 10
    target_seq_len = 8
    source_vocab_size = 1000
    target_vocab_size = 1000
    model_dim = 512
    
    # Create transformer
    transformer = Transformer(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        model_dim=model_dim
    )
    
    # Create sample inputs
    source = torch.randint(0, source_vocab_size, (batch_size, source_seq_len))
    target = torch.randint(0, target_vocab_size, (batch_size, target_seq_len))
    
    # Test encoding
    memory = transformer.encode(source)
    assert memory.shape == (batch_size, source_seq_len, model_dim)
    
    # Test decoding
    output = transformer.decode(target, memory)
    assert output.shape == (batch_size, target_seq_len, target_vocab_size)


def test_transformer_masks():
    """Test mask creation in transformer."""
    batch_size = 2
    seq_len = 10
    pad_idx = 0
    
    # Create transformer
    transformer = Transformer(source_vocab_size=1000, target_vocab_size=1000)
    
    # Test padding mask
    seq = torch.randint(1, 1000, (batch_size, seq_len))
    seq[:, -2:] = pad_idx  # Add padding
    mask = transformer.create_padding_mask(seq, pad_idx)
    
    assert mask.shape == (batch_size, 1, 1, seq_len)
    assert torch.all(mask[:, :, :, :-2] == 1)  # Non-pad positions
    assert torch.all(mask[:, :, :, -2:] == 0)  # Pad positions
    
    # Test causal mask
    mask = transformer.create_causal_mask(seq_len)
    assert mask.shape == (1, 1, seq_len, seq_len)
    assert torch.all(torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == ~mask[0, 0])
