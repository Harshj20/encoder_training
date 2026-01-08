"""Test model wrapper functionality."""

import pytest
import torch
from src.model_wrapper import EncoderClassifier, Pooling


def test_encoder_classifier_init():
    """Test EncoderClassifier initialization."""
    model = EncoderClassifier(
        model_name_or_path="prajjwal1/bert-tiny",  # Small model for testing
        pooling_config={"mode": "mean", "layers": [-1], "normalize": True},
        num_labels=2,
    )
    
    assert model.num_labels == 2
    assert model.tokenizer is not None
    assert model.encoder is not None


def test_pooling_modes():
    """Test different pooling modes."""
    batch_size, seq_len, hidden_size = 2, 10, 128
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Test mean pooling
    pooling_mean = Pooling(mode="mean", normalize=False)
    output_mean = pooling_mean(hidden_states, attention_mask)
    assert output_mean.shape == (batch_size, hidden_size)
    
    # Test cls pooling
    pooling_cls = Pooling(mode="cls", normalize=False)
    output_cls = pooling_cls(hidden_states, attention_mask)
    assert output_cls.shape == (batch_size, hidden_size)
    
    # Test max pooling
    pooling_max = Pooling(mode="max", normalize=False)
    output_max = pooling_max(hidden_states, attention_mask)
    assert output_max.shape == (batch_size, hidden_size)


def test_encode_returns_correct_shape():
    """Test that encode returns correct embedding shape."""
    model = EncoderClassifier(
        model_name_or_path="prajjwal1/bert-tiny",
        pooling_config={"mode": "mean"},
        num_labels=2,
    )
    
    # Create fake inputs
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)
    
    embeddings = model.encode(input_ids, attention_mask)
    
    assert embeddings.shape[0] == batch_size
    assert embeddings.shape[1] == model.hidden_size


def test_forward_classification_mode():
    """Test forward pass in classification mode."""
    model = EncoderClassifier(
        model_name_or_path="prajjwal1/bert-tiny",
        num_labels=2,
    )
    
    batch_size = 2
    input_ids = torch.randint(0, 1000, (batch_size, 10))
    attention_mask = torch.ones(batch_size, 10)
    labels = torch.tensor([0, 1])
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        mode="classification"
    )
    
    assert "logits" in outputs
    assert "embeddings" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, 2)


def test_freeze_unfreeze():
    """Test freeze_base and unfreeze_all."""
    model = EncoderClassifier(
        model_name_or_path="prajjwal1/bert-tiny",
        num_labels=2,
    )
    
    # Initially all should be trainable
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_before > 0
    
    # Freeze base
    model.freeze_base()
    frozen_count = sum(p.numel() for p in model.encoder.parameters() if not p.requires_grad)
    assert frozen_count > 0
    
    # Unfreeze all
    model.unfreeze_all()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_after == trainable_before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
