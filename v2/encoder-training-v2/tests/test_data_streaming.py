"""Test data streaming functionality."""

import pytest
from src.data import load_streaming_dataset, create_dataloader, DataCollator
from transformers import AutoTokenizer


def test_load_csv_dataset():
    """Test loading CSV dataset."""
    dataset = load_streaming_dataset(
        dataset_path="examples/sample_dataset.csv",
        split="train",
        streaming=False,
        max_samples=5,
    )
    
    # Check we can iterate
    items = list(dataset)
    assert len(items) <= 5
    assert "text" in items[0] or "label" in items[0]


def test_data_collator_single():
    """Test DataCollator for single-text mode."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collator = DataCollator(tokenizer=tokenizer, data_type="single")
    
    # Create fake tokenized features
    features = [
        {"input_ids": [101, 2003, 102], "labels": 1},
        {"input_ids": [101, 2023, 2003, 102], "labels": 0},
    ]
    
    batch = collator(features)
    
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 2


def test_data_collator_triplet():
    """Test DataCollator for triplet mode."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    collator = DataCollator(tokenizer=tokenizer, data_type="triplet")
    
    features = [
        {
            "anchor_input_ids": [101, 2003, 102],
            "positive_input_ids": [101, 2023, 102],
            "negative_input_ids": [101, 2019, 102],
        },
    ]
    
    batch = collator(features)
    
    assert "anchor_input_ids" in batch
    assert "positive_input_ids" in batch
    assert "negative_input_ids" in batch


def test_streaming_yields_finite_batches():
    """Test that streaming dataset yields finite batches."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Create small dataloader
    data_config = {
        "type": "single",
        "streaming": False,
        "text_field": "text",
        "label_field": "label",
        "max_samples": 10,
    }
    
    training_config = {
        "per_device_train_batch_size": 2,
        "max_seq_length": 128,
    }
    
    dataloader = create_dataloader(
        dataset_path="examples/sample_dataset.csv",
        tokenizer=tokenizer,
        data_config=data_config,
        training_config=training_config,
        split="train",
    )
    
    # Should be able to iterate
    batches = list(dataloader)
    assert len(batches) > 0
    assert len(batches) <= 10  # max_samples / batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
