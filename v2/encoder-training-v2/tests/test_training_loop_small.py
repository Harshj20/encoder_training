"""Test training loop with small dataset."""

import pytest
import torch
from pathlib import Path
from src.config import Config
from src.model_wrapper import EncoderClassifier
from src.data import create_dataloader


def test_training_loop_single_step():
    """Test that we can run one training step without OOM."""
    # Create minimal config
    config_dict = {
        "model": {
            "name_or_path": "prajjwal1/bert-tiny",
            "pooling": {"mode": "mean"},
            "num_labels": 2,
        },
        "training": {
            "mode": "head_only",
            "epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            "max_seq_length": 64,
            "lr": 1e-3,
            "output_dir": "./test_outputs",
            "max_steps": 2,  # Just 2 steps for smoke test
        },
        "data": {
            "type": "single",
            "dataset": "examples/sample_dataset.csv",
            "streaming": False,
            "text_field": "text",
            "label_field": "label",
            "max_samples": 4,
        },
        "loss": {
            "type": "classification",
        },
    }
    
    config = Config.from_dict(config_dict)
    
    # Create model
    model = EncoderClassifier(
        model_name_or_path=config.model.name_or_path,
        pooling_config=config.model.pooling.__dict__,
        num_labels=config.model.num_labels,
    )
    
    model.freeze_base()  # head_only mode
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset_path=config.data.dataset,
        tokenizer=model.tokenizer,
        data_config=config.data.__dict__,
        training_config=config.training.__dict__,
        split="train",
    )
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    model.train()
    
    # Run one training step
    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            mode="classification"
        )
        
        loss = outputs["loss"]
        assert loss is not None
        assert not torch.isnan(loss)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        break  # Just one batch
    
    # Cleanup
    import shutil
    if Path("./test_outputs").exists():
        shutil.rmtree("./test_outputs")
    
    # If we got here, training works!
    assert True


def test_head_only_reduces_trainable_params():
    """Test that head-only mode significantly reduces trainable parameters."""
    model = EncoderClassifier(
        model_name_or_path="prajjwal1/bert-tiny",
        num_labels=2,
    )
    
    # Count trainable params before freezing
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Freeze base
    model.freeze_base()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Should be much fewer trainable params
    assert trainable_after < trainable_before * 0.1  # Less than 10% trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
