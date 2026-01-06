# Example Training Script for Classification
# This script demonstrates programmatic usage of the training module

from encoder_training.core.config import FullConfig
from encoder_training.models import get_model
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Create or load configuration
config_dict = {
    "model": {
        "name": "kalm-embedding-2.5",
        "pretrained_path": "Kalm/kalm-embedding-2.5-instruct",
        "max_seq_length": 512,
    },
    "training": {
        "output_dir": "./output/example-classification",
        "batch_size": 16,
        "learning_rate": 2e-5,
        "num_epochs": 3,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "fp16": True,
    },
    "data": {
        "train_file": "./examples/datasets/classification_train.csv",
        "val_file": "./examples/datasets/classification_val.csv",
        "text_column": "text",
        "label_column": "label",
        "data_format": "csv",
    },
    "evaluation": {
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 50,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
    },
    "seed": 42,
    "device": "cuda",
}

# Create config object
config = FullConfig.from_dict(config_dict)

# Get model trainer
trainer = get_model(
    model_name=config.model.name,
    pretrained_path=config.model.pretrained_path
)

# Train
print("Starting training...")
results = trainer.train_classification(config)

print("\nTraining completed!")
print(f"Best metric: {results['best_metric']}")
print(f"Training history: {results['history']}")
