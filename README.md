# Encoder Training Module

A model-based training framework for encoder models supporting multiple NLP tasks. Designed with extensibility in mind to easily add new models and tasks.

## Features

- **Multiple Tasks**: Text Classification, Text Embedding (Contrastive Learning), Named Entity Recognition
- **Pre-configured Models**:
  - KALM Embedding 2.5 Instruct
  - Qwen 0.6B Embedding Instruct
  - Multilingual 300M
- **Extensible Architecture**: Easy to add new models via the registry system
- **Comprehensive Metrics**: Task-specific evaluation metrics
- **Flexible Configuration**: YAML-based configuration files
- **CLI Interface**: Simple command-line tools for training, inference, and evaluation

## Installation

```bash
# Clone the repository
cd encoder-training

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Prepare Your Data

**Classification** (`data/train.csv`):
```csv
text,label
"This is a great product",positive
"Terrible experience",negative
```

**Embedding** (`data/train_pairs.csv`):
```csv
text,text_pair
"What is AI?","Artificial Intelligence explained"
"How to cook pasta","Pasta cooking tutorial"
```

**NER** (`data/train.conll`):
```
John B-PER
Smith I-PER
works O
at O
Google B-ORG
```

### 2. Configure Your Training

Edit the configuration file for your task (e.g., `configs/kalm_embedding/classification.yaml`):

```yaml
model:
  name: "kalm-embedding-2.5"
  pretrained_path: "path/to/model"  # Update with actual model path
  max_seq_length: 512

training:
  output_dir: "./output/my-model"
  batch_size: 16
  learning_rate: 2e-5
  num_epochs: 10

data:
  train_file: "data/train.csv"
  val_file: "data/val.csv"
  text_column: "text"
  label_column: "label"
```

### 3. Train Your Model

```bash
# Text Classification
encoder-train train \
  --config configs/kalm_embedding/classification.yaml \
  --task classification

# Text Embedding
encoder-train train \
  --config configs/kalm_embedding/embedding.yaml \
  --task embedding

# Named Entity Recognition
encoder-train train \
  --config configs/kalm_embedding/ner.yaml \
  --task ner
```

### 4. Run Inference

```bash
encoder-train infer \
  --model kalm-embedding-2.5 \
  --task classification \
  --checkpoint ./output/my-model/best_model \
  --input "This is a test sentence"
```

### 5. Evaluate on Test Set

```bash
encoder-train eval \
  --config configs/kalm_embedding/classification.yaml \
  --task classification \
  --checkpoint ./output/my-model/best_model \
  --test-data data/test.csv
```

## Project Structure

```
encoder-training/
├── encoder_training/          # Main package
│   ├── core/                 # Core utilities
│   │   ├── config.py        # Configuration management
│   │   ├── data_loader.py   # Dataset loading
│   │   ├── metrics.py       # Evaluation metrics
│   │   └── utils.py         # Training utilities
│   ├── models/              # Model implementations
│   │   ├── base.py         # Base interface
│   │   ├── registry.py     # Model registry
│   │   ├── kalm_embedding/ # KALM model
│   │   ├── qwen_embedding/ # Qwen model
│   │   └── multilingual_300m/ # Multilingual model
│   └── cli.py              # CLI interface
├── configs/                # Configuration templates
│   ├── kalm_embedding/
│   ├── qwen_embedding/
│   └── multilingual_300m/
├── examples/               # Example scripts and data
├── requirements.txt        # Dependencies
├── setup.py               # Package setup
└── README.md              # This file
```

## Available Models

List all registered models:

```bash
encoder-train list-models
```

Models:
- `kalm-embedding-2.5`: KALM Embedding 2.5 Instruct
- `qwen-0.6b-embedding`: Qwen 0.6B Embedding Instruct
- `multilingual-300m`: Multilingual 300M

## Adding a New Model

1. Create a new module under `encoder_training/models/your_model/`
2. Implement the trainer by inheriting from `BaseModelTrainer`
3. Register the model using the `@register_model` decorator:

```python
from encoder_training.models.base import BaseModelTrainer
from encoder_training.models.registry import register_model

@register_model("my-new-model")
class MyModelTrainer(BaseModelTrainer):
    def __init__(self, model_name, pretrained_path, device=None):
        super().__init__(model_name, pretrained_path, device)
    
    def train_classification(self, config):
        # Implementation
        pass
    
    # Implement other required methods...
```

4. The model is now available via the CLI!

## Configuration Options

### Model Config
- `name`: Model identifier
- `pretrained_path`: HuggingFace model ID or local path
- `max_seq_length`: Maximum sequence length
- `num_labels`: Number of labels (auto-inferred for classification/NER)
- `pooling_mode`: Pooling strategy for embeddings (mean, cls, max)

### Training Config
- `output_dir`: Output directory for checkpoints
- `batch_size`: Training batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `warmup_ratio`: Warmup ratio for lr scheduler
- `weight_decay`: Weight decay for optimizer
- `fp16`: Use mixed precision training
- `contrastive_loss`: Loss type for embeddings (infonce, triplet)
- `temperature`: Temperature for contrastive learning

### Data Config
- `train_file`, `val_file`, `test_file`: Data file paths
- `text_column`, `label_column`: Column names
- `data_format`: File format (csv, json, jsonl, conll)
- `max_samples`: Limit number of training samples (for debugging)

### Evaluation Config
- `eval_steps`: Evaluate every N steps
- `save_steps`: Save checkpoint every N steps
- `metric_for_best_model`: Metric to track for best model
- `save_total_limit`: Maximum number of checkpoints to keep
- `early_stopping_patience`: Early stopping patience

## Metrics

### Classification
- Accuracy
- Precision, Recall, F1 (weighted average)
- Per-class metrics

### Embedding
- Mean Reciprocal Rank (MRR)
- Recall@K (K=1,5,10,20)
- NDCG@K (K=1,5,10,20)

### NER
- Token-level: Accuracy, Precision, Recall, F1
- Entity-level: Precision, Recall, F1
- Per-entity-type metrics

## Python API

You can also use the package programmatically:

```python
from encoder_training.core.config import FullConfig
from encoder_training.models import get_model

# Load config
config = FullConfig.from_yaml("configs/kalm_embedding/classification.yaml")

# Get trainer
trainer = get_model(
    model_name="kalm-embedding-2.5",
    pretrained_path="path/to/model"
)

# Train
results = trainer.train_classification(config)
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- See `requirements.txt` for full list

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{encoder_training,
  title={Encoder Training Module},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/encoder-training}
}
```

## Support

For issues and questions, please open an issue on GitHub.

## Model-Specific Notes

### KALM Embedding 2.5 Instruct
- **Model ID**: Update `pretrained_path` in config with actual HuggingFace model ID
- **Recommended Settings**: 
  - Classification: lr=2e-5, batch_size=16
  - Embedding: lr=1e-5, batch_size=32

### Qwen 0.6B Embedding Instruct
- **Model ID**: Update `pretrained_path` in config
- Inherits behavior from KALM trainer

### Multilingual 300M
- **Model ID**: Update `pretrained_path` in config
- Supports multiple languages out of the box
- Example: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
