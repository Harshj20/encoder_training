"""Configuration management for training module."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import json


@dataclass
class BaseConfig:
    """Base configuration class with serialization support."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class ModelConfig(BaseConfig):
    """Model-specific configuration."""
    
    name: str = "kalm-embedding-2.5"
    pretrained_path: str = ""  # HuggingFace model ID or local path
    max_seq_length: int = 512
    num_labels: Optional[int] = None  # For classification/NER
    dropout: float = 0.1
    pooling_mode: str = "mean"  # For embeddings: mean, cls, max
    use_crf: bool = False  # For NER
    
    # Task-specific settings
    embedding_dim: Optional[int] = None
    projection_dim: Optional[int] = None  # For embedding projection


@dataclass
class DataConfig(BaseConfig):
    """Data loading configuration."""
    
    train_file: str = ""
    val_file: str = ""
    test_file: str = ""
    
    # Column names
    text_column: str = "text"
    label_column: str = "label"
    text_pair_column: Optional[str] = None  # For embeddings
    
    # Data format
    data_format: str = "csv"  # csv, json, jsonl, hf_dataset
    
    # Preprocessing
    lowercase: bool = False
    remove_special_chars: bool = False
    max_samples: Optional[int] = None  # For debugging
    
    # For NER
    token_column: Optional[str] = "tokens"
    tag_column: Optional[str] = "tags"
    tagging_scheme: str = "BIO"  # BIO, BIOES


@dataclass
class TrainingConfig(BaseConfig):
    """Training hyperparameters."""
    
    output_dir: str = "./output"
    
    # Training params
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Precision
    fp16: bool = False
    bf16: bool = False
    
    # Optimizer
    optimizer: str = "adamw"  # adamw, sgd, adafactor
    scheduler: str = "linear"  # linear, cosine, constant
    
    # Multi-GPU
    dataloader_num_workers: int = 4
    
    # Regularization
    label_smoothing: float = 0.0
    
    # Task-specific
    contrastive_loss: str = "infonce"  # For embeddings: infonce, triplet
    temperature: float = 0.05  # For contrastive learning
    margin: float = 0.5  # For triplet loss


@dataclass
class EvaluationConfig(BaseConfig):
    """Evaluation configuration."""
    
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    metric_for_best_model: str = "f1"  # f1, accuracy, loss, etc.
    greater_is_better: bool = True
    
    save_total_limit: int = 3  # Max checkpoints to keep
    load_best_model_at_end: bool = True
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0


@dataclass
class FullConfig(BaseConfig):
    """Complete configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General settings
    seed: int = 42
    device: str = "cuda"  # cuda, cpu, auto
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create full config from nested dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        evaluation_config = EvaluationConfig(**config_dict.get("evaluation", {}))
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            evaluation=evaluation_config,
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
        )
    
    def validate(self) -> None:
        """Validate configuration."""
        # Check required fields
        if not self.model.pretrained_path:
            raise ValueError("model.pretrained_path is required")
        
        if not self.data.train_file:
            raise ValueError("data.train_file is required")
        
        # Check output directory
        Path(self.training.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate learning rate
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        # Validate batch size
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
