"""Configuration management with YAML/JSON support and validation."""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class PoolingConfig:
    """Pooling configuration for sentence embeddings."""
    mode: str = "mean"  # mean, cls, max, lasttoken, weighted
    layers: list = field(default_factory=lambda: [-1])  # which transformer layers to use
    normalize: bool = True  # L2 normalize embeddings


@dataclass
class ModelConfig:
    """Model configuration."""
    name_or_path: str = "intfloat/multilingual-e5-small"
    tokenizer: Optional[str] = None
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    classifier_hidden: int = 512
    num_labels: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    mode: str = "head_only"  # head_only, full, peft
    epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    bf16: bool = False
    max_seq_length: int = 256
    optimizer: str = "adamw"
    lr: float = 5e-5
    weight_decay: float = 0.01
    scheduler: str = "linear"
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    max_steps: Optional[int] = None
    max_grad_norm: float = 1.0
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3
    output_dir: str = "./outputs"
    resume_from_checkpoint: Optional[str] = None


@dataclass
class DataConfig:
    """Data configuration."""
    type: str = "single"  # single, pair, triplet
    dataset: str = "path_or_hf_name"
    split: str = "train"
    streaming: bool = True
    text_field: str = "text"
    text1_field: str = "text1"
    text2_field: str = "text2"
    anchor_field: str = "anchor"
    positive_field: str = "positive"
    negative_field: str = "negative"
    label_field: str = "label"
    max_samples: Optional[int] = None  # For debugging
    num_proc: int = 4  # For non-streaming map


@dataclass
class PEFTConfig:
    """PEFT (LoRA) configuration."""
    enabled: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[list] = None  # None means auto-detect


@dataclass
class DeepSpeedConfig:
    """DeepSpeed configuration."""
    enabled: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    config_file: Optional[str] = None


@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str = "classification"  # classification, contrastive, triplet
    margin: float = 0.2  # For triplet loss
    temperature: float = 0.07  # For contrastive loss
    multi_label: bool = False  # Use BCEWithLogitsLoss instead of CrossEntropy


@dataclass
class Config:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    deepspeed: DeepSpeedConfig = field(default_factory=DeepSpeedConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Load config from dictionary."""
        # Parse pooling config
        pooling_dict = config_dict.get("model", {}).get("pooling", {})
        pooling = PoolingConfig(**pooling_dict) if pooling_dict else PoolingConfig()
        
        # Parse each section
        model_dict = config_dict.get("model", {})
        model_dict["pooling"] = pooling
        model = ModelConfig(**{k: v for k, v in model_dict.items() if k != "pooling"})
        model.pooling = pooling
        
        training = TrainingConfig(**config_dict.get("training", {}))
        data = DataConfig(**config_dict.get("data", {}))
        peft = PEFTConfig(**config_dict.get("peft", {}))
        deepspeed = DeepSpeedConfig(**config_dict.get("deepspeed", {}))
        loss = LossConfig(**config_dict.get("loss", {}))
        
        return cls(
            model=model,
            training=training,
            data=data,
            peft=peft,
            deepspeed=deepspeed,
            loss=loss,
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        # Custom conversion to handle nested dataclasses
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, dict) and "pooling" in value:
                # Special handling for model with pooling
                result[key] = value
            else:
                result[key] = value
        return result

    def save_yaml(self, path: Union[str, Path]):
        """Save config to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: Union[str, Path]):
        """Save config to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def validate(self):
        """Validate configuration."""
        # Training mode
        assert self.training.mode in ["head_only", "full", "peft"], \
            f"Invalid training mode: {self.training.mode}"
        
        # Data type
        assert self.data.type in ["single", "pair", "triplet"], \
            f"Invalid data type: {self.data.type}"
        
        # Pooling mode
        assert self.model.pooling.mode in ["mean", "cls", "max", "lasttoken", "weighted"], \
            f"Invalid pooling mode: {self.model.pooling.mode}"
        
        # Loss type
        assert self.loss.type in ["classification", "contrastive", "triplet"], \
            f"Invalid loss type: {self.loss.type}"
        
        # PEFT mode requires peft.enabled
        if self.training.mode == "peft" and not self.peft.enabled:
            raise ValueError("PEFT mode requires peft.enabled=true")
        
        # Validate paths exist if resuming
        if self.training.resume_from_checkpoint:
            ckpt_path = Path(self.training.resume_from_checkpoint)
            if not ckpt_path.exists():
                raise ValueError(f"Checkpoint not found: {ckpt_path}")
        
        return True
