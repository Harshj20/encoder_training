from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union

@dataclass
class ModelConfig:
    model_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    mode: Literal["frozen_head", "full_finetune", "lora", "adapter", "sce"] = "frozen_head"
    # Note: Pooling is now handled automatically by SentenceTransformer logic.
    # This field is kept for backward compatibility or future override implementation.
    pooling: Literal["cls", "mean"] = "mean"
    
    # PEFT / LoRA specific
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # SCE specific
    sce_hidden_dim: int = 512
    
@dataclass
class DataConfig:
    dataset_path: str = "data/dataset.csv"  # Path to local file or HF dataset name
    text_column: str = "text"
    label_column: str = "label"
    
    # SCE specific: list of candidate labels for triple format
    candidates_column: Optional[str] = None 
    
    max_length: int = 128
    validation_split: float = 0.1
    
    # Dynamic Label Sampling for SCE
    dynamic_label_sampling: bool = False
    num_negatives: int = 4

@dataclass
class TrainerConfig:
    output_dir: str = "output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 0
    weight_decay: float = 0.01
    logging_steps: int = 10
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 1
    fp16: bool = False
    use_cpu: bool = False
