"""Model-specific configuration for Qwen 0.6B Embedding Instruct."""

from dataclasses import dataclass


@dataclass
class QwenEmbeddingConfig:
    """Qwen Embedding specific configuration."""
    
    # Model identifier - UPDATE THIS with actual model path/ID
    default_model_id: str = "Qwen/Qwen-0.6B-embedding-instruct"
    
    # Model-specific settings
    use_gradient_checkpointing: bool = False
    freeze_embeddings: bool = False
    pooling_strategy: str = "mean"
