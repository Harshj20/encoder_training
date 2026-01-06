"""Model-specific configuration for KALM Embedding 2.5 Instruct."""

from dataclasses import dataclass


@dataclass
class KalmEmbeddingConfig:
    """KALM Embedding specific configuration."""
    
    # Model identifier - UPDATE THIS with actual HuggingFace model ID
    default_model_id: str = "Kalm/kalm-embedding-2.5-instruct"
    
    # Model-specific settings
    use_gradient_checkpointing: bool = False
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = 0  # Number of encoder layers to freeze
    
    # Pooling strategy for embeddings
    pooling_strategy: str = "mean"  # mean, cls, max, weighted_mean
    
    # Task-specific heads
    add_pooler_layer: bool = True
