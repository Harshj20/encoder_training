"""Model-specific configuration for Multilingual 300M."""

from dataclasses import dataclass


@dataclass
class Multilingual300MConfig:
    """Multilingual 300M specific configuration."""
    
    # Model identifier - UPDATE THIS with actual model path/ID
    default_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Model-specific settings
    use_gradient_checkpointing: bool = False
    freeze_embeddings: bool = False
    pooling_strategy: str = "mean"
    
    # Multilingual-specific
    supported_languages: list = None  # None means all languages
