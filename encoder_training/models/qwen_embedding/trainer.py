"""Trainer for Qwen 0.6B Embedding Instruct model.

This implementation inherits most functionality from KalmEmbeddingTrainer.
Override methods as needed for Qwen-specific behavior.
"""

from ..kalm_embedding.trainer import KalmEmbeddingTrainer
from ..registry import register_model


@register_model("qwen-0.6b-embedding")
class QwenEmbeddingTrainer(KalmEmbeddingTrainer):
    """Trainer for Qwen 0.6B Embedding Instruct model."""
    
    def __init__(
        self,
        model_name: str = "qwen-0.6b-embedding",
        pretrained_path: str = "Qwen/Qwen-0.6B-embedding-instruct",
        device=None
    ):
        super().__init__(model_name, pretrained_path, device)
    
    # Override methods here if Qwen requires different behavior
    # For example, if Qwen uses different pooling or special preprocessing
