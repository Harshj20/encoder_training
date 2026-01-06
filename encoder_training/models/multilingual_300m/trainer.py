"""Trainer for Multilingual 300M model."""

from ..kalm_embedding.trainer import KalmEmbeddingTrainer
from ..registry import register_model


@register_model("multilingual-300m")
class Multilingual300MTrainer(KalmEmbeddingTrainer):
    """Trainer for Multilingual 300M model."""
    
    def __init__(
        self,
        model_name: str = "multilingual-300m",
        pretrained_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=None
    ):
        super().__init__(model_name, pretrained_path, device)
    
    # Override methods here if multilingual model requires different behavior
