from transformers import Trainer
from torch import nn

class EncoderTrainer(Trainer):
    """
    Custom Trainer for Unified Encoder.
    Mainly allows custom behavior if needed in future (e.g. custom loss logging).
    Currently leverages standard Trainer logic since model handles loss computation.
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # The model returns a dict with 'loss' key.
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        return (loss, outputs) if return_outputs else loss
