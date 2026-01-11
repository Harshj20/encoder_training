import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from .heads import ClassificationHead, SoftContextualizedHead
from ..config import ModelConfig

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

class UnifiedEncoderClassifier(nn.Module):
    """
    Unified Encoder-Only Classifier (v3).
    Wraps a SentenceTransformer model to handle encoding and pooling automatically.
    Supports:
    - Feature extraction (frozen encoder)
    - Full finetuning
    - LoRA / Adapters (applied to underlying transformer)
    - Soft Contextualized Encoder (SCE)
    """
    def __init__(self, config: ModelConfig, num_labels: int = 0):
        super().__init__()
        self.config = config
        
        # Load Base Encoder via SentenceTransformer
        # This automatically handles modules.json and pooling definitions
        self.encoder = SentenceTransformer(config.model_name_or_path, device="cpu")
        hidden_size = self.encoder.get_sentence_embedding_dimension()

        # Apply Adaptation Strategy
        if config.mode == "lora":
            if not PEFT_AVAILABLE:
                raise ImportError("PEFT not installed but mode='lora' requested.")
            
            # SentenceTransformer usually has the Transformer model at index 0
            # We apply LoRA to this sub-module
            transformer_module = self.encoder[0]
            if hasattr(transformer_module, "auto_model"):
                target_model = transformer_module.auto_model
                
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, 
                    inference_mode=False, 
                    r=config.lora_rank, 
                    lora_alpha=config.lora_alpha, 
                    lora_dropout=config.lora_dropout,
                    target_modules=config.lora_target_modules
                )
                
                # Wrap the auto_model with PEFT
                transformer_module.auto_model = get_peft_model(target_model, peft_config)
                transformer_module.auto_model.print_trainable_parameters()
            else:
                # Fallback or warning if structure is different
                print("Warning: Could not find auto_model in encoder[0]. LoRA might not be applied correctly.")
                
        elif config.mode == "frozen_head":
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        # Setup Head
        if config.mode == "sce":
            self.head = SoftContextualizedHead(hidden_size, config.sce_hidden_dim)
        else:
            if num_labels == 0:
                raise ValueError("num_labels must be > 0 for fixed classification modes")
            self.head = ClassificationHead(hidden_size, num_labels)

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        labels=None, 
        candidate_input_ids=None, 
        candidate_attention_mask=None,
        candidate_set_mask=None,
        target_index=None,
        **kwargs
    ):
        """
        Forward pass handling both standard and SCE inputs.
        Delegates encoding to SentenceTransformer.
        """
        # 1. Encode Main Text (Query)
        # SentenceTransformer expects a dict with input keys
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        
        # Forward pass through ST returns dict with 'sentence_embedding'
        # We use strict=False to ignore extra args if any not needed by modules
        # but ST forward usually takes features dict.
        outputs = self.encoder(features)
        query_emb = outputs['sentence_embedding']
        
        logits = None
        loss = None
        
        # 2. Head Forward
        if isinstance(self.head, SoftContextualizedHead):
            if candidate_input_ids is None:
                raise ValueError("candidate_input_ids required for SCE mode")
            
            # Encode Candidates using the SAME encoder (re-entrant)
            # candidate_input_ids: [B, K, L]
            b, k, l = candidate_input_ids.size()
            
            # Flatten to [B*K, L] for efficient batching
            flat_input_ids = candidate_input_ids.view(b * k, l)
            flat_attention_mask = candidate_attention_mask.view(b * k, l)
            
            cand_features = {'input_ids': flat_input_ids, 'attention_mask': flat_attention_mask}
            cand_outputs = self.encoder(cand_features)
            cand_embs = cand_outputs['sentence_embedding'] # [B*K, Hidden]
            
            # Reshape back to [B, K, Hidden]
            cand_embs = cand_embs.view(b, k, -1)
            
            # Compute similarity scores
            logits = self.head(query_emb, cand_embs, mask=candidate_set_mask) # [B, K]
            
            if target_index is not None:
                # SCE Loss: Cross Entropy over the K candidates
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, target_index)
                
        else:
            # Standard Fixed Classification
            logits = self.head(query_emb) # [B, NumLabels]
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)

        return {"logits": logits, "loss": loss}
