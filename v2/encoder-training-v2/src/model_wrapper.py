"""Model wrapper with encoder, pooling, classification head, and PEFT support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
)
from typing import Dict, Any, Optional, List
import logging

# PEFT imports - graceful fallback if not available
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available. Install with: pip install peft")


class Pooling(nn.Module):
    """
    Sentence-Transformers style pooling layer.
    
    Supports: mean, cls, max, lasttoken, weighted pooling.
    """
    
    def __init__(
        self,
        mode: str = "mean",
        layers: List[int] = [-1],
        normalize: bool = True,
        hidden_size: Optional[int] = None
    ):
        super().__init__()
        self.mode = mode
        self.layers = layers
        self.normalize = normalize
        
        # For weighted pooling, we need a learned attention vector
        if mode == "weighted" and hidden_size is not None:
            self.attention_weights = nn.Parameter(torch.randn(hidden_size) * 0.01)
        else:
            self.attention_weights = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states to fixed-size vector.
        
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            Pooled embeddings (batch_size, hidden_size)
        """
        if self.mode == "mean":
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        
        elif self.mode == "cls":
            # CLS token (first token)
            embeddings = hidden_states[:, 0, :]
        
        elif self.mode == "max":
            # Max pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.clone()
            hidden_states[input_mask_expanded == 0] = -1e9  # Set padding to large negative
            embeddings = torch.max(hidden_states, dim=1)[0]
        
        elif self.mode == "lasttoken":
            # Last non-padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            embeddings = hidden_states[torch.arange(batch_size), sequence_lengths]
        
        elif self.mode == "weighted":
            # Weighted pooling using learned attention
            if self.attention_weights is None:
                raise ValueError("Weighted pooling requires hidden_size to be specified")
            
            # Compute attention scores
            attention_scores = torch.matmul(hidden_states, self.attention_weights)  # (batch, seq_len)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
            attention_probs = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            
            # Weighted sum
            embeddings = torch.sum(hidden_states * attention_probs, dim=1)
        
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")
        
        # L2 normalization
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class ClassificationHead(nn.Module):
    """Classification head with optional hidden layer."""
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim and hidden_dim > 0:
            # Two-layer MLP
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels)
            )
        else:
            # Single linear layer
            self.classifier = nn.Linear(input_dim, num_labels)
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.classifier(embeddings)


class EncoderClassifier(nn.Module):
    """
    Encoder model with pooling and classification head.
    
    Supports:
    - Multiple pooling strategies
    - Detachable classification head
    - PEFT (LoRA) integration
    - Freeze/unfreeze for different training modes
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        pooling_config: Optional[Dict[str, Any]] = None,
        classifier_hidden: Optional[int] = 512,
        num_labels: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.model_name = model_name_or_path
        
        # Load config to get hidden size
        config = AutoConfig.from_pretrained(model_name_or_path)
        self.hidden_size = config.hidden_size
        
        # Load encoder
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        
        # Load tokenizer
        tokenizer_path = tokenizer_name_or_path or model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Setup pooling
        pooling_config = pooling_config or {"mode": "mean", "layers": [-1], "normalize": True}
        self.pooling = Pooling(
            mode=pooling_config.get("mode", "mean"),
            layers=pooling_config.get("layers", [-1]),
            normalize=pooling_config.get("normalize", True),
            hidden_size=self.hidden_size
        )
        
        # Classification head
        self.classifier_head = ClassificationHead(
            input_dim=self.hidden_size,
            num_labels=num_labels,
            hidden_dim=classifier_hidden,
            dropout=dropout
        )
        
        self.num_labels = num_labels
        self.peft_enabled = False
    
    def freeze_base(self):
        """Freeze encoder parameters for head-only training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Keep pooling trainable if using weighted mode
        if self.pooling.mode == "weighted":
            for param in self.pooling.parameters():
                param.requires_grad = True
        
        logging.info("Froze encoder parameters")
    
    def unfreeze_all(self):
        """Unfreeze all parameters for full finetuning."""
        for param in self.parameters():
            param.requires_grad = True
        
        logging.info("Unfroze all parameters")
    
    def enable_peft(self, peft_config: Dict[str, Any]):
        """
        Enable PEFT (LoRA) adapters.
        
        Args:
            peft_config: PEFT configuration dict
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT not available. Install with: pip install peft\n"
                "See: https://github.com/huggingface/peft"
            )
        
        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=peft_config.get("lora_rank", 8),
            lora_alpha=peft_config.get("lora_alpha", 32),
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            target_modules=peft_config.get("target_modules"),  # None = auto-detect
        )
        
        # Apply PEFT to encoder
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.peft_enabled = True
        
        logging.info(f"Enabled PEFT with rank={lora_config.r}, alpha={lora_config.lora_alpha}")
        self.encoder.print_trainable_parameters()
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode inputs to embeddings.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            Embeddings (batch_size, hidden_size)
        """
        # Run encoder
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Pool hidden states
        embeddings = self.pooling(outputs.last_hidden_state, attention_mask)
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "classification",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Labels for classification
            mode: "classification" or "embedding"
        
        Returns:
            Dict with loss, logits, and/or embeddings
        """
        # Get embeddings
        embeddings = self.encode(input_ids, attention_mask, **kwargs)
        
        output = {"embeddings": embeddings}
        
        if mode == "classification":
            # Classification mode
            logits = self.classifier_head(embeddings)
            output["logits"] = logits
            
            # Compute loss if labels provided
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                output["loss"] = loss
        
        return output
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        from pathlib import Path
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save encoder (handles PEFT automatically)
        self.encoder.save_pretrained(save_directory / "encoder")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_directory / "tokenizer")
        
        # Save full model state
        torch.save({
            'pooling_state_dict': self.pooling.state_dict(),
            'classifier_state_dict': self.classifier_head.state_dict(),
            'config': {
                'model_name': self.model_name,
                'num_labels': self.num_labels,
                'hidden_size': self.hidden_size,
                'peft_enabled': self.peft_enabled,
            }
        }, save_directory / "model_head.pt")
        
        logging.info(f"Model saved to {save_directory}")
