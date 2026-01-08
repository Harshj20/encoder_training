"""Loss functions for classification, contrastive, and triplet learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    multi_label: bool = False
) -> torch.Tensor:
    """
    Classification loss (cross-entropy or BCE).
    
    Args:
        logits: Model logits (batch_size, num_labels)
        labels: Ground truth labels
        multi_label: If True, use BCEWithLogitsLoss for multi-label classification
    
    Returns:
        Loss tensor
    """
    if multi_label:
        # Multi-label classification
        loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn(logits, labels.float())
    else:
        # Single-label classification
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(logits, labels)


def contrastive_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: Optional[torch.Tensor] = None,
    temperature: float = 0.07,
    labels: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Contrastive loss (InfoNCE / NT-Xent style).
    
    Implements MultipleNegativesRankingLoss: uses in-batch negatives where
    each anchor's positive is the corresponding position, others are negatives.
    
    Args:
        embeddings_a: Anchor embeddings (batch_size, hidden_size)
        embeddings_b: Positive embeddings (batch_size, hidden_size). If None, uses in-batch negatives only
        temperature: Temperature for scaling similarities
        labels: Not used (kept for API compatibility)
    
    Returns:
        Loss tensor
    """
    if embeddings_b is None:
        # In-batch negatives only (each sample is positive for itself)
        embeddings = F.normalize(embeddings_a, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        batch_size = embeddings.shape[0]
        labels = torch.arange(batch_size, device=embeddings.device)
        loss = F.cross_entropy(similarity_matrix, labels)
    else:
        # Anchor-positive pairs with in-batch negatives
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        # Concatenate for in-batch negatives
        embeddings = torch.cat([embeddings_a, embeddings_b], dim=0)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        batch_size = embeddings_a.shape[0]
        # Positive pairs: (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=embeddings.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
    normalize: bool = True
) -> torch.Tensor:
    """
    Triplet margin loss.
    
    Args:
        anchor: Anchor embeddings (batch_size, hidden_size)
        positive: Positive embeddings (batch_size, hidden_size)
        negative: Negative embeddings (batch_size, hidden_size)
        margin: Margin for triplet loss
        normalize: Whether to normalize embeddings before computing distances
    
    Returns:
        Loss tensor
    """
    if normalize:
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
    
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
    return loss_fn(anchor, positive, negative)


class ContrastiveLossWrapper(nn.Module):
    """Wrapper for contrastive loss with configurable temperature."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return contrastive_loss(embeddings_a, embeddings_b, self.temperature, labels)


class TripletLossWrapper(nn.Module):
    """Wrapper for triplet loss with configurable margin."""
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        return triplet_loss(anchor, positive, negative, self.margin)


def get_loss_fn(loss_type: str, loss_config: dict):
    """
    Factory function to get loss function based on type.
    
    Args:
        loss_type: Type of loss (classification, contrastive, triplet)
        loss_config: Loss configuration dict
    
    Returns:
        Loss function
    """
    if loss_type == "classification":
        return lambda logits, labels: classification_loss(
            logits, labels, multi_label=loss_config.get("multi_label", False)
        )
    elif loss_type == "contrastive":
        return ContrastiveLossWrapper(temperature=loss_config.get("temperature", 0.07))
    elif loss_type == "triplet":
        return TripletLossWrapper(margin=loss_config.get("margin", 0.2))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
