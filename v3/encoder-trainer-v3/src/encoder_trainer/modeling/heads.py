import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationHead(nn.Module):
    """
    Standard classification head (Linear or MLP).
    """
    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, num_labels)
        
    def forward(self, x):
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits

class SoftContextualizedHead(nn.Module):
    """
    Soft Contextualized Encoder (SCE) Head.
    Computes similarity between query embedding and candidate label embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # Query Adapter: Projects text embedding to query space
        self.query_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # Normalize for stability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim) # Final projection
        )
        
        # Key Projection: Projects label embeddings to key space
        # Assuming label embeddings come from same encoder (input_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, text_embs: torch.Tensor, label_embs: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            text_embs: [Batch, Dim]
            label_embs: [Batch, K, Dim] (Candidate label embeddings)
            mask: [Batch, K] (1 if valid candidate, 0 if padding)
            
        Returns:
            scores: [Batch, K]
        """
        Q = self.query_adapter(text_embs).unsqueeze(1) # [B, 1, Hidden]
        K = self.key_proj(label_embs)                  # [B, K, Hidden]
        
        # Dot-product attention
        # [B, 1, H] @ [B, H, K] -> [B, 1, K]
        scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(1) # [B, K]
        
        # Scale by sqrt(d) (standard attention scaling) to prevent gradient explosion
        d_k = K.size(-1)
        scores = scores / (d_k ** 0.5)
        
        if mask is not None:
            # Mask padded candidates with very large negative value
            scores = scores.masked_fill(mask == 0, -1e9)
            
        return scores
