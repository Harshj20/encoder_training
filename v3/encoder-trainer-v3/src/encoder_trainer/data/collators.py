from typing import List, Dict, Any, Union
import torch
from transformers import PreTrainedTokenizerBase

class DynamicLabelCollator:
    """
    Collator for UnifiedDataset.
    Handles:
    1. Text options (padding, truncation)
    2. Standard integer labels
    3. SCE candidate label tokenization (dynamic sets)
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 128, add_candidates: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_candidates = add_candidates

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {}
        
        # 1. Process Main Text
        texts = [f["text"] for f in features]
        text_encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        batch["input_ids"] = text_encodings["input_ids"]
        batch["attention_mask"] = text_encodings["attention_mask"]
        
        # 2. Process Standard Labels
        if "labels" in features[0]:
            batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            
        # 3. Process SCE Candidates (if present)
        if "candidates" in features[0]:
            # Expecting list of strings per sample
            # Strategy: Pad to max_K candidates in this batch
            all_candidates = [f["candidates"] for f in features] # List[List[str]]
            
            # Find max K in this batch
            max_k = max(len(c) for c in all_candidates)
            
            # Flatten but keep track of boundaries or pad manually
            # Padded approach: [B, K_max, L]
            # Warning: This can be memory intensive if K or L is large.
            # Efficient optimized approach: Flatten all to [Total_Candidates, L] and use indices?
            # For simplicity in 'UnifiedModel', let's use [B, K_max, L] for now, masking empty slots.
            
            # We need to construct a huge list of texts to tokenize
            # To handle padding candidates, we can just use the tokenizer on the flattened list
            # and then reshape/pad the tensors.
            
            flat_candidates = []
            candidate_mask = [] # 1 if real candidate, 0 if padded candidate
            
            for candidates in all_candidates:
                # Add real candidates
                flat_candidates.extend(candidates)
                # Add placeholders for padding to reach max_k
                num_pad = max_k - len(candidates)
                if num_pad > 0:
                    flat_candidates.extend([""] * num_pad) 
                
                # Create mask: [1, 1, ..., 0, 0]
                mask = [1] * len(candidates) + [0] * num_pad
                candidate_mask.append(mask)
            
            # Tokenize all
            cand_encodings = self.tokenizer(
                flat_candidates,
                padding=True,
                truncation=True,
                max_length=self.max_length, # Maybe smaller max_len for labels?
                return_tensors="pt"
            )
            
            # Reshape [B * K, L] -> [B, K, L]
            b_size = len(features)
            seq_len = cand_encodings["input_ids"].shape[1]
            
            batch["candidate_input_ids"] = cand_encodings["input_ids"].view(b_size, max_k, seq_len)
            batch["candidate_attention_mask"] = cand_encodings["attention_mask"].view(b_size, max_k, seq_len)
            
            # Validity mask for the candidates themselves (not just tokens)
            batch["candidate_set_mask"] = torch.tensor(candidate_mask, dtype=torch.long) # [B, K]

            if "target_index" in features[0]:
                 batch["target_index"] = torch.tensor([f["target_index"] for f in features], dtype=torch.long)

        return batch
