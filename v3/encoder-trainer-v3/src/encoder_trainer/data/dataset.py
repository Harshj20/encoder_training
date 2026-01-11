import logging
from typing import List, Dict, Union, Optional, Any
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset
import torch
from torch.utils.data import Dataset
from ..config import DataConfig

logger = logging.getLogger(__name__)

class LabelManager:
    """
    Manages label to ID mapping for fixed classification tasks.
    """
    def __init__(self, labels: List[str]):
        self.labels = sorted(list(set(labels)))
        self.label2id = {l: i for i, l in enumerate(self.labels)}
        self.id2label = {i: l for i, l in enumerate(self.labels)}
        
    def __len__(self):
        return len(self.labels)
    
    def get_id(self, label: str) -> int:
        return self.label2id.get(label, -1)

    def get_label(self, idx: int) -> str:
        return self.id2label.get(idx, "UNKNOWN")

class UnifiedDataset(Dataset):
    """
    Dataset wrapper supporting both standard classification and SCE triplets.
    """
    def __init__(self, config: DataConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.data: List[Dict[str, Any]] = []
        self.label_manager: Optional[LabelManager] = None
        
        self._load_data()

    def _load_data(self):
        """
        Loads data from file or HF datasets.
        """
        logger.info(f"Loading data from {self.config.dataset_path}...")
        
        # Simple loading strategy: supports CSV, JSON, JSONL via pandas/HF
        if self.config.dataset_path.endswith(".csv"):
            df = pd.read_csv(self.config.dataset_path)
        elif self.config.dataset_path.endswith(".json") or self.config.dataset_path.endswith(".jsonl"):
            df = pd.read_json(self.config.dataset_path, lines=self.config.dataset_path.endswith(".jsonl"))
        else:
            # Assume it's an HF dataset name
            try:
                dataset = load_dataset(self.config.dataset_path, split=self.split)
                df = dataset.to_pandas()
            except Exception as e:
                raise ValueError(f"Could not load dataset from {self.config.dataset_path}: {e}")

        # Validate columns
        if self.config.text_column not in df.columns:
            raise ValueError(f"Text column '{self.config.text_column}' not found in dataset. Available: {df.columns.tolist()}")

        # Handle Label extraction
        if self.config.candidates_column:
            # SCE Mode: We might not have a global label set, just candidates per row
            pass
        elif self.config.label_column in df.columns:
            # Standard Mode: Build label manager from all unique labels in 'train'
            # Note: In real prod, label manager should be built from a separate vocab file or fixed set.
            # Here we infer it if not provided (simplified).
            unique_labels = df[self.config.label_column].unique().tolist()
            # Convert to strings to be safe
            unique_labels = [str(l) for l in unique_labels]
            self.label_manager = LabelManager(unique_labels)
        
        # Convert to list of dicts for faster access
        self.data = df.to_dict('records')
        logger.info(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[self.config.text_column]
        
        sample = {"text": text}
        
        # Standard Label handling
        if self.label_manager and self.config.label_column in item:
            label_raw = str(item[self.config.label_column])
            label_id = self.label_manager.get_id(label_raw)
            if label_id != -1:
                sample["labels"] = label_id
        
        # SCE Candidate handling
        if self.config.candidates_column and self.config.candidates_column in item:
            candidates = item[self.config.candidates_column]
            # candidates should be a list of strings
            sample["candidates"] = candidates
            
            # If label_column is present, it might be an index or the string itself
            if self.config.label_column in item:
                target = item[self.config.label_column]
                # If target is string, find index in candidates
                if isinstance(target, str):
                    try:
                        target_idx = candidates.index(target)
                        sample["target_index"] = target_idx
                    except ValueError:
                        # Fallback or invalid
                        sample["target_index"] = -1
                elif isinstance(target, int):
                    sample["target_index"] = target

        return sample
