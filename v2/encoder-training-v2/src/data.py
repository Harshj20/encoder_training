"""Streaming dataset loading, tokenization, and collation."""

import torch
from datasets import load_dataset, IterableDataset, Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging


def load_streaming_dataset(
    dataset_path: str,
    split: str = "train",
    streaming: bool = True,
    max_samples: Optional[int] = None,
) -> Union[IterableDataset, Dataset]:
    """
    Load dataset with streaming support.
    
    Args:
        dataset_path: Path to dataset (file path or HF dataset name)
        split: Dataset split
        streaming: Whether to use streaming mode
        max_samples: Maximum number of samples (for debugging)
    
    Returns:
        Dataset or IterableDataset
    """
    try:
        # Try loading as HuggingFace dataset
        dataset = load_dataset(dataset_path, split=split, streaming=streaming)
    except Exception as e:
        # Try loading from file
        if dataset_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=dataset_path, split=split, streaming=streaming)
        elif dataset_path.endswith('.json') or dataset_path.endswith('.jsonl'):
            dataset = load_dataset('json', data_files=dataset_path, split=split, streaming=streaming)
        else:
            raise ValueError(f"Unable to load dataset from {dataset_path}: {e}")
    
    # Limit samples for debugging
    if max_samples and streaming:
        dataset = dataset.take(max_samples)
    elif max_samples and not streaming:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset


def prepare_single_text_example(
    example: Dict[str, Any],
    text_field: str = "text",
    label_field: str ="label"
) -> Dict[str, Any]:
    """
    Prepare single-text classification example.
    
    Args:
        example: Raw example dict
        text_field: Name of text field
        label_field: Name of label field
    
    Returns:
        Prepared example with 'text' and 'label' keys
    """
    return {
        "text": example[text_field],
        "label": example.get(label_field, 0),  # Default to 0 if no label
    }


def prepare_pair_example(
    example: Dict[str, Any],
    text1_field: str = "text1",
    text2_field: str = "text2",
    label_field: str = "label"
) -> Dict[str, Any]:
    """
    Prepare text-pair example.
    
    Args:
        example: Raw example dict
        text1_field: Name of first text field
        text2_field: Name of second text field
        label_field: Name of label field
    
    Returns:
        Prepared example with 'text1', 'text2', and 'label' keys
    """
    return {
        "text1": example[text1_field],
        "text2": example[text2_field],
        "label": example.get(label_field, 0),
    }


def prepare_triplet_example(
    example: Dict[str, Any],
    anchor_field: str = "anchor",
    positive_field: str = "positive",
    negative_field: str = "negative"
) -> Dict[str, Any]:
    """
    Prepare triplet example.
    
    Args:
        example: Raw example dict
        anchor_field: Name of anchor field
        positive_field: Name of positive field
        negative_field: Name of negative field
    
    Returns:
        Prepared example with 'anchor', 'positive', 'negative' keys
    """
    return {
        "anchor": example[anchor_field],
        "positive": example[positive_field],
        "negative": example[negative_field],
    }


class TokenizeFunction:
    """Tokenization function for batched map."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        data_type: str = "single"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_type = data_type
    
    def __call__(self, examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Tokenize batch of examples.
        
        Args:
            examples: Batch of examples (dict of lists)
        
        Returns:
            Tokenized batch
        """
        if self.data_type == "single":
            # Single text
            tokenized = self.tokenizer(
                examples["text"],
                max_length=self.max_length,
                truncation=True,
                # Don't pad here - let collator handle it dynamically
            )
            if "label" in examples:
                tokenized["labels"] = examples["label"]
        
        elif self.data_type == "pair":
            # Text pairs
            tokenized = self.tokenizer(
                examples["text1"],
                examples["text2"],
                max_length=self.max_length,
                truncation=True,
            )
            if "label" in examples:
                tokenized["labels"] = examples["label"]
        
        elif self.data_type == "triplet":
            # Triplet - tokenize each separately
            tokenized = {}
            
            anchor_tokens = self.tokenizer(
                examples["anchor"],
                max_length=self.max_length,
                truncation=True,
            )
            tokenized["anchor_input_ids"] = anchor_tokens["input_ids"]
            tokenized["anchor_attention_mask"] = anchor_tokens["attention_mask"]
            
            positive_tokens = self.tokenizer(
                examples["positive"],
                max_length=self.max_length,
                truncation=True,
            )
            tokenized["positive_input_ids"] = positive_tokens["input_ids"]
            tokenized["positive_attention_mask"] = positive_tokens["attention_mask"]
            
            negative_tokens = self.tokenizer(
                examples["negative"],
                max_length=self.max_length,
                truncation=True,
            )
            tokenized["negative_input_ids"] = negative_tokens["input_ids"]
            tokenized["negative_attention_mask"] = negative_tokens["attention_mask"]
        
        return tokenized


@dataclass
class DataCollator:
    """
    Data collator with dynamic padding for single/pair/triplet modes.
    """
    
    tokenizer: PreTrainedTokenizer
    data_type: str = "single"  # single, pair, triplet
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch with dynamic padding.
        
        Args:
            features: List of feature dicts
        
        Returns:
            Batched tensors
        """
        if self.data_type == "single" or self.data_type == "pair":
            # Extract input_ids and attention_mask
            input_ids = [f["input_ids"] for f in features]
            
            # Pad sequences
            batch = self.tokenizer.pad(
                {"input_ids": input_ids},
                padding=True,
                return_tensors="pt"
            )
            
            # Add labels if present
            if "labels" in features[0]:
                batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            
            return batch
        
        elif self.data_type == "triplet":
            # Pad anchor, positive, negative separately
            anchor_input_ids = [f["anchor_input_ids"] for f in features]
            positive_input_ids = [f["positive_input_ids"] for f in features]
            negative_input_ids = [f["negative_input_ids"] for f in features]
            
            anchor_batch = self.tokenizer.pad(
                {"input_ids": anchor_input_ids},
                padding=True,
                return_tensors="pt"
            )
            positive_batch = self.tokenizer.pad(
                {"input_ids": positive_input_ids},
                padding=True,
                return_tensors="pt"
            )
            negative_batch = self.tokenizer.pad(
                {"input_ids": negative_input_ids},
                padding=True,
                return_tensors="pt"
            )
            
            return {
                "anchor_input_ids": anchor_batch["input_ids"],
                "anchor_attention_mask": anchor_batch["attention_mask"],
                "positive_input_ids": positive_batch["input_ids"],
                "positive_attention_mask": positive_batch["attention_mask"],
                "negative_input_ids": negative_batch["input_ids"],
                "negative_attention_mask": negative_batch["attention_mask"],
            }
        
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")


def create_dataloader(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    data_config: Dict[str, Any],
    training_config: Dict[str, Any],
    split: str = "train",
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with streaming dataset.
    
    Args:
        dataset_path: Path to dataset
        tokenizer: Tokenizer
        data_config: Data configuration
        training_config: Training configuration
        split: Dataset split
    
    Returns:
        DataLoader
    """
    # Load dataset
    dataset = load_streaming_dataset(
        dataset_path=dataset_path,
        split=split,
        streaming=data_config.get("streaming", True),
        max_samples=data_config.get("max_samples"),
    )
    
    # Prepare examples based on data type
    data_type = data_config.get("type", "single")
    
    if data_type == "single":
        prepare_fn = lambda x: prepare_single_text_example(
            x,
            text_field=data_config.get("text_field", "text"),
            label_field=data_config.get("label_field", "label")
        )
    elif data_type == "pair":
        prepare_fn = lambda x: prepare_pair_example(
            x,
            text1_field=data_config.get("text1_field", "text1"),
            text2_field=data_config.get("text2_field", "text2"),
            label_field=data_config.get("label_field", "label")
        )
    elif data_type == "triplet":
        prepare_fn = lambda x: prepare_triplet_example(
            x,
            anchor_field=data_config.get("anchor_field", "anchor"),
            positive_field=data_config.get("positive_field", "positive"),
            negative_field=data_config.get("negative_field", "negative")
        )
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Map prepare function
    dataset = dataset.map(prepare_fn)
    
    # Tokenize
    tokenize_fn = TokenizeFunction(
        tokenizer=tokenizer,
        max_length=training_config.get("max_seq_length", 256),
        data_type=data_type
    )
    
    # Apply tokenization with batched=True for efficiency
    # Note: for streaming datasets, this doesn't load everything into memory
    dataset = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    
    # Create collator
    collator = DataCollator(tokenizer=tokenizer, data_type=data_type)
    
    # Determine batch size
    batch_size = training_config.get("per_device_train_batch_size", 8)
    if split != "train":
        batch_size = training_config.get("per_device_eval_batch_size", batch_size * 2)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,  # Streaming datasets don't support multiprocessing well
    )
    
    return dataloader
