"""Dataset loading and preprocessing utilities."""

import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class InputExample:
    """A single training/test example."""
    
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[Union[str, int, List[str]]] = None
    tokens: Optional[List[str]] = None  # For NER
    tags: Optional[List[str]] = None  # For NER


class ClassificationDataset(Dataset):
    """Dataset for text classification."""
    
    def __init__(
        self,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_to_id: Optional[Dict[str, int]] = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            example.text_a,
            example.text_b,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        
        # Add label if available
        if example.label is not None:
            if self.label_to_id is not None:
                label_id = self.label_to_id[example.label]
            else:
                label_id = example.label
            item['labels'] = torch.tensor(label_id, dtype=torch.long)
        
        return item


class EmbeddingDataset(Dataset):
    """Dataset for embedding/contrastive learning."""
    
    def __init__(
        self,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize anchor text
        anchor_encoding = self.tokenizer(
            example.text_a,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'anchor_input_ids': anchor_encoding['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoding['attention_mask'].squeeze(0),
        }
        
        # Tokenize positive text if available
        if example.text_b is not None:
            positive_encoding = self.tokenizer(
                example.text_b,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            item['positive_input_ids'] = positive_encoding['input_ids'].squeeze(0)
            item['positive_attention_mask'] = positive_encoding['attention_mask'].squeeze(0)
        
        return item


class NERDataset(Dataset):
    """Dataset for Named Entity Recognition."""
    
    def __init__(
        self,
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_to_id: Optional[Dict[str, int]] = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id or {}
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize with special handling for word-level tokens
        encoding = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Align labels with tokenized input
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get -100 (ignored in loss)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                tag = example.tags[word_idx]
                label_ids.append(self.label_to_id.get(tag, 0))
            else:
                # Subsequent tokens of a word get -100
                label_ids.append(-100)
            
            previous_word_idx = word_idx
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label_ids, dtype=torch.long),
        }
        
        return item


def load_classification_data(
    file_path: str,
    text_column: str = 'text',
    label_column: str = 'label',
    file_format: str = 'csv',
    max_samples: Optional[int] = None,
) -> List[InputExample]:
    """
    Load classification data from file.
    
    Args:
        file_path: Path to data file
        text_column: Name of text column
        label_column: Name of label column
        file_format: File format (csv, json, jsonl)
        max_samples: Maximum samples to load
    
    Returns:
        List of InputExamples
    """
    file_path = Path(file_path)
    
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'json':
        df = pd.read_json(file_path)
    elif file_format == 'jsonl':
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    examples = []
    for idx, row in df.iterrows():
        examples.append(InputExample(
            guid=f"train-{idx}",
            text_a=str(row[text_column]),
            label=row[label_column] if label_column in row else None,
        ))
    
    return examples


def load_embedding_data(
    file_path: str,
    text_column: str = 'text',
    text_pair_column: Optional[str] = 'text_pair',
    file_format: str = 'csv',
    max_samples: Optional[int] = None,
) -> List[InputExample]:
    """
    Load embedding/contrastive learning data from file.
    
    Args:
        file_path: Path to data file
        text_column: Name of anchor text column
        text_pair_column: Name of positive text column
        file_format: File format (csv, json, jsonl)
        max_samples: Maximum samples to load
    
    Returns:
        List of InputExamples
    """
    file_path = Path(file_path)
    
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'json':
        df = pd.read_json(file_path)
    elif file_format == 'jsonl':
        df = pd.read_json(file_path, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    if max_samples is not None:
        df = df.head(max_samples)
    
    examples = []
    for idx, row in df.iterrows():
        text_b = None
        if text_pair_column and text_pair_column in row:
            text_b = str(row[text_pair_column])
        
        examples.append(InputExample(
            guid=f"train-{idx}",
            text_a=str(row[text_column]),
            text_b=text_b,
        ))
    
    return examples


def load_ner_data(
    file_path: str,
    file_format: str = 'conll',
    max_samples: Optional[int] = None,
) -> List[InputExample]:
    """
    Load NER data from file.
    
    Args:
        file_path: Path to data file
        file_format: File format (conll, json, jsonl)
        max_samples: Maximum samples to load
    
    Returns:
        List of InputExamples
    """
    file_path = Path(file_path)
    examples = []
    
    if file_format == 'conll':
        # CoNLL format: token tag (space/tab separated)
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = []
            tags = []
            
            for line in f:
                line = line.strip()
                
                if not line:
                    # Empty line marks end of sequence
                    if tokens:
                        examples.append(InputExample(
                            guid=f"train-{len(examples)}",
                            text_a=' '.join(tokens),
                            tokens=tokens,
                            tags=tags,
                        ))
                        tokens = []
                        tags = []
                    
                    if max_samples and len(examples) >= max_samples:
                        break
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        tokens.append(parts[0])
                        tags.append(parts[1])
            
            # Add last example if exists
            if tokens:
                examples.append(InputExample(
                    guid=f"train-{len(examples)}",
                    text_a=' '.join(tokens),
                    tokens=tokens,
                    tags=tags,
                ))
    
    elif file_format in ['json', 'jsonl']:
        # JSON format: {"tokens": [...], "tags": [...]}
        if file_format == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        
        if max_samples is not None:
            data = data[:max_samples]
        
        for idx, item in enumerate(data):
            tokens = item['tokens']
            tags = item['tags']
            examples.append(InputExample(
                guid=f"train-{idx}",
                text_a=' '.join(tokens),
                tokens=tokens,
                tags=tags,
            ))
    
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return examples


def get_labels(examples: List[InputExample]) -> List[str]:
    """Extract unique labels from examples."""
    labels = set()
    for example in examples:
        if example.label is not None:
            if isinstance(example.label, list):
                labels.update(example.label)
            else:
                labels.add(example.label)
    return sorted(list(labels))


def get_tag_labels(examples: List[InputExample]) -> List[str]:
    """Extract unique NER tags from examples."""
    tags = set()
    for example in examples:
        if example.tags is not None:
            tags.update(example.tags)
    return sorted(list(tags))


def create_label_mapping(labels: List[str]) -> Dict[str, int]:
    """Create label to ID mapping."""
    return {label: idx for idx, label in enumerate(labels)}
