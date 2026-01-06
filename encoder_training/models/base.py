"""Base model trainer interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseModelTrainer(ABC):
    """
    Abstract base class for model trainers.
    
    All model-specific trainers should inherit from this class and implement
    the abstract methods for each supported task.
    """
    
    def __init__(
        self,
        model_name: str,
        pretrained_path: str,
        device: Optional[torch.device] = None
    ):
        """
        Initialize base trainer.
        
        Args:
            model_name: Name/identifier of the model
            pretrained_path: Path or HuggingFace model ID
            device: Device to use for training
        """
        self.model_name = model_name
        self.pretrained_path = pretrained_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
    
    @abstractmethod
    def train_classification(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model for text classification task.
        
        Args:
            config: Training configuration
        
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def train_embedding(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model for text embedding task.
        
        Args:
            config: Training configuration
        
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def train_ner(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train model for NER task.
        
        Args:
            config: Training configuration
        
        Returns:
            Dictionary containing training results and metrics
        """
        pass
    
    @abstractmethod
    def inference(
        self,
        task: str,
        inputs: Any,
        checkpoint_path: Optional[str] = None
    ) -> Any:
        """
        Run inference for a specific task.
        
        Args:
            task: Task type (classification, embedding, ner)
            inputs: Input data
            checkpoint_path: Path to model checkpoint
        
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        task: str,
        test_data_path: str,
        checkpoint_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            task: Task type (classification, embedding, ner)
            test_data_path: Path to test data
            checkpoint_path: Path to model checkpoint
            config: Evaluation configuration
        
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    def load_model(self) -> PreTrainedModel:
        """Load pretrained model."""
        raise NotImplementedError("Subclasses should implement load_model")
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer."""
        raise NotImplementedError("Subclasses should implement load_tokenizer")
    
    def save_model(self, output_dir: str):
        """Save model and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            self.model.save_pretrained(output_dir)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
