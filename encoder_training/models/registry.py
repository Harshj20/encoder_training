"""Model registry for dynamic model loading."""

from typing import Dict, Type, Optional
from .base import BaseModelTrainer


class ModelRegistry:
    """Registry for managing available model trainers."""
    
    _models: Dict[str, Type[BaseModelTrainer]] = {}
    
    @classmethod
    def register(cls, model_name: str, model_class: Type[BaseModelTrainer]):
        """
        Register a model trainer.
        
        Args:
            model_name: Unique identifier for the model
            model_class: Model trainer class
        """
        cls._models[model_name] = model_class
    
    @classmethod
    def get(cls, model_name: str) -> Optional[Type[BaseModelTrainer]]:
        """
        Get a registered model trainer class.
        
        Args:
            model_name: Model identifier
        
        Returns:
            Model trainer class or None if not found
        """
        return cls._models.get(model_name)
    
    @classmethod
    def list_models(cls) -> list:
        """Get list of registered model names."""
        return list(cls._models.keys())
    
    @classmethod
    def is_registered(cls, model_name: str) -> bool:
        """Check if a model is registered."""
        return model_name in cls._models


def register_model(model_name: str):
    """
    Decorator to register a model trainer.
    
    Usage:
        @register_model("my-model")
        class MyModelTrainer(BaseModelTrainer):
            ...
    """
    def decorator(cls: Type[BaseModelTrainer]):
        ModelRegistry.register(model_name, cls)
        return cls
    return decorator


def get_model(model_name: str, **kwargs) -> BaseModelTrainer:
    """
    Get an instance of a registered model trainer.
    
    Args:
        model_name: Model identifier
        **kwargs: Arguments to pass to model trainer constructor
    
    Returns:
        Model trainer instance
    
    Raises:
        ValueError: If model is not registered
    """
    model_class = ModelRegistry.get(model_name)
    
    if model_class is None:
        available_models = ModelRegistry.list_models()
        raise ValueError(
            f"Model '{model_name}' is not registered. "
            f"Available models: {available_models}"
        )
    
    return model_class(**kwargs)
