"""Common utilities for training."""

import logging
import random
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (if None, only console logging)
        level: Logging level
        format_string: Custom format string
    
    Returns:
        Logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logger
    logger = logging.getLogger("encoder_training")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(format_string))
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """
    Get torch device.
    
    Args:
        device: Device string ("cuda", "cpu", "auto")
    
    Returns:
        torch.device
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    save_path: str,
    config: Optional[Dict[str, Any]] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        metrics: Evaluation metrics
        save_path: Path to save checkpoint
        config: Configuration dictionary
        scheduler: Learning rate scheduler
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
    }
    
    if config is not None:
        checkpoint['config'] = config
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    logging.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load model to
    
    Returns:
        Dictionary containing checkpoint metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Checkpoint loaded from {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint.get('epoch')}, Step: {checkpoint.get('step')}")
    logging.info(f"Metrics: {checkpoint.get('metrics')}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
    }


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ProgressTracker:
    """Track training progress and metrics."""
    
    def __init__(self):
        self.history = []
        self.best_metric = None
        self.best_epoch = None
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """Update with new metrics."""
        self.history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
        })
    
    def update_best(self, epoch: int, metric_value: float, greater_is_better: bool = True):
        """Update best metric."""
        if self.best_metric is None:
            self.best_metric = metric_value
            self.best_epoch = epoch
        else:
            if greater_is_better:
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self.best_epoch = epoch
            else:
                if metric_value < self.best_metric:
                    self.best_metric = metric_value
                    self.best_epoch = epoch
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training."""
        return {
            'total_epochs': len(self.history),
            'best_epoch': self.best_epoch,
            'best_metric': self.best_metric,
            'history': self.history,
        }
