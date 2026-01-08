"""Utility functions for checkpointing, logging, metrics, and distributed training."""

import os
import random
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Optional log file path
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger("encoder_training")
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for HF datasets
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device.
    
    Args:
        device_str: Device string (cuda, cpu, auto)
    
    Returns:
        torch.device
    """
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    save_path: Path,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        global_step: Global training step
        save_path: Path to save checkpoint
        config: Training configuration
        metrics: Current metrics
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'timestamp': datetime.now().isoformat(),
    }
    
    if config:
        checkpoint['config'] = config
    
    if metrics:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path / "checkpoint.pt")
    
    # Save config separately for easy access
    if config:
        with open(save_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    logging.info(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Device to load model to
    
    Returns:
        Checkpoint metadata dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_file = Path(checkpoint_path) / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    logging.info(f"Epoch: {checkpoint.get('epoch')}, Step: {checkpoint.get('global_step')}")
    
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dict with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params,
        'trainable_percentage': 100 * trainable_params / total_params if total_params > 0 else 0,
    }


def get_rank() -> int:
    """Get distributed training rank."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is main process in distributed training."""
    return get_rank() == 0


class MetricsTracker:
    """Track and aggregate metrics during training."""
    
    def __init__(self):
        self.history = []
        self.current_metrics = {}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics for current step."""
        self.current_metrics = metrics.copy()
        self.current_metrics['step'] = step
        self.history.append(self.current_metrics.copy())
    
    def update_best(self, metric_name: str, metric_value: float, greater_is_better: bool = True):
        """Update best metric value."""
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = metric_value
        else:
            if greater_is_better:
                if metric_value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
            else:
                if metric_value < self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'total_steps': len(self.history),
            'current_metrics': self.current_metrics,
            'best_metrics': self.best_metrics,
            'history': self.history,
        }
    
    def save(self, path: Path):
        """Save metrics to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


def cleanup_checkpoints(
    output_dir: Path,
    save_total_limit: int = 3,
    best_checkpoint_dir: Optional[Path] = None
):
    """
    Clean up old checkpoints, keeping only the most recent ones.
    
    Args:
        output_dir: Directory containing checkpoints
        save_total_limit: Maximum number of checkpoints to keep
        best_checkpoint_dir: Path to best checkpoint (never deleted)
    """
    output_dir = Path(output_dir)
    checkpoints = sorted(
        [d for d in output_dir.glob("checkpoint-*") if d.is_dir()],
        key=lambda x: x.stat().st_mtime
    )
    
    # Don't delete best checkpoint
    if best_checkpoint_dir:
        best_checkpoint_dir = Path(best_checkpoint_dir)
        checkpoints = [c for c in checkpoints if c != best_checkpoint_dir]
    
    # Delete old checkpoints
    if len(checkpoints) > save_total_limit:
        for checkpoint in checkpoints[:-save_total_limit]:
            logging.info(f"Deleting old checkpoint: {checkpoint}")
            import shutil
            shutil.rmtree(checkpoint)
