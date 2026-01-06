"""Main trainer for KALM Embedding 2.5 Instruct model."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from pathlib import Path
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    get_scheduler,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from ..base import BaseModelTrainer
from ..registry import register_model
from ...core.config import FullConfig
from ...core.data_loader import (
    load_classification_data,
    load_embedding_data,
    load_ner_data,
    ClassificationDataset,
    EmbeddingDataset,
    NERDataset,
    get_labels,
    get_tag_labels,
    create_label_mapping,
)
from ...core.metrics import ClassificationMetrics, EmbeddingMetrics, NERMetrics
from ...core.utils import (
    setup_logging,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    AverageMeter,
    ProgressTracker,
)


@register_model("kalm-embedding-2.5")
class KalmEmbeddingTrainer(BaseModelTrainer):
    """Trainer for KALM Embedding 2.5 Instruct model."""
    
    def __init__(
        self,
        model_name: str = "kalm-embedding-2.5",
        pretrained_path: str = "Kalm/kalm-embedding-2.5-instruct",
        device: Optional[torch.device] = None
    ):
        super().__init__(model_name, pretrained_path, device)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, task: str = "classification", num_labels: Optional[int] = None):
        """Load model for specific task."""
        if task == "classification":
            if num_labels is None:
                raise ValueError("num_labels is required for classification task")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.pretrained_path,
                num_labels=num_labels
            )
        elif task == "embedding":
            self.model = AutoModel.from_pretrained(self.pretrained_path)
        elif task == "ner":
            if num_labels is None:
                raise ValueError("num_labels is required for NER task")
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.pretrained_path,
                num_labels=num_labels
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        self.model.to(self.device)
        return self.model
    
    def load_tokenizer(self):
        """Load tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)
        return self.tokenizer
    
    def train_classification(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model for text classification."""
        # Parse config
        if isinstance(config, dict):
            config = FullConfig.from_dict(config)
        
        # Setup
        set_seed(config.seed)
        setup_logging(
            log_file=str(Path(config.training.output_dir) / "train.log")
        )
        
        self.logger.info("Starting classification training...")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Pretrained path: {self.pretrained_path}")
        
        # Load data
        self.logger.info("Loading data...")
        train_examples = load_classification_data(
            config.data.train_file,
            text_column=config.data.text_column,
            label_column=config.data.label_column,
            file_format=config.data.data_format,
            max_samples=config.data.max_samples,
        )
        
        val_examples = load_classification_data(
            config.data.val_file,
            text_column=config.data.text_column,
            label_column=config.data.label_column,
            file_format=config.data.data_format,
        ) if config.data.val_file else None
        
        # Get labels
        labels = get_labels(train_examples)
        label_to_id = create_label_mapping(labels)
        id_to_label = {v: k for k, v in label_to_id.items()}
        config.model.num_labels = len(labels)
        
        self.logger.info(f"Number of labels: {len(labels)}")
        self.logger.info(f"Labels: {labels}")
        
        # Load model and tokenizer
        self.load_tokenizer()
        self.load_model(task="classification", num_labels=len(labels))
        
        # Log parameter count
        param_counts = count_parameters(self.model)
        self.logger.info(f"Total parameters: {param_counts['total']:,}")
        self.logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
        
        # Create datasets
        train_dataset = ClassificationDataset(
            train_examples,
            self.tokenizer,
            max_length=config.model.max_seq_length,
            label_to_id=label_to_id,
        )
        
        val_dataset = None
        if val_examples:
            val_dataset = ClassificationDataset(
                val_examples,
                self.tokenizer,
                max_length=config.model.max_seq_length,
                label_to_id=label_to_id,
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.dataloader_num_workers,
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=config.training.dataloader_num_workers,
            )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        total_steps = len(train_loader) * config.training.num_epochs
        scheduler = get_scheduler(
            config.training.scheduler,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * config.training.warmup_ratio),
            num_training_steps=total_steps,
        )
        
        # Training loop
        best_metric = float('-inf') if config.evaluation.greater_is_better else float('inf')
        progress_tracker = ProgressTracker()
        global_step = 0
        
        for epoch in range(config.training.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = AverageMeter()
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss = loss / config.training.gradient_accumulation_steps
                loss.backward()
                
                if (global_step + 1) % config.training.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        config.training.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss.update(loss.item() * config.training.gradient_accumulation_steps)
                progress_bar.set_postfix({'loss': train_loss.avg})
                
                global_step += 1
            
            self.logger.info(f"Train loss: {train_loss.avg:.4f}")
            
            # Validation
            if val_loader:
                val_metrics = self._evaluate_classification(val_loader, id_to_label)
                self.logger.info(f"Validation metrics: {val_metrics}")
                
                # Track progress
                progress_tracker.update(epoch, val_metrics)
                
                # Save best model
                metric_value = val_metrics[config.evaluation.metric_for_best_model]
                is_best = (
                    (config.evaluation.greater_is_better and metric_value > best_metric) or
                    (not config.evaluation.greater_is_better and metric_value < best_metric)
                )
                
                if is_best:
                    best_metric = metric_value
                best_model_path = Path(config.training.output_dir) / "best_model"
                    save_checkpoint(
                        self.model,
                        optimizer,
                        epoch,
                        global_step,
                        val_metrics,
                        best_model_path / "checkpoint.pt",
                        config=config.to_dict(),
                        scheduler=scheduler,
                    )
                    self.save_model(str(best_model_path))
                    self.logger.info(f"Saved best model to {best_model_path}")
            
            # Save checkpoint
            if (epoch + 1) % config.evaluation.save_steps == 0:
                checkpoint_path = Path(config.training.output_dir) / f"checkpoint-epoch-{epoch+1}"
                save_checkpoint(
                    self.model,
                    optimizer,
                    epoch,
                    global_step,
                    val_metrics if val_loader else {},
                    checkpoint_path / "checkpoint.pt",
                    config=config.to_dict(),
                    scheduler=scheduler,
                )
                self.save_model(str(checkpoint_path))
        
        return {
            'best_metric': best_metric,
            'history': progress_tracker.get_summary(),
        }
    
    def _evaluate_classification(
        self,
        data_loader: DataLoader,
        id_to_label: Dict[int, str]
    ) -> Dict[str, float]:
        """Evaluate classification model."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        metrics = ClassificationMetrics.compute(
            predictions=np.array(all_predictions),
            labels=np.array(all_labels),
        )
        
        return metrics
    
    def train_embedding(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model for text embedding (contrastive learning)."""
        # Parse config
        if isinstance(config, dict):
            config = FullConfig.from_dict(config)
        
        # Setup
        set_seed(config.seed)
        setup_logging(
            log_file=str(Path(config.training.output_dir) / "train.log")
        )
        
        self.logger.info("Starting embedding training...")
        
        # Load data
        train_examples = load_embedding_data(
            config.data.train_file,
            text_column=config.data.text_column,
            text_pair_column=config.data.text_pair_column,
            file_format=config.data.data_format,
            max_samples=config.data.max_samples,
        )
        
        # Load model and tokenizer
        self.load_tokenizer()
        self.load_model(task="embedding")
        
        # Create dataset
        train_dataset = EmbeddingDataset(
            train_examples,
            self.tokenizer,
            max_length=config.model.max_seq_length,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.dataloader_num_workers,
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        total_steps = len(train_loader) * config.training.num_epochs
        scheduler = get_scheduler(
            config.training.scheduler,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * config.training.warmup_ratio),
            num_training_steps=total_steps,
        )
        
        # Contrastive loss
        loss_fn = self._get_contrastive_loss(config.training.contrastive_loss)
        
        # Training loop
        for epoch in range(config.training.num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{config.training.num_epochs}")
            
            self.model.train()
            train_loss = AverageMeter()
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                # Get embeddings for anchor and positive
anchor_outputs = self.model(
                    input_ids=batch['anchor_input_ids'].to(self.device),
                    attention_mask=batch['anchor_attention_mask'].to(self.device),
                )
                anchor_embeddings = self._mean_pooling(
                    anchor_outputs.last_hidden_state,
                    batch['anchor_attention_mask'].to(self.device)
                )
                
                if 'positive_input_ids' in batch:
                    positive_outputs = self.model(
                        input_ids=batch['positive_input_ids'].to(self.device),
                        attention_mask=batch['positive_attention_mask'].to(self.device),
                    )
                    positive_embeddings = self._mean_pooling(
                        positive_outputs.last_hidden_state,
                        batch['positive_attention_mask'].to(self.device)
                    )
                    
                    # Calculate loss
                    loss = loss_fn(anchor_embeddings, positive_embeddings, config.training.temperature)
                else:
                    # In-batch negatives only
                    loss = self._infonce_loss(anchor_embeddings, config.training.temperature)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.training.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                train_loss.update(loss.item())
                progress_bar.set_postfix({'loss': train_loss.avg})
            
            self.logger.info(f"Train loss: {train_loss.avg:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % config.evaluation.save_steps == 0:
                checkpoint_path = Path(config.training.output_dir) / f"checkpoint-epoch-{epoch+1}"
                self.save_model(str(checkpoint_path))
        
        final_path = Path(config.training.output_dir) / "final_model"
        self.save_model(str(final_path))
        
        return {'final_model_path': str(final_path)}
    
    def _mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling for embeddings."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def _infonce_loss(self, embeddings, temperature):
        """InfoNCE loss for in-batch negatives."""
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature
        
        # Labels: positive pairs are on the diagonal
        batch_size = embeddings.shape[0]
        labels = torch.arange(batch_size).to(embeddings.device)
        
        # Cross-entropy loss
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        return loss
    
    def _get_contrastive_loss(self, loss_type):
        """Get contrastive loss function."""
        if loss_type == "infonce":
            return lambda anc, pos, temp: self._infonce_loss(torch.cat([anc, pos], dim=0), temp)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def train_ner(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model for NER."""
        # Implementation similar to classification but with NER-specific handling
        # Abbreviated for brevity - full implementation would follow same pattern
        raise NotImplementedError("NER training will be implemented in next iteration")
    
    def inference(
        self,
        task: str,
        inputs: Any,
        checkpoint_path: Optional[str] = None
    ) -> Any:
        """Run inference."""
        if checkpoint_path:
            # Load from checkpoint
            pass
        
        self.model.eval()
        # Implementation for inference
        raise NotImplementedError("Inference will be implemented in next iteration")
    
    def evaluate(
        self,
        task: str,
        test_data_path: str,
        checkpoint_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, float]:
        """Evaluate model."""
        raise NotImplementedError("Evaluation will be implemented in next iteration")


import numpy as np
