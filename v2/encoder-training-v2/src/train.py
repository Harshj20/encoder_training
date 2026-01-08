"""Training CLI with support for head-only, full, and PEFT modes."""

import argparse
import logging
import torch
from pathlib import Path
from accelerate import Accelerator
from transformers import get_scheduler
from tqdm import tqdm

from .config import Config
from .model_wrapper import EncoderClassifier
from .data import create_dataloader
from .losses import get_loss_fn, classification_loss, contrastive_loss, triplet_loss
from .utils import (
    setup_logging,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    is_main_process,
    MetricsTracker,
    cleanup_checkpoints,
)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    accelerator,
    loss_fn,
    data_type,
    gradient_accumulation_steps,
    max_grad_norm,
    global_step,
    metrics_tracker,
    logging_steps,
):
    """
    Train for one epoch.
    
    Returns:
        Updated global_step
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(
        dataloader,
        desc="Training",
        disable=not is_main_process()
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device (accelerator handles this)
        with accelerator.accumulate(model):
            if data_type == "single" or data_type == "pair":
                # Classification mode
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                    mode="classification"
                )
                loss = outputs.get("loss")
                
                if loss is None:
                    # Compute loss manually
                    loss = classification_loss(outputs["logits"], batch["labels"])
            
            elif data_type == "triplet":
                # Triplet mode
                anchor_emb = model.encode(
                    input_ids=batch["anchor_input_ids"],
                    attention_mask=batch["anchor_attention_mask"]
                )
                positive_emb = model.encode(
                    input_ids=batch["positive_input_ids"],
                    attention_mask=batch["positive_attention_mask"]
                )
                negative_emb = model.encode(
                    input_ids=batch["negative_input_ids"],
                    attention_mask=batch["negative_attention_mask"]
                )
                loss = triplet_loss(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics
        total_loss += accelerator.gather(loss).mean().item()
        num_batches += 1
        
        if accelerator.sync_gradients:
            global_step += 1
            
            # Logging
            if global_step % logging_steps == 0:
                avg_loss = total_loss / num_batches
                metrics = {"loss": avg_loss, "lr": scheduler.get_last_lr()[0]}
                metrics_tracker.update(metrics, global_step)
                
                if is_main_process():
                    logging.info(
                        f"Step {global_step}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.6f}"
                    )
                
                progress_bar.set_postfix(loss=avg_loss)
    
    return global_step


def train(config: Config):
    """
    Main training function.
    
    Args:
        config: Training configuration
    """
    # Setup logging
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(
        log_file=str(output_dir / "train.log")
    )
    
    # Set seed
    set_seed(config.training.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision="fp16" if config.training.fp16 else ("bf16" if config.training.bf16 else "no"),
        log_with="tensorboard" if is_main_process() else None,
        project_dir=str(output_dir),
    )
    
    if is_main_process():
        logger.info("="*50)
        logger.info("Training Configuration")
        logger.info("="*50)
        logger.info(f"Model: {config.model.name_or_path}")
        logger.info(f"Training mode: {config.training.mode}")
        logger.info(f"Data type: {config.data.type}")
        logger.info(f"Loss type: {config.loss.type}")
        logger.info(f"Epochs: {config.training.epochs}")
        logger.info(f"Batch size: {config.training.per_device_train_batch_size}")
        logger.info(f"Gradient accumulation: {config.training.gradient_accumulation_steps}")
        logger.info(f"Learning rate: {config.training.lr}")
        logger.info(f"FP16: {config.training.fp16}")
        logger.info(f"Output: {output_dir}")
        logger.info("="*50)
    
    # Load model
    model = EncoderClassifier(
        model_name_or_path=config.model.name_or_path,
        tokenizer_name_or_path=config.model.tokenizer,
        pooling_config=config.model.pooling.__dict__,
        classifier_hidden=config.model.classifier_hidden,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
    )
    
    # Configure training mode
    if config.training.mode == "head_only":
        model.freeze_base()
        logger.info("Mode: Head-only training (encoder frozen)")
    elif config.training.mode == "full":
        model.unfreeze_all()
        logger.info("Mode: Full finetuning")
    elif config.training.mode == "peft":
        model.enable_peft(config.peft.__dict__)
        logger.info("Mode: PEFT (LoRA) training")
    
    # Log parameter counts
    param_counts = count_parameters(model)
    if is_main_process():
        logger.info(f"Total parameters: {param_counts['total']:,}")
        logger.info(f"Trainable parameters: {param_counts['trainable']:,} ({param_counts['trainable_percentage']:.2f}%)")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        dataset_path=config.data.dataset,
        tokenizer=model.tokenizer,
        data_config=config.data.__dict__,
        training_config=config.training.__dict__,
        split=config.data.split,
    )
    
    # TODO: Add validation dataloader if eval dataset specified
    # eval_dataloader = create_dataloader(..., split="validation")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    
    # Calculate total steps
    if config.training.max_steps:
        total_steps = config.training.max_steps
    else:
        # Estimate from epochs
        # Note: For streaming datasets, we may not know exact length
        total_steps = config.training.epochs * 1000  # Rough estimate
    
    # Setup scheduler
    num_warmup_steps = config.training.warmup_steps or int(total_steps * config.training.warmup_ratio)
    scheduler = get_scheduler(
        config.training.scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Prepare with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    if is_main_process():
        logger.info("Starting training...")
    
    for epoch in range(config.training.epochs):
        if is_main_process():
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{config.training.epochs}")
            logger.info(f"{'='*50}")
        
        global_step = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            loss_fn=None,  # Computed inline
            data_type=config.data.type,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            max_grad_norm=config.training.max_grad_norm,
            global_step=global_step,
            metrics_tracker=metrics_tracker,
            logging_steps=config.training.logging_steps,
        )
        
        # Save checkpoint
        if is_main_process() and (epoch + 1) % (config.training.save_steps // 1000) == 0:
            checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
            
            # Unwrap model for saving
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(str(checkpoint_dir))
            
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Cleanup old checkpoints
            cleanup_checkpoints(
                output_dir,
                save_total_limit=config.training.save_total_limit
            )
        
        # Check max steps
        if config.training.max_steps and global_step >= config.training.max_steps:
            if is_main_process():
                logger.info(f"Reached max_steps={config.training.max_steps}, stopping training")
            break
    
    # Save final model
    if is_main_process():
        final_dir = output_dir / "final_model"
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(str(final_dir))
        logger.info(f"Saved final model to {final_dir}")
        
        # Save metrics
        metrics_tracker.save(output_dir / "metrics.json")
        logger.info("Training completed!")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train encoder model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda, cpu, auto)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override with CLI args
    if args.resume_from_checkpoint:
        config.training.resume_from_checkpoint = args.resume_from_checkpoint
    
    # Validate config
    config.validate()
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
