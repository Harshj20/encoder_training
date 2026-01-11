import os
import sys
import logging
from transformers import HfArgumentParser, AutoTokenizer

# Add src to path to allow importing encoder_trainer without install
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from encoder_trainer.config import ModelConfig, DataConfig, TrainerConfig
from encoder_trainer.data.dataset import UnifiedDataset
from encoder_trainer.data.collators import DynamicLabelCollator
from encoder_trainer.modeling.encoder import UnifiedEncoderClassifier
from encoder_trainer.trainer import EncoderTrainer
from encoder_trainer.metrics import compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelConfig, DataConfig, TrainerConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model Args: {model_args}")
    logger.info(f"Data Args: {data_args}")
    logger.info(f"Training Args: {training_args}")

    # 1. Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    
    # 2. Load Dataset
    train_dataset = UnifiedDataset(data_args, split="train")
    if data_args.validation_split > 0:
        # Simple split logic handling needed if dataset doesn't have split
        # For this script, we assume UnifiedDataset loads 'train' by default
        # If we used HF datasets, we could split. 
        # Here we just reuse same dataset for demo or rely on HF loading valid split if available
        # To make it robust:
        try:
             eval_dataset = UnifiedDataset(data_args, split="validation")
        except:
             logger.warning("No validation split found, using subset of train or skipping.")
             eval_dataset = None
    else:
        eval_dataset = None

    # Determine num_labels
    num_labels = 0
    if train_dataset.label_manager:
        num_labels = len(train_dataset.label_manager)
        logger.info(f"Detected {num_labels} fixed labels: {train_dataset.label_manager.labels}")

    # 3. Load Model
    model = UnifiedEncoderClassifier(model_args, num_labels=num_labels)
    
    # 4. Data Collator
    collator = DynamicLabelCollator(
        tokenizer=tokenizer, 
        max_length=data_args.max_length, 
        add_candidates=(model_args.mode == "sce")
    )
    
    # 5. Initialize Trainer
    trainer = EncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    
    # 6. Train
    logger.info("Starting training...")
    trainer.train()
    
    # 7. Save
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()
