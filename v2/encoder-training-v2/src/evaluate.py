"""Evaluation CLI with comprehensive metrics."""

import argparse
import json
import logging
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from .model_wrapper import EncoderClassifier
from .data import create_dataloader
from .config import Config
from .utils import setup_logging, get_device


def evaluate_classification(
    model: EncoderClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate classification model.
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode="classification"
            )
            
            predictions = torch.argmax(outputs["logits"], dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = (
        precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
        "per_class_metrics": {
            "precision": precision_per_class.tolist(),
            "recall": recall_per_class.tolist(),
            "f1": f1_per_class.tolist(),
            "support": support.tolist(),
        },
        "confusion_matrix": conf_matrix.tolist(),
    }
    
    return metrics


def evaluate_embeddings(
    model: EncoderClassifier,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate embedding quality (placeholder).
    
    TODO: Implement retrieval metrics (MRR, MAP, etc.)
    """
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            embeddings = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            all_embeddings.append(embeddings.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Basic statistics
    metrics = {
        "num_samples": len(all_embeddings),
        "embedding_dim": all_embeddings.shape[1],
        "mean_l2_norm": float(torch.norm(all_embeddings, p=2, dim=1).mean()),
        "std_l2_norm": float(torch.norm(all_embeddings, p=2, dim=1).std()),
    }
    
    # TODO: Implement MRR, MAP, cosine similarity metrics for pairs/triplets
    logging.warning("Full embedding evaluation metrics not yet implemented")
    
    return metrics


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Override dataset path from config"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "pair", "triplet"],
        help="Override data mode from config"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for metrics"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    device = get_device()
    
    # Load config
    config = Config.from_yaml(args.config)
    
    # Override args
    if args.dataset:
        config.data.dataset = args.dataset
    if args.mode:
        config.data.type = args.mode
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    checkpoint_path = Path(args.checkpoint)
    
    # Load model config
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            model_config = json.load(f)
    else:
        model_config = {}
    
    model = EncoderClassifier(
        model_name_or_path=str(checkpoint_path / "encoder"),
        pooling_config=model_config.get("pooling", config.model.pooling.__dict__),
        num_labels=config.model.num_labels,
    )
    
    # Load head weights
    head_file = checkpoint_path / "model_head.pt"
    if head_file.exists():
        head_state = torch.load(head_file, map_location=device)
        model.pooling.load_state_dict(head_state["pooling_state_dict"])
        model.classifier_head.load_state_dict(head_state["classifier_state_dict"])
    
    model.to(device)
    model.eval()
    
    logging.info("Model loaded successfully")
    
    # Create dataloader
    dataloader = create_dataloader(
        dataset_path=config.data.dataset,
        tokenizer=model.tokenizer,
        data_config=config.data.__dict__,
        training_config=config.training.__dict__,
        split=args.split,
    )
    
    # Evaluate
    logging.info(f"Evaluating on {args.split} split...")
    
    if config.data.type in ["single", "pair"]:
        metrics = evaluate_classification(model, dataloader, device)
    elif config.data.type == "triplet":
        metrics = evaluate_embeddings(model, dataloader, device)
    else:
        raise ValueError(f"Unknown data type: {config.data.type}")
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(json.dumps(metrics, indent=2))
    print("="*50)
    
    # Save to file if specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logging.info(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
