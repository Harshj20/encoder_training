"""Inference CLI for batch processing and single-sample prediction."""

import argparse
import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm
import csv

from .model_wrapper import EncoderClassifier
from .utils import setup_logging, get_device


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> EncoderClassifier:
    """Load model from checkpoint directory."""
    checkpoint_path = Path(checkpoint_path)
    
    # Load model config
    config_file = checkpoint_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            model_config = json.load(f)
    else:
        model_config = {}
    
    # Load model
    model = EncoderClassifier(
        model_name_or_path=str(checkpoint_path / "encoder"),
        pooling_config=model_config.get("pooling", {"mode": "mean"}),
        num_labels=model_config.get("num_labels", 2),
    )
    
    # Load head weights
    head_file = checkpoint_path / "model_head.pt"
    if head_file.exists():
        head_state = torch.load(head_file, map_location=device)
        model.pooling.load_state_dict(head_state["pooling_state_dict"])
        model.classifier_head.load_state_dict(head_state["classifier_state_dict"])
    
    model.to(device)
    model.eval()
    
    return model


def batch_inference(
    model: EncoderClassifier,
    input_file: str,
    output_file: str,
    mode: str = "classify",
    batch_size: int = 64,
    text_column: str = "text",
    device: torch.device = None,
):
    """
    Run batch inference on input file.
    
    Args:
        model: Loaded model
        input_file: Input CSV or JSONL file
        output_file: Output JSONL file
        mode: "classify" or "embed"
        batch_size: Batch size for inference
        text_column: Name of text column
        device: Device to run on
    """
    if device is None:
        device = get_device()
    
    # Read input file
    input_path = Path(input_file)
    if input_path.suffix == '.csv':
        with open(input_path) as f:
            reader = csv.DictReader(f)
            data = list(reader)
    elif input_path.suffix in ['.json', '.jsonl']:
        with open(input_path) as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Extract texts
    texts = [item[text_column] for item in data]
    
    # Open output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Process in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
            batch_texts = texts[i:i+batch_size]
            batch_data = data[i:i+batch_size]
            
            # Tokenize
            encoded = model.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )
            
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            
            if mode == "classify":
                # Classification mode
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    mode="classification"
                )
                logits = outputs["logits"]
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                # Add results
                for j, (text, pred, prob) in enumerate(zip(batch_texts, predictions, probs)):
                    result = {
                        "input": text,
                        "prediction": int(pred),
                        "score": float(prob[pred]),
                        "all_scores": prob.cpu().tolist(),
                    }
                    # Include original fields
                    result.update({k: v for k, v in batch_data[j].items() if k != text_column})
                    results.append(result)
            
            elif mode == "embed":
                # Embedding mode
                embeddings = model.encode(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Add results
                for j, (text, emb) in enumerate(zip(batch_texts, embeddings)):
                    result = {
                        "input": text,
                        "embedding": emb.cpu().tolist(),
                    }
                    # Include original fields
                    result.update({k: v for k, v in batch_data[j].items() if k != text_column})
                    results.append(result)
    
    # Write output
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logging.info(f"Wrote {len(results)} predictions to {output_path}")


def predict_one(
    model: EncoderClassifier,
    text: str,
    mode: str = "classify",
    device: torch.device = None,
):
    """
    Predict single text.
    
    Args:
        model: Loaded model
        text: Input text
        mode: "classify" or "embed"
        device: Device to run on
    """
    if device is None:
        device = get_device()
    
    with torch.no_grad():
        # Tokenize
        encoded = model.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        
        if mode == "classify":
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode="classification"
            )
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1)
            
            print(f"\nInput: {text}")
            print(f"Prediction: {int(prediction[0])}")
            print(f"Confidence: {float(probs[0][prediction[0]]):.4f}")
            print(f"All scores: {probs[0].cpu().tolist()}")
        
        elif mode == "embed":
            embeddings = model.encode(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            print(f"\nInput: {text}")
            print(f"Embedding shape: {embeddings.shape}")
            print(f"Embedding (first 10 dims): {embeddings[0][:10].cpu().tolist()}")
            print(f"L2 norm: {torch.norm(embeddings[0]).item():.4f}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["classify", "embed"],
        default="classify",
        help="Inference mode"
    )
    
    # Batch mode args
    parser.add_argument(
        "--input",
        type=str,
        help="Input file (CSV or JSONL) for batch inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSONL file for batch inference"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in input file"
    )
    
    # Single sample mode
    parser.add_argument(
        "--text",
        type=str,
        help="Single text input for prediction"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    device = get_device()
    
    # Load model
    logging.info(f"Loading model from {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)
    logging.info("Model loaded successfully")
    
    # Run inference
    if args.text:
        # Single sample mode
        predict_one(model, args.text, args.mode, device)
    elif args.input and args.output:
        # Batch mode
        batch_inference(
            model=model,
            input_file=args.input,
            output_file=args.output,
            mode=args.mode,
            batch_size=args.batch_size,
            text_column=args.text_column,
            device=device,
        )
    else:
        parser.error("Either --text or both --input and --output must be specified")


if __name__ == "__main__":
    main()
