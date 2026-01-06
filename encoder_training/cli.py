"""Command-line interface for encoder training."""

import argparse
import sys
from pathlib import Path
import yaml
import logging

from encoder_training.core.config import FullConfig
from encoder_training.models import get_model, ModelRegistry


def train_command(args):
    """Execute training command."""
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = FullConfig.from_dict(config_dict)
    
    # Override with CLI arguments if provided
    if args.model:
        config.model.pretrained_path = args.model
    if args.output_dir:
        config.training.output_dir = args.output_dir
    
    # Get model trainer
    trainer = get_model(
        model_name=config.model.name,
        pretrained_path=config.model.pretrained_path
    )
    
    # Train based on task
    task = args.task.lower()
    if task == "classification":
        results = trainer.train_classification(config)
    elif task == "embedding":
        results = trainer.train_embedding(config)
    elif task == "ner":
        results = trainer.train_ner(config)
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    print(f"\nTraining completed!")
    print(f"Results: {results}")


def infer_command(args):
    """Execute inference command."""
    trainer = get_model(
        model_name=args.model,
        pretrained_path=args.checkpoint
    )
    
    # Run inference
    results = trainer.inference(
        task=args.task,
        inputs=args.input,
        checkpoint_path=args.checkpoint
    )
    
    print(f"Predictions: {results}")


def eval_command(args):
    """Execute evaluation command."""
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    config = FullConfig.from_dict(config_dict)
    
    trainer = get_model(
        model_name=config.model.name,
        pretrained_path=args.checkpoint
    )
    
    # Evaluate
    metrics = trainer.evaluate(
        task=args.task,
        test_data_path=args.test_data,
        checkpoint_path=args.checkpoint,
        config=config
    )
    
    print(f"\nEvaluation Results:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


def list_models_command(args):
    """List available models."""
    models = ModelRegistry.list_models()
    print("Available models:")
    for model in models:
        print(f"  - {model}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Encoder Training Module - Train encoder models for various NLP tasks"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to configuration YAML file')
    train_parser.add_argument('--task', type=str, required=True,
                             choices=['classification', 'embedding', 'ner'],
                             help='Task type')
    train_parser.add_argument('--model', type=str,
                             help='Model path/ID (overrides config)')
    train_parser.add_argument('--output-dir', type=str,
                             help='Output directory (overrides config)')
    train_parser.set_defaults(func=train_command)
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', type=str, required=True,
                             help='Model name')
    infer_parser.add_argument('--task', type=str, required=True,
                             choices=['classification', 'embedding', 'ner'],
                             help='Task type')
    infer_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Path to model checkpoint')
    infer_parser.add_argument('--input', type=str, required=True,
                             help='Input text')
    infer_parser.set_defaults(func=infer_command)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a model')
    eval_parser.add_argument('--config', type=str, required=True,
                            help='Path to configuration YAML file')
    eval_parser.add_argument('--task', type=str, required=True,
                            choices=['classification', 'embedding', 'ner'],
                            help='Task type')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--test-data', type=str, required=True,
                            help='Path to test data')
    eval_parser.set_defaults(func=eval_command)
    
    # List models command
    list_parser = subparsers.add_parser('list-models', help='List available models')
    list_parser.set_defaults(func=list_models_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
