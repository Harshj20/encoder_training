#!/bin/bash

# Example: Head-only finetuning
# This trains only the classification head while keeping the encoder frozen

python -m src.train \
    --config src/experiments/head_only.yaml

echo "Training completed! Model saved to ./outputs/head_only"
