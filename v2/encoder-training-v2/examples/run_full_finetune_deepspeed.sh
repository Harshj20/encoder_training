#!/bin/bash

# Example: Full finetuning with DeepSpeed
# Uses DeepSpeed ZeRO stage 2 for memory efficiency on multi-GPU setups

# Note: Requires accelerate config or use torchrun/deepspeed launcher

# Option 1: Using accelerate
accelerate launch --config_file accelerate_config.yaml src/train.py \
    --config src/experiments/full_finetune_deepspeed.yaml

# Option 2: Using torchrun (PyTorch distributed)
# torchrun --nproc_per_node=2 src/train.py --config src/experiments/full_finetune_deepspeed.yaml

echo "Training completed! Model saved to ./outputs/full_deepspeed"
