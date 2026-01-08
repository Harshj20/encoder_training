# Encoder Training Framework v2.0

Production-ready training/inference/evaluation module for text classification using encoder-only embedding models (~1B parameters). Features memory-efficient streaming, multi-GPU support, and PEFT integration.

## Features

- ✅ **Three Training Modes**: `head_only`, `full`, `peft` (LoRA)
- ✅ **Memory-Efficient Streaming**: Never load full dataset into RAM
- ✅ **Multi-GPU/Multi-Node**: Via `accelerate` + optional DeepSpeed
- ✅ **Multiple Pooling Strategies**: mean, cls, max, lasttoken, weighted
- ✅ **Three Data Types**: single-text, pair, triplet
- ✅ **Production Defaults**: Conservative batch sizes + gradient accumulation for ~1B models
- ✅ **Comprehensive Testing**: pytest suite + CI/CD

## Quick Start

### Installation

```bash
# Clone repository
cd encoder-training-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

**1. Train (Head-Only)**
```bash
python -m src.train --config src/experiments/head_only.yaml
```

**2. Train (PEFT/LoRA)**
```bash
python -m src.train --config src/experiments/peft_lora.yaml
```

**3. Train (Full Finetuning with DeepSpeed)**
```bash
accelerate launch src/train.py --config src/experiments/full_finetune_deepspeed.yaml
```

**4. Inference (Batch)**
```bash
python -m src.infer \
    --checkpoint outputs/head_only/final_model \
    --input data/test.csv \
    --output predictions.jsonl \
    --mode classify \
    --batch_size 64
```

**5. Inference (Single Sample)**
```bash
python -m src.infer \
    --checkpoint outputs/head_only/final_model \
    --mode classify \
    --text "This is a test sentence"
```

**6. Evaluation**
```bash
python -m src.evaluate \
    --checkpoint outputs/head_only/final_model \
    --config src/experiments/head_only.yaml \
    --split test
```

## Configuration

### Example Config (Head-Only Mode)

```yaml
model:
  name_or_path: "intfloat/multilingual-e5-small"
  pooling:
    mode: "mean"  # mean, cls, max, lasttoken, weighted
    layers: [-1]
    normalize: true
  num_labels: 2

training:
  mode: "head_only"  # head_only, full, peft
  epochs: 3
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  fp16: true
  lr: 1e-3train:
  output_dir: "./outputs/head_only"

data:
  type: "single"  # single, pair, triplet
  dataset: "path/to/data.csv"
  text_field: "text"
  label_field: "label"
```

See `src/experiments/` for complete examples.

## Project Structure

```
encoder-training-v2/
├── src/
│   ├── train.py              # Training CLI
│   ├── infer.py              # Inference CLI
│   ├── evaluate.py           # Evaluation CLI
│   ├── model_wrapper.py      # EncoderClassifier with pooling & PEFT
│   ├── data.py               # Streaming data pipeline
│   ├── losses.py             # Loss functions
│   ├── config.py             # Configuration management
│   ├── utils.py              # Utilities
│   └── experiments/          # YAML configs
├── tests/                    # pytest test suite
├── examples/                 # Example scripts & datasets
├── docker/                   # Dockerfile & compose
└── requirements.txt
```

## Training Modes

### 1. Head-Only (`mode: head_only`)
- **Use Case**: Quick adaptation with limited compute
- **Memory**: ~4-8GB VRAM for ~1B models
- **Speed**: Fastest (only head parameters trained)
- **Typical LR**: 1e-3 to 5e-3
- **Trainable Params**: <1% of total

### 2. Full Finetuning (`mode: full`)
- **Use Case**: Maximum performance when compute available
- **Memory**: ~24-40GB VRAM for ~1B models (or use DeepSpeed)
- **Speed**: Slowest
- **Typical LR**: 2e-5 to 5e-5
- **Trainable Params**: 100%

### 3. PEFT/LoRA (`mode: peft`)
- **Use Case**: Best balance of performance and efficiency
- **Memory**: ~8-16GB VRAM for ~1B models
- **Speed**: Medium
- **Typical LR**: 5e-5 to 1e-4
- **Trainable Params**: <1% (adapters only)

## Expected Resource Footprints (~1B Models)

| Mode      | GPU Memory | Training Speed | Recommended GPU |
|-----------|-----------|----------------|-----------------|
| Head-Only | 4-8 GB    | ~1000 samples/sec | RTX 3060 12GB |
| PEFT      | 8-16 GB   | ~500 samples/sec | RTX 3090 24GB |
| Full      | 24-40 GB  | ~200 samples/sec | A100 40/80GB |

*Note: With DeepSpeed ZeRO-2/3, can fit larger models on smaller GPUs*

## Multi-GPU Training

### Using Accelerate

1. Create accelerate config:
```bash
accelerate config
```

2. Launch training:
```bash
accelerate launch src/train.py --config src/experiments/full_finetune_deepspeed.yaml
```

### Using torchrun

```bash
torchrun --nproc_per_node=4 src/train.py --config src/experiments/head_only.yaml
```

## Data Formats

### Single-Text Classification
```csv
text,label
"Positive example",1
"Negative example",0
```

### Text-Pair
```csv
text1,text2,label
"Query text","Related doc",1
"Query text","Unrelated doc",0
```

### Triplet
```csv
anchor,positive,negative
"Anchor text","Similar text","Dissimilar text"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_model_wrapper.py -v
```

## Docker

```bash
# Build image
docker build -t encoder-training -f docker/Dockerfile .

# Run with docker-compose
docker-compose -f docker/docker-compose.yaml up
```

## Scaling Guide

See [scaling.md](scaling.md) for detailed guidance on:
- Gradient accumulation strategies
- DeepSpeed ZeRO configuration
- FSDP vs DeepSpeed
- FP16 vs BF16 tradeoffs
- Multi-node training setup

## Common Issues

**OOM (Out of Memory)**
- Reduce `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`
- Enable DeepSpeed ZeRO-2 or ZeRO-3
- Use `fp16` or `bf16`

**Slow Training**
- Increase `per_device_train_batch_size` if memory allows
- Use faster pooling modes (cls > mean > weighted)
- Enable `dataloader_num_workers` for non-streaming

**NaN Loss**
- Lower learning rate
- Use `bf16` instead of `fp16` if available
- Add gradient clipping (`max_grad_norm: 1.0`)

## Citation

```bibtex
@software{encoder_training_v2,
  title={Encoder Training Framework v2.0},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/encoder-training-v2}
}
```

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your.email@example.com
