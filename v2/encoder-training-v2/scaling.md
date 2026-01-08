# Scaling Guide for Production Deployment

This guide provides prescriptive instructions for scaling encoder training to production workloads with ~1B parameter models.

## Table of Contents
1. [Gradient Accumulation](#gradient-accumulation)
2. [DeepSpeed ZeRO](#deepspeed-zero)
3. [FSDP (Fully Sharded Data Parallel)](#fsdp)
4. [Mixed Precision Training](#mixed-precision-training)
5. [Multi-Node Training](#multi-node-training)
6. [Memory Optimization Checklist](#memory-optimization-checklist)

---

## Gradient Accumulation

### What is it?
Gradient accumulation simulates larger batch sizes by accumulating gradients over multiple forward/backward passes before updating weights.

### When to use
- Limited GPU memory prevents using desired batch size
- Want to maintain equivalent training dynamics to larger batch training
- Single GPU or small multi-GPU setup

### How to configure

**Effective Batch Size = `per_device_batch_size` × `gradient_accumulation_steps` × `num_gpus`**

Example configurations:

```yaml
# Target effective batch size: 128
# Single GPU (24GB)
training:
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 32
  # Effective: 4 * 32 * 1 = 128

# 4 GPUs (24GB each)
training:
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  # Effective: 8 * 4 * 4 = 128
```

### Tradeoffs
- ✅ **Pro**: Can simulate any batch size
- ✅ **Pro**: No code changes needed
- ❌ **Con**: Slower training (more forward/backward passes per update)
- ❌ **Con**: BatchNorm statistics may be affected (use LayerNorm instead)

---

## DeepSpeed ZeRO

### What is it?
ZeRO (Zero Redundancy Optimizer) partitions model states (optimizer, gradients, parameters) across GPUs to reduce per-GPU memory.

### Stages

| Stage | Partitions | Memory Reduction | Speed Impact |
|-------|-----------|------------------|--------------|
| ZeRO-1 | Optimizer states | ~4x | Minimal |
| ZeRO-2 | + Gradients | ~8x | Small (~10%) |
| ZeRO-3 | + Parameters | ~Linear with #GPUs | Moderate (~20-30%) |

### When to use each stage

**ZeRO-1**: Multi-GPU with plenty of VRAM
**ZeRO-2**: **Recommended for ~1B models** - best balance
**ZeRO-3**: Very large models (>50B) or limited VRAM

### Configuration Example (ZeRO-2)

```yaml
deepspeed:
  enabled: true
  config:
    train_batch_size: 128  # Total across all GPUs
    train_micro_batch_size_per_gpu: 4
    gradient_accumulation_steps: 8
    
    fp16:
      enabled: true
      loss_scale: 0
      loss_scale_window: 1000
      hysteresis: 2
      min_loss_scale: 1
    
    zero_optimization:
      stage: 2  # ZeRO-2
      
      # Offload optimizer to CPU (saves more memory)
      offload_optimizer:
        device: "cpu"
        pin_memory: true
      
      # Communication optimizations
      overlap_comm: true
      contiguous_gradients: true
      reduce_bucket_size: 5e8
      allgather_bucket_size: 5e8
```

### CPU Offloading

Offload optimizer and/or parameters to CPU RAM:

```yaml
zero_optimization:
  stage: 2
  offload_optimizer:
    device: "cpu"
    pin_memory: true  # Faster CPU-GPU transfers
  
  # Stage 3 only: offload parameters
  offload_param:
    device: "cpu"
    pin_memory: true
```

**Tradeoff**: ~30% slower but 2-3x memory reduction

### Running with DeepSpeed

```bash
# Single node, 4 GPUs
deepspeed --num_gpus=4 src/train.py --config experiments/full_finetune_deepspeed.yaml

# Or via accelerate
accelerate launch --config_file accelerate_ds.yaml src/train.py --config experiments/full_finetune_deepspeed.yaml
```

---

## FSDP (Fully Sharded Data Parallel)

### What is it?
PyTorch-native alternative to DeepSpeed. Similar to ZeRO-3 but integrated into PyTorch.

### When to use
- PyTorch 2.0+
- Prefer native PyTorch stack
- Simpler than DeepSpeed for basic use cases

### Configuration (via Accelerate)

Create `accelerate_fsdp.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_sharding_strategy: 1  # Full shard (like ZeRO-3)
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: BertLayer  # Model-specific
mixed_precision: fp16
```

Launch:
```bash
accelerate launch --config_file accelerate_fsdp.yaml src/train.py --config experiments/head_only.yaml
```

### FSDP vs DeepSpeed

| Feature | FSDP | DeepSpeed |
|---------|------|-----------|
| Ease of use | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Memory efficiency | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Flexibility | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Recommendation**: Start with DeepSpeed ZeRO-2; move to FSDP if staying in PyTorch ecosystem.

---

## Mixed Precision Training

### FP16 (Float16)

**Best for**: Volta+ GPUs (V100, A100, RTX 30xx/40xx)

```yaml
training:
  fp16: true
  bf16: false
```

**Characteristics**:
- 2x memory reduction
- 2-3x speedup on Tensor Cores
- Risk of numerical instability (especially small gradients)

**When to avoid**:
- Training diverges (NaN loss)
- Very small learning rates (<1e-6)

### BF16 (BFloat16)

**Best for**: Ampere+ GPUs (A100, RTX 30xx/40xx)

```yaml
training:
  fp16: false
  bf16: true
```

**Characteristics**:
- 2x memory reduction
- 2-3x speedup on Ampere+ Tensor Cores
- Better numerical stability than FP16 (wider dynamic range)
- **Recommended for production if available**

### FP16 vs BF16

| Aspect | FP16 | BF16 |
|--------|------|------|
| Memory | ✅ 2x reduction | ✅ 2x reduction |
| Speed | ✅ Fast (Volta+) | ✅ Fast (Ampere+) |
| Stability | ⚠️ Can diverge | ✅ Stable |
| Gradient Scaling | Required | Not needed |
| GPU Support | Broader | Ampere+ only |

**Rule of thumb**: Use BF16 if GPU supports it, otherwise FP16 with gradient scaling.

---

## Multi-Node Training

### Setup

**Node 1** (master):
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    src/train.py --config experiments/full_finetune_deepspeed.yaml
```

**Node 2**:
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    src/train.py --config experiments/full_finetune_deepspeed.yaml
```

### Network Requirements
- Low-latency interconnect (InfiniBand ideally, 10GbE minimum)
- All nodes must be able to reach master node
- Shared filesystem or consistent dataset access

### Debugging Multi-Node

1. **Test single-node first**: Ensure training works on one node
2. **Check network**: `ping` between nodes
3. **Verify NCCL**: Set `NCCL_DEBUG=INFO` for verbose logging
4. **Firewall**: Ensure port 29500 (or chosen port) is open

---

## Memory Optimization Checklist

When facing OOM errors, try in order:

1. ✅ **Reduce batch size**
   ```yaml
   per_device_train_batch_size: 4  # Start small
   ```

2. ✅ **Increase gradient accumulation**
   ```yaml
   gradient_accumulation_steps: 8  # Maintain effective batch size
   ```

3. ✅ **Enable mixed precision**
   ```yaml
   fp16: true  # or bf16: true
   ```

4. ✅ **Use gradient checkpointing**
   ```python
   model.gradient_checkpointing_enable()  # Trades compute for memory
   ```

5. ✅ **Enable DeepSpeed ZeRO-2**
   ```yaml
   deepspeed:
     enabled: true
     config:
       zero_optimization:
         stage: 2
   ```

6. ✅ **Offload optimizer to CPU**
   ```yaml
   zero_optimization:
     offload_optimizer:
       device: "cpu"
   ```

7. ✅ **Use DeepSpeed ZeRO-3** (last resort - slow)
   ```yaml
   zero_optimization:
     stage: 3
   ```

---

## Recommended Configurations by GPU

### RTX 3060 (12GB)
```yaml
training:
  mode: "head_only"  # or "peft"
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4
  fp16: true
```

### RTX 3090 / A6000 (24GB)
```yaml
training:
  mode: "peft"  # or "full" with ZeRO-2
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 2
  fp16: true  # or bf16 for 30xx series
deepspeed:
  enabled: true  # Only for full mode
  config:
    zero_optimization:
      stage: 2
```

### A100 (40GB)
```yaml
training:
  mode: "full"
  per_device_train_batch_size: 32
  gradient_accumulation_steps: 1
  bf16: true
deepspeed:
  enabled: true
  config:
    zero_optimization:
      stage: 2
```

### A100 (80GB)
```yaml
training:
  mode: "full"
  per_device_train_batch_size: 64
  gradient_accumulation_steps: 1
  bf16: true
# DeepSpeed optional (plenty of VRAM)
```

---

## Performance Monitoring

### Key Metrics

1. **Throughput**: samples/second
2. **GPU Utilization**: Should be >90% during training
3. **Memory Usage**: Monitor with `nvidia-smi`
4. **Loss**: Should decrease smoothly

### Tools

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# TensorBoard
tensorboard --logdir outputs/

# W&B (if integrated)
wandb login
```

---

## Summary Table

| Optimization | Memory Savings | Speed Impact | Complexity |
|--------------|---------------|--------------|------------|
| Reduce batch size | Linear | Slower (more steps) | Easy |
| Gradient accumulation | 0 | Slower (more forward/backward) | Easy |
| FP16/BF16 | 2x | 2-3x faster | Easy |
| Gradient checkpointing | 30-40% | 20-30% slower | Medium |
| DeepSpeed ZeRO-2 | 2-4x | 10% slower | Medium |
| CPU offloading | 2-3x | 30% slower | Medium |
| DeepSpeed ZeRO-3 | Linear with GPUs | 30% slower | Medium |

**Golden rule**: Start simple (batch size + FP16), then add optimizations as needed.
