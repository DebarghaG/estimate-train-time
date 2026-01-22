# Creating Custom Configurations

This guide explains how to create your own configuration files for estimate-train-time.

## Starting from a Template

The easiest way to create a custom config is to start from a bundled example.

### Export an Example

```bash
estimate-train-time show-example llemma_7b_4_2_2_P > my_config.yml
```

Then edit `my_config.yml` to match your setup.

### Template Structure

```yaml
{
  # Hardware identification
  "gpu_name": "NVIDIAA100-SXM4-80GB",

  # Data paths (relative to bundled data or absolute)
  "operator_data_folder": "./regressors/Perlmutter/operator",
  "nccl_data_folder": "./regressors/Perlmutter/nccl",

  # Parallelism and model config
  # [pp, mp, dp, batch, heads, seq_len, hidden_dim, grad_accum, gpus_per_node]
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],

  # Communication settings
  "comm_bucket": 1260000000,

  # Model architecture
  "encoders": 32,  # Number of transformer layers

  # Operator lists
  "function_list": ['embedding', 'RMSlayernorm', 'linear1', 'RoPE',
                    'flash_atten', 'linear2', 'linear3', 'gelu',
                    'linear4', 'linear_final', 'parallel_cross_entropy_128',
                    'res_add'],
  "encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten',
                            'linear2', 'RMSlayernorm', 'res_add', 'linear3',
                            'gelu', 'linear4', 'res_add'],
  "layernorm_name": 'RMSlayernorm',

  # Synchronization points per layer
  "fwd_syncs": 2,
  "bwd_syncs": 2,
}
```

## Common Model Configurations

### LLaMA 7B

```yaml
{
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
  "encoders": 32,
  "layernorm_name": "RMSlayernorm",
}
```

Parameters:
- Hidden dimension: 4096
- Attention heads: 32
- Layers: 32
- Vocabulary: ~32K (uses 50257 internally padded)

### LLaMA 13B

```yaml
{
  "training_config": [4, 4, 2, 4, 40, 2048, 5120, 8, 4],
  "encoders": 40,
  "layernorm_name": "RMSlayernorm",
}
```

Parameters:
- Hidden dimension: 5120
- Attention heads: 40
- Layers: 40

### LLaMA 70B

```yaml
{
  "training_config": [8, 4, 4, 2, 64, 2048, 8192, 16, 4],
  "encoders": 80,
  "layernorm_name": "RMSlayernorm",
}
```

Parameters:
- Hidden dimension: 8192
- Attention heads: 64
- Layers: 80
- Note: Requires 128 GPUs (8 × 4 × 4)

## Parallelism Strategy Selection

### Understanding the Tradeoffs

| Strategy | Communication | Memory | Efficiency |
|----------|--------------|--------|------------|
| More PP | P2P (low bandwidth) | Lower per-GPU | Pipeline bubbles |
| More MP | All-reduce (high bandwidth) | Lower per-GPU | Higher sync overhead |
| More DP | All-reduce (gradients) | Full model per GPU | Best efficiency if fits |

### General Guidelines

1. **Start with DP**: Use data parallelism if the model fits in GPU memory
2. **Add MP for memory**: Use tensor parallelism to reduce per-GPU memory
3. **Add PP for scale**: Use pipeline parallelism for very large models

### Example: 16 GPU Configurations

All configurations below use 16 GPUs total:

```yaml
# Configuration A: Balanced (recommended starting point)
"training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
# pp=4, mp=2, dp=2 → 4 × 2 × 2 = 16 GPUs

# Configuration B: More tensor parallelism
"training_config": [2, 4, 2, 4, 32, 4096, 4096, 8, 4],
# pp=2, mp=4, dp=2 → 2 × 4 × 2 = 16 GPUs

# Configuration C: More data parallelism
"training_config": [2, 2, 4, 4, 32, 4096, 4096, 8, 4],
# pp=2, mp=2, dp=4 → 2 × 2 × 4 = 16 GPUs

# Configuration D: Deep pipeline
"training_config": [8, 1, 2, 4, 32, 4096, 4096, 8, 4],
# pp=8, mp=1, dp=2 → 8 × 1 × 2 = 16 GPUs
```

### Comparing Strategies

```python
from estimate_train_time import one_batch_predict
import yaml
import os

def create_config(pp, mp, dp, filename):
    config = {
        "gpu_name": "NVIDIAA100-SXM4-80GB",
        "operator_data_folder": "./regressors/Perlmutter/operator",
        "nccl_data_folder": "./regressors/Perlmutter/nccl",
        "training_config": [pp, mp, dp, 4, 32, 4096, 4096, 8, 4],
        "comm_bucket": 1260000000,
        "encoders": 32,
        "function_list": ['embedding', 'RMSlayernorm', 'linear1', 'RoPE',
                          'flash_atten', 'linear2', 'linear3', 'gelu',
                          'linear4', 'linear_final', 'parallel_cross_entropy_128',
                          'res_add'],
        "encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten',
                                  'linear2', 'RMSlayernorm', 'res_add', 'linear3',
                                  'gelu', 'linear4', 'res_add'],
        "layernorm_name": 'RMSlayernorm',
        "fwd_syncs": 2,
        "bwd_syncs": 2,
    }
    with open(filename, 'w') as f:
        yaml.dump(config, f)
    return filename

# Compare strategies
strategies = [
    (4, 2, 2, "Balanced"),
    (2, 4, 2, "More TP"),
    (2, 2, 4, "More DP"),
    (8, 1, 2, "Deep PP"),
]

print("Strategy Comparison (16 GPUs, LLaMA 7B)")
print("=" * 50)

for pp, mp, dp, name in strategies:
    config_path = create_config(pp, mp, dp, f"config_{pp}_{mp}_{dp}.yml")
    time_us = one_batch_predict(config_path)
    os.remove(config_path)  # Clean up

    print(f"{name:12} (PP={pp}, MP={mp}, DP={dp}): {time_us/1e6:.2f}s")
```

## Batch Size and Sequence Length

### Adjusting Batch Size

The batch size parameter is **per-GPU micro-batch size**:

```yaml
# Smaller batch for memory constraints
"training_config": [4, 2, 2, 2, 32, 4096, 4096, 16, 4],
#                          ^ batch=2, grad_accum=16

# Larger batch for throughput
"training_config": [4, 2, 2, 8, 32, 4096, 4096, 4, 4],
#                          ^ batch=8, grad_accum=4
```

Effective batch size = `batch × dp × gradient_accumulation`

### Adjusting Sequence Length

```yaml
# Shorter sequences (2K)
"training_config": [4, 2, 2, 8, 32, 2048, 4096, 8, 4],
#                                  ^ l=2048

# Longer sequences (8K)
"training_config": [4, 2, 2, 2, 32, 8192, 4096, 8, 4],
#                                  ^ l=8192 (reduce batch for memory)
```

## Cluster Configuration

### GPUs Per Node

The `gpus_per_node` parameter affects communication topology:

```yaml
# 4 GPUs per node (typical HPC)
"training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
#                                                ^ gpus_per_node=4

# 8 GPUs per node (DGX)
"training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 8],
#                                                ^ gpus_per_node=8

# 1 GPU per node (GH200 style)
"training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 1],
#                                                ^ gpus_per_node=1
```

### Communication Bucket Size

Controls gradient bucketing for data-parallel all-reduce:

```yaml
# Default (1.26 GB)
"comm_bucket": 1260000000,

# Smaller buckets (more frequent, smaller transfers)
"comm_bucket": 500000000,

# Larger buckets (fewer, larger transfers)
"comm_bucket": 2000000000,
```

## Using Custom Regressor Data

If you've profiled your own hardware (see [Kernel Sampling](../advanced/kernel-sampling.md)):

```yaml
{
  "gpu_name": "MyCustomGPU",
  "operator_data_folder": "/path/to/my/operator/regressors",
  "nccl_data_folder": "/path/to/my/nccl/regressors",
  ...
}
```

Ensure your regressor data includes:
- `{gpu_name}.csv` - Config with best model parameters
- `{gpu_name}_{operator}_{precision}_{propagation}.json` or `.pkl`

## Testing Your Configuration

### Validate Syntax

```bash
# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('my_config.yml'))"
```

### Run Prediction

```bash
estimate-train-time predict --config my_config.yml
```

### Common Errors

**Path not found:**
```
FileNotFoundError: Regressor data not found
```
→ Check `operator_data_folder` and `nccl_data_folder` paths

**Invalid operator:**
```
KeyError: 'unknown_op'
```
→ Check operator names in `function_list`

**Wrong config length:**
```
ValueError: training_config must have 9 elements
```
→ Ensure all 9 values: `[pp, mp, dp, b, h, l, dim, steps, gpus_per_node]`

## Full Example: Custom LLaMA 13B Config

```yaml
{
  # Target: NVIDIA A100 on Perlmutter cluster
  "gpu_name": "NVIDIAA100-SXM4-80GB",
  "operator_data_folder": "./regressors/Perlmutter/operator",
  "nccl_data_folder": "./regressors/Perlmutter/nccl",

  # 32 GPUs: 4 pipeline × 4 tensor × 2 data parallel
  # Batch=4, 40 heads, 2K seq len, 5120 hidden, 8 grad accum, 4 GPUs/node
  "training_config": [4, 4, 2, 4, 40, 2048, 5120, 8, 4],

  "comm_bucket": 1260000000,

  # LLaMA 13B has 40 transformer layers
  "encoders": 40,

  # Standard LLaMA operator set
  "function_list": ['embedding', 'RMSlayernorm', 'linear1', 'RoPE',
                    'flash_atten', 'linear2', 'linear3', 'gelu',
                    'linear4', 'linear_final', 'parallel_cross_entropy_128',
                    'res_add'],

  # Operators per transformer layer
  "encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten',
                            'linear2', 'RMSlayernorm', 'res_add', 'linear3',
                            'gelu', 'linear4', 'res_add'],

  # RMS LayerNorm (LLaMA style)
  "layernorm_name": 'RMSlayernorm',

  # 2 sync points per layer (after attention, after FFN)
  "fwd_syncs": 2,
  "bwd_syncs": 2,
}
```

Save as `llama_13b_32gpu.yml` and run:

```bash
estimate-train-time predict --config llama_13b_32gpu.yml
```

## See Also

- [Configuration Reference](../configuration.md) - Complete parameter documentation
- [Core Concepts](../concepts.md) - Understanding parallelism strategies
- [Basic Usage](basic-usage.md) - Using bundled examples
