# Kernel Sampling Guide

This guide explains how to profile GPU compute operators to create custom regressor data for estimate-train-time.

## Overview

Kernel sampling profiles individual GPU operators (embedding, attention, linear layers, etc.) to build timing models. The process:

1. Run operators with various input shapes
2. Capture GPU kernel execution times via PyTorch profiler
3. Train regressors to predict time from input shape

## Prerequisites

### Hardware

- One or more NVIDIA GPUs (same model for consistency)
- Sufficient GPU memory for largest tensor shapes

### Software

Install the GPU dependencies:

```bash
pip install estimate-train-time[gpu]
```

Or manually:

```bash
pip install torch flash-attn>=2.5.6 deepspeed
```

Required environment:
- CUDA toolkit compatible with your PyTorch version
- Megatron-style fused kernels (included in package)

## Sampling Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Kernel Sampling Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Shape     │    │   Operator   │    │   Profiler   │      │
│  │  Generator   │───►│   Execution  │───►│   Capture    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │   Timing     │       │
│                                          │  Extraction  │       │
│                                          └──────────────┘       │
│                                                 │                │
│                                                 ▼                │
│                                          ┌──────────────┐       │
│                                          │     CSV      │       │
│                                          │   Output     │       │
│                                          └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Files

Sampling is controlled by YAML configuration files. Two directories exist:

- `Kernel_sampling/configs/collect/` - Full parameter sweeps
- `Kernel_sampling/configs/test/` - Quick test configurations

### Configuration Structure

Example: `flash_atten.yml`

```yaml
{
  # Module and function to profile
  "module_name": "target_functions",
  "kernel_name": "flash_atten",
  "function_name": "flash_atten",

  # Output CSV columns
  "columns_name": ['mp', 'b', 'h', 'l', 'dim', 'F_dur(us)', 'B_dur(us)'],

  # Profiler trace targets for extraction
  "targets": [
    ['built-in method apply of FunctionMeta object', 'FlashAttnFunc'],
    ['autograd::engine::evaluate_function: FlashAttnFuncBackward']
  ],

  # Number of shape parameters
  "first_n_column": 5,

  # Parameter sweep ranges
  # mp, b, h, l, dim
  "starts": [1, 4, 16, 1024, 2048],
  "steps": [2, 2, 8, 512, 512],
  "ends": [16, 8, 80, 5120, 8192],
  "operators": ['mul', 'mul', 'add', 'add', 'add'],

  # Warmup shapes (small, fast)
  "warmup_shapes": [1, 4, 16, 1024, 2048],

  # Profiler settings
  "wait": 2,      # Iterations before profiling
  "warmup": 2,    # Warmup iterations
  "active": 10,   # Profiled iterations

  # Test/debug shapes
  "shapes": [1, 4, 16, 1024, 2048],

  # Run mode: get_all, get_one, get_profiler
  "run": "get_all"
}
```

### Parameter Sweep

The shape generator creates combinations using these rules:

| Operator | Description |
|----------|-------------|
| `'mul'` | Multiply by step: `start, start*step, start*step*step, ...` |
| `'add'` | Add step: `start, start+step, start+2*step, ...` |

Example for `mp`:
- `starts[0]=1, steps[0]=2, ends[0]=16, operators[0]='mul'`
- Values: 1, 2, 4, 8, 16

## Running Sampling

### Single Operator Test

Test with a specific shape:

```bash
cd Kernel_sampling
python sampling_controller.py \
    --config_path ./configs/test/flash_atten.yml \
    --precision fp16
```

### Full Collection

Run complete parameter sweep:

```bash
cd Kernel_sampling
python sampling_controller.py \
    --config_path ./configs/collect/flash_atten.yml \
    --precision fp16
```

### Parallelizing Across GPUs

For large sweeps, split work across multiple GPUs:

```bash
# GPU 0: process part 1 of 4
CUDA_VISIBLE_DEVICES=0 python sampling_controller.py \
    --config_path ./configs/collect/flash_atten.yml \
    --precision fp16 \
    --parts 4 \
    --part 1 \
    --device_num 0

# GPU 1: process part 2 of 4
CUDA_VISIBLE_DEVICES=1 python sampling_controller.py \
    --config_path ./configs/collect/flash_atten.yml \
    --precision fp16 \
    --parts 4 \
    --part 2 \
    --device_num 0
```

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --job-name=kernel_sampling

cd Kernel_sampling

# Run 4 parallel sampling jobs
for i in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((i-1)) python sampling_controller.py \
        --config_path ./configs/collect/flash_atten.yml \
        --precision fp16 \
        --parts 4 \
        --part $i \
        --device_num 0 &
done

wait
```

## Available Operators

The following operators can be profiled:

### Compute Operators

| Operator | Description | Shape Parameters |
|----------|-------------|------------------|
| `embedding` | Token embedding | mp, b, l, dim |
| `layernorm` | Standard LayerNorm | b, l, dim |
| `RMSlayernorm` | RMS LayerNorm | b, l, dim |
| `linear1` | QKV projection | mp, b, l, dim |
| `linear2` | Output projection | mp, b, l, dim |
| `linear3` | FFN up-projection | mp, b, l, dim |
| `linear4` | FFN down-projection | mp, b, l, dim |
| `linear_final` | Output layer | mp, b, l, dim |
| `flash_atten` | Flash Attention | mp, b, h, l, dim |
| `gelu` | GELU activation | mp, b, l, dim |
| `RoPE` | Rotary embeddings | mp, b, h, l, dim |
| `res_add` | Residual addition | b, l, dim |
| `parallel_cross_entropy_128` | Parallel CE loss | mp, b, l |

### Optimizer Operators

| Operator | Description |
|----------|-------------|
| `firstStage_optimizer` | Optimizer for first pipeline stage |
| `middleStage_optimizer` | Optimizer for middle stages |
| `lastStage_optimizer` | Optimizer for last pipeline stage |

## Output Files

### Raw Data

Sampling data is saved to `Kernel_sampling/sampling_data/`:

```
{GPU_NAME}_{operator}_{precision}_{parts}_{part}.csv
```

Example: `NVIDIAA100-SXM4-80GB_flash_atten_fp16_4_1.csv`

CSV format:
```csv
mp,b,h,l,dim,F_dur(us),B_dur(us)
1,4,16,1024,2048,45.2,89.3
1,4,16,1024,2560,52.1,103.4
...
```

### Progress Logs

Logs are saved to `Kernel_sampling/sampling_log/`:

```
{GPU_NAME}_{operator}_{precision}_{parts}_{part}.txt
```

Sample log content:
```
[2024-01-15 10:30:45]Kernel: flash_atten_fp16 | Progress: 100/1000=10.00% | Speed: 1.23s/sample | Remaining: 0d3h12m45s
```

## Processing Results

After sampling, merge multi-part data and train regressors.

### Merge CSV Files

```python
import pandas as pd
import glob

# Merge parts
pattern = "sampling_data/NVIDIAA100*_flash_atten_fp16_*.csv"
files = glob.glob(pattern)

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)

# Remove duplicates and sort
merged = merged.drop_duplicates()
merged = merged.sort_values(list(merged.columns[:-2]))

# Save merged data
merged.to_csv("NVIDIAA100-SXM4-80GB_flash_atten_fp16.csv", index=False)
```

### Train Regressors

The estimator automatically trains regressors when first needed. To pre-train:

```python
from estimate_train_time.estimator.predictor import Predictor
import pandas as pd

# Load sampling data
data_path = "NVIDIAA100-SXM4-80GB_flash_atten_fp16.csv"
df = pd.read_csv(data_path)

# Split features and targets
X = df.iloc[:, :-2].values  # All columns except last 2
y_fwd = df['F_dur(us)'].values
y_bwd = df['B_dur(us)'].values

# Create predictor and trigger model building
predictor = Predictor()
# Models will be built on first prediction request
```

## Custom Operators

To add a new operator:

### 1. Implement the Operator Function

Create in `Kernel_sampling/target_functions.py`:

```python
def my_custom_op(shapes, precision, device_num):
    """Profile a custom operation."""
    mp, b, l, dim = shapes

    dtype = torch.float16 if precision == 'fp16' else torch.float32
    device = f'cuda:{device_num}'

    # Create input tensors
    x = torch.randn(b, l, dim, dtype=dtype, device=device, requires_grad=True)

    # Forward pass
    y = my_operation(x)

    # Backward pass (for gradient timing)
    y.sum().backward()

    # Sync GPU
    torch.cuda.synchronize()
```

### 2. Create Shape Transform Functions

In `encoder_config_to_layer_input.py`:

```python
def my_custom_op(input):
    """Map encoder config to operator input shape."""
    mp, b, h, l, dim = input
    return np.array([mp, b, l, dim])
```

In `layer_input_to_predictor_input.py`:

```python
def my_custom_op(input):
    """Map operator input to predictor features."""
    mp, b, l, dim = input
    return np.array([b * l, dim])
```

### 3. Create Sampling Config

Create `configs/collect/my_custom_op.yml`:

```yaml
{
  "module_name": "target_functions",
  "kernel_name": "my_custom_op",
  "function_name": "my_custom_op",
  "columns_name": ['mp', 'b', 'l', 'dim', 'F_dur(us)', 'B_dur(us)'],
  "targets": [['my_op_kernel_name'], ['my_op_backward']],
  "first_n_column": 4,
  "starts": [1, 4, 1024, 2048],
  "steps": [2, 2, 512, 512],
  "ends": [16, 8, 5120, 8192],
  "operators": ['mul', 'mul', 'add', 'add'],
  "warmup_shapes": [1, 4, 1024, 2048],
  "wait": 2,
  "warmup": 2,
  "active": 10,
  "shapes": [1, 4, 1024, 2048],
  "run": "get_all"
}
```

### 4. Run Sampling

```bash
python sampling_controller.py \
    --config_path ./configs/collect/my_custom_op.yml \
    --precision fp16
```

## Troubleshooting

### Out of Memory

Reduce the largest shape parameters:

```yaml
"ends": [8, 4, 2048, 4096],  # Smaller max values
```

Or reduce batch size in warmup:

```yaml
"warmup_shapes": [1, 2, 512, 2048],  # Smaller warmup
```

### Profiler Not Capturing

Check that `targets` strings match the profiler output. Run with `get_profiler` mode:

```yaml
"run": "get_profiler"
```

Then inspect the generated trace file in TensorBoard.

### Empty Results

Check the log file for errors. Common issues:
- CUDA out of memory for some shapes
- Function not returning expected tensors
- Profiler target strings incorrect

## See Also

- [NCCL Sampling](nccl-sampling.md) - Communication profiling
- [Extending](extending.md) - Adding new capabilities
- [Core Concepts](../concepts.md) - Understanding the prediction model
