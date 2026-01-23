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
pip install estimate-train-time[gpu]  # Coming soon to PyPI
```

**Note:** For now, install from repository with GPU extras:

```bash
git clone https://github.com/AI4CI/estimate-train-time.git
cd estimate-train-time
pip install -e ".[gpu]"
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

Sampling is controlled by YAML configuration files in the package data directory:

- `src/estimate_train_time/data/configs/kernel_sampling/collect/` - Full parameter sweeps
- `src/estimate_train_time/data/configs/kernel_sampling/test/` - Quick test configurations
- `src/estimate_train_time/data/configs/kernel_sampling/profilers/` - Profiler configurations

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

  # Run mode: get_all, get_one, get_profiler, run_function
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

**Note:** Kernel sampling must be run from the source directory due to relative imports in `sampling_controller.py`. Clone the repository and run from `src/estimate_train_time/kernel_sampling/`. The pip-installed package does not currently support running the sampling scripts directly.

### Single Operator Test

Test with a specific shape:

```bash
cd src/estimate_train_time/kernel_sampling
python sampling_controller.py \
    --config_path ../data/configs/kernel_sampling/test/flash_atten.yml \
    --precision fp16
```

### Full Collection

Run complete parameter sweep:

```bash
cd src/estimate_train_time/kernel_sampling
python sampling_controller.py \
    --config_path ../data/configs/kernel_sampling/collect/flash_atten.yml \
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

cd src/estimate_train_time/kernel_sampling

# Run 4 parallel sampling jobs
for i in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$((i-1)) python sampling_controller.py \
        --config_path ../data/configs/kernel_sampling/collect/flash_atten.yml \
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
| `softmax` | Softmax operation | mp, b, h, l |
| `fillmask` | Attention mask filling | mp, b, h, l, dim |
| `baddbmm` | Batch add + matrix mult | mp, b, h, l, dim |
| `bmm` | Batch matrix multiply | mp, b, h, l, dim |
| `ScaledUpperTriangMaskedSoftmax` | Fused masked softmax | mp, b, h, l |
| `moe` | Mixture of Experts | mp, b, l, dim, moe_num_experts, intermediate_size, top_k |

### Optimizer Operators

| Operator | Description | Shape Parameters |
|----------|-------------|------------------|
| `firstStage_optimizer` | Optimizer for first pipeline stage | mp, dim, encoders |
| `middleStage_optimizer` | Optimizer for middle stages | mp, dim, encoders |
| `lastStage_optimizer` | Optimizer for last pipeline stage | mp, dim, encoders |

## Output Files

### Raw Data

Sampling data is saved to `sampling_data/` in the current working directory:

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

Logs are saved to `sampling_log/` in the current working directory:

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

The estimator automatically trains regressors when first needed. Place merged CSV files in the appropriate regressor data folder and ensure a configuration CSV exists with model hyperparameters.

```python
from estimate_train_time.estimator.predictor import Predictor

# Create predictor instance
predictor = Predictor()

# Predict operator timing (models are built/loaded automatically)
# Required parameters:
#   - gpu_name: GPU identifier (e.g., "NVIDIAA100-SXM4-80GB")
#   - function: operator name (e.g., "flash_atten")
#   - precision: "fp16", "bf16", or "fp32"
#   - propagation: "fwd" or "bwd"
#   - shape: input shape as list (e.g., [1, 4, 32, 2048, 4096])
#   - model_dict: dictionary to cache loaded models
#   - config_folder: path to folder with model config CSV
#   - data_folder: path to folder with sampling data CSV files

result = predictor.predict_operator(
    gpu_name="NVIDIAA100-SXM4-80GB",
    function="flash_atten",
    precision="fp16",
    propagation="fwd",
    shape=[1, 4, 32, 2048, 4096],
    model_dict=predictor.operator_dict,
    config_folder="path/to/config",
    data_folder="path/to/data"
)
```

## Custom Operators

To add a new operator:

### 1. Implement the Operator Function

Create in `src/estimate_train_time/kernel_sampling/target_functions.py`:

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

In `src/estimate_train_time/estimator/encoder_config_to_layer_input.py`:

```python
def my_custom_op(input):
    """Map encoder config to operator input shape."""
    mp, b, h, l, dim = input
    return np.array([mp, b, l, dim])
```

In `src/estimate_train_time/estimator/layer_input_to_predictor_input.py`:

```python
def my_custom_op(input):
    """Map operator input to predictor features."""
    mp, b, l, dim = input
    return np.array([b * l, dim])
```

### 3. Create Sampling Config

Create `src/estimate_train_time/data/configs/kernel_sampling/collect/my_custom_op.yml`:

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
cd src/estimate_train_time/kernel_sampling
python sampling_controller.py \
    --config_path ../data/configs/kernel_sampling/collect/my_custom_op.yml \
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
