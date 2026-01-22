# Getting Started

This guide walks you through installing estimate-train-time and running your first prediction.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation

### Basic Installation (Prediction Only)

For running predictions using pre-trained regressors:

```bash
pip install estimate-train-time
```

This installs the core package with dependencies:
- pandas
- numpy
- scikit-learn
- xgboost
- pyyaml
- ijson
- joblib

### GPU Installation (For Sampling)

If you plan to profile your own GPUs and create custom regressors:

```bash
pip install estimate-train-time[gpu]
```

This adds:
- torch
- flash-attn (>=2.5.6)
- deepspeed

### Development Installation

For contributing or development:

```bash
git clone https://github.com/AI4CI/estimate-train-time.git
cd estimate-train-time
pip install -e ".[dev]"
```

## Verify Installation

Check that the package is installed correctly:

```bash
estimate-train-time --version
```

Expected output:
```
estimate-train-time 0.1.0
```

## Run Your First Prediction

### Step 1: List Available Examples

The package comes with bundled example configurations:

```bash
estimate-train-time list-examples
```

Output:
```
Available example configurations:
----------------------------------------
  llemma_7b_4_2_2_P
  llemma_7b_4_2_2_V

Use 'estimate-train-time show-example <name>' to view a configuration.
Use 'estimate-train-time predict --example <name>' to run prediction.
```

### Step 2: View an Example Configuration

Inspect what's in an example config:

```bash
estimate-train-time show-example llemma_7b_4_2_2_P
```

This shows the YAML configuration including:
- GPU type (NVIDIA A100-SXM4-80GB)
- Parallelism settings (4 pipeline, 2 model, 2 data)
- Model architecture (32 transformer layers, 4096 hidden dim)
- Operator list (embedding, attention, linear layers, etc.)

### Step 3: Run Prediction

Run the estimator with an example:

```bash
estimate-train-time predict --example llemma_7b_4_2_2_P
```

Output:
```
Running prediction with config: /path/to/examples/llemma_7b_4_2_2_P.yml
----------------------------------------
Loading /path/to/regressors/...
Function:embedding_fp16_fwd    Input:[...]    Prediction:...
...

Estimated time cost of current training config: 9480819.17 us
                                               = 9480.82 ms
                                               = 9.4808 s
```

## Understanding the Output

The prediction shows:

1. **Loading messages**: The tool loads pre-trained regressor models for each operator
2. **Per-operator predictions**: Time estimates for each compute/communication operation
3. **Final estimate**: Total time for one training step (parameter update) in:
   - Microseconds (us)
   - Milliseconds (ms)
   - Seconds (s)

The output represents the wall-clock time for **one complete training step** including:
- Forward passes through all pipeline stages
- Backward passes with gradient computation
- Gradient synchronization across data-parallel ranks
- Optimizer step with parameter updates

## Using Your Own Configuration

Create a custom config file (see [Configuration Reference](configuration.md)):

```bash
estimate-train-time predict --config /path/to/your/config.yml
```

## Python API

For programmatic access:

```python
from estimate_train_time import one_batch_predict

# Using a config file
time_us = one_batch_predict("config.yml")
print(f"Training step takes {time_us / 1e6:.2f} seconds")

# Calculate time for full training
steps = 10000
total_hours = (time_us * steps) / 1e6 / 3600
print(f"Full training: {total_hours:.1f} hours")
```

## Next Steps

- [Core Concepts](concepts.md) - Understand how the estimator works
- [Configuration Reference](configuration.md) - Learn all config parameters
- [Examples](examples/basic-usage.md) - More usage examples
- [Advanced: Kernel Sampling](advanced/kernel-sampling.md) - Profile your own GPUs
