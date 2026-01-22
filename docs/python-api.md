# Python API Reference

This document describes the Python API for estimate-train-time.

## Installation

```bash
pip install estimate-train-time
```

## Quick Start

```python
from estimate_train_time import one_batch_predict

# Predict training time from a config file
time_us = one_batch_predict("path/to/config.yml")
print(f"One training step takes {time_us / 1e6:.2f} seconds")
```

## Module: estimate_train_time

### one_batch_predict

```python
def one_batch_predict(config_path: str) -> float
```

Predict the time cost for one training step (parameter update).

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` | Path to a YAML configuration file |

**Returns:**

| Type | Description |
|------|-------------|
| `float` | Estimated time in microseconds |

**Raises:**

| Exception | Condition |
|-----------|-----------|
| `FileNotFoundError` | Config file doesn't exist |
| `yaml.YAMLError` | Invalid YAML syntax |
| `KeyError` | Missing required config parameter |
| `ValueError` | Invalid parameter values |

**Example:**

```python
from estimate_train_time import one_batch_predict

# Basic usage
time_us = one_batch_predict("config.yml")
print(f"Time per step: {time_us:.2f} us")

# Convert to other units
time_ms = time_us / 1000
time_s = time_us / 1_000_000
print(f"Time per step: {time_ms:.2f} ms = {time_s:.4f} s")

# Calculate training time
total_steps = 10_000
total_time_hours = (time_us * total_steps) / 1e6 / 3600
print(f"Training time for {total_steps} steps: {total_time_hours:.1f} hours")
```

### Predictor

```python
class Predictor:
    def __init__(self) -> None
```

Advanced class for fine-grained control over predictions. Use this when you need to:
- Reuse loaded regressor models across multiple predictions
- Access individual operator timing predictions
- Customize the prediction pipeline

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `operator_dict` | `dict` | Cache of loaded compute operator regressors |
| `nccl_dict` | `dict` | Cache of loaded communication regressors |

**Methods:**

#### operator_statistic

```python
def operator_statistic(
    self,
    data_folder: str,
    config_folder: str,
    gpu_name: str,
    function_list: list[str],
    precision: str,
    encoder_config: tuple,
    propagation: str
) -> float
```

Get timing statistics for compute operators.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_folder` | `str` | Path to operator regressor data |
| `config_folder` | `str` | Path to operator config CSV |
| `gpu_name` | `str` | GPU identifier string |
| `function_list` | `list[str]` | Operators to time |
| `precision` | `str` | Data precision (`'fp16'` or `'fp32'`) |
| `encoder_config` | `tuple` | `(mp, b, h, l, dim)` |
| `propagation` | `str` | `'fwd'` or `'bwd'` |

**Returns:** Total time in microseconds for all specified operators.

#### mp_allreduce

```python
def mp_allreduce(
    self,
    nccl_data_folder: str,
    nccl_config_folder: str,
    gpu_name: str,
    shape: int,
    mp: int,
    gpu_per_node: int
) -> float
```

Predict all-reduce time for model (tensor) parallelism.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `nccl_data_folder` | `str` | Path to NCCL regressor data |
| `nccl_config_folder` | `str` | Path to NCCL config CSV |
| `gpu_name` | `str` | GPU identifier string |
| `shape` | `int` | Tensor size in elements |
| `mp` | `int` | Model parallelism degree |
| `gpu_per_node` | `int` | GPUs per node |

**Returns:** All-reduce time in microseconds.

#### dp_allreduce

```python
def dp_allreduce(
    self,
    nccl_data_folder: str,
    nccl_config_folder: str,
    gpu_name: str,
    shape: int,
    mp: int,
    dp: int,
    gpu_per_node: int
) -> float
```

Predict all-reduce time for data parallelism gradient synchronization.

#### dp_allgather

```python
def dp_allgather(
    self,
    nccl_data_folder: str,
    nccl_config_folder: str,
    gpu_name: str,
    shape: int,
    mp: int,
    dp: int,
    gpu_per_node: int
) -> float
```

Predict all-gather time for ZeRO optimizer parameter gathering.

#### pp_p2p

```python
def pp_p2p(
    self,
    nccl_data_folder: str,
    nccl_config_folder: str,
    gpu_name: str,
    shape: int,
    mp: int,
    dp: int,
    pp: int,
    gpu_per_node: int
) -> float
```

Predict point-to-point send/recv time for pipeline parallelism.

**Example:**

```python
from estimate_train_time import Predictor

# Create predictor instance
predictor = Predictor()

# Configure paths
operator_folder = "./regressors/Perlmutter/operator"
nccl_folder = "./regressors/Perlmutter/nccl"
gpu_name = "NVIDIAA100-SXM4-80GB"

# Encoder config: (mp, b, h, l, dim)
encoder_config = (2, 4, 32, 4096, 4096)

# Get forward pass time for flash attention
fwd_time = predictor.operator_statistic(
    operator_folder,
    operator_folder,
    gpu_name,
    ['flash_atten'],
    'fp16',
    encoder_config,
    'fwd'
)
print(f"Flash attention forward: {fwd_time:.2f} us")

# Get communication time for tensor parallel all-reduce
allreduce_time = predictor.mp_allreduce(
    nccl_folder,
    nccl_folder,
    gpu_name,
    shape=4 * 4096 * 4096,  # b * l * dim
    mp=2,
    gpu_per_node=4
)
print(f"TP all-reduce: {allreduce_time:.2f} us")
```

## Module: estimate_train_time.data

Utilities for accessing bundled data paths.

### get_data_path

```python
def get_data_path() -> Path
```

Get the path to the bundled data directory.

**Returns:** Path object to the data directory.

### get_regressors_path

```python
def get_regressors_path() -> Path
```

Get the path to the bundled regressors directory.

**Returns:** Path object to the regressors directory.

### get_examples_path

```python
def get_examples_path() -> Path
```

Get the path to the bundled examples directory.

**Returns:** Path object to the examples directory.

**Example:**

```python
from estimate_train_time.data import get_examples_path, get_regressors_path

# Get paths to bundled data
examples_dir = get_examples_path()
print(f"Examples at: {examples_dir}")

regressors_dir = get_regressors_path()
print(f"Regressors at: {regressors_dir}")

# Use in your code
import os
config_path = os.path.join(examples_dir, "llemma_7b_4_2_2_P.yml")
```

## Module: estimate_train_time.estimator.tools

Low-level utility functions.

### config_decoder

```python
def config_decoder(file_path: str) -> dict
```

Parse a YAML configuration file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_path` | `str` | Path to YAML file |

**Returns:** Dictionary with configuration values.

### get_bundled_data_path

```python
def get_bundled_data_path() -> str
```

Get the path to bundled package data.

**Returns:** String path to data directory.

### get_bundled_regressors_path

```python
def get_bundled_regressors_path(cluster_name: str = None) -> str
```

Get path to bundled regressor data.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `cluster_name` | `str` | Optional cluster name (e.g., 'Perlmutter', 'Vista') |

**Returns:** String path to regressors directory or cluster subdirectory.

## Usage Patterns

### Compare Parallelism Strategies

```python
from estimate_train_time import one_batch_predict
import yaml

def create_config(pp, mp, dp, output_path):
    """Create a config with specified parallelism."""
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
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    return output_path

# Compare different configurations (all using 16 GPUs)
strategies = [
    (4, 2, 2),  # pp=4, mp=2, dp=2
    (2, 4, 2),  # pp=2, mp=4, dp=2
    (2, 2, 4),  # pp=2, mp=2, dp=4
    (8, 1, 2),  # pp=8, mp=1, dp=2
]

for pp, mp, dp in strategies:
    config_path = create_config(pp, mp, dp, f"config_{pp}_{mp}_{dp}.yml")
    time_us = one_batch_predict(config_path)
    print(f"pp={pp}, mp={mp}, dp={dp}: {time_us/1e6:.2f}s per step")
```

### Batch Processing

```python
from estimate_train_time import one_batch_predict
from pathlib import Path
import csv

# Process all configs in a directory
config_dir = Path("./configs")
results = []

for config_path in config_dir.glob("*.yml"):
    try:
        time_us = one_batch_predict(str(config_path))
        results.append({
            'config': config_path.name,
            'time_us': time_us,
            'time_s': time_us / 1e6
        })
    except Exception as e:
        print(f"Error processing {config_path}: {e}")

# Save results to CSV
with open('results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['config', 'time_us', 'time_s'])
    writer.writeheader()
    writer.writerows(results)
```

### Training Time Estimation

```python
from estimate_train_time import one_batch_predict

# Model and training parameters
config_path = "llama_7b_config.yml"
total_tokens = 1_000_000_000_000  # 1T tokens
batch_size = 4
sequence_length = 4096
gradient_accumulation = 8

# Calculate steps
tokens_per_step = batch_size * sequence_length * gradient_accumulation
total_steps = total_tokens // tokens_per_step

# Predict time
time_per_step_us = one_batch_predict(config_path)
time_per_step_s = time_per_step_us / 1e6

# Calculate totals
total_time_s = time_per_step_s * total_steps
total_time_hours = total_time_s / 3600
total_time_days = total_time_hours / 24

print(f"Time per step: {time_per_step_s:.2f}s")
print(f"Total steps: {total_steps:,}")
print(f"Estimated training time: {total_time_days:.1f} days ({total_time_hours:.0f} hours)")
```

## See Also

- [Getting Started](getting-started.md) - Installation and first steps
- [Configuration Reference](configuration.md) - Config file parameters
- [CLI Reference](cli-reference.md) - Command-line interface
