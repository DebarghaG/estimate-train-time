# Basic Usage Examples

This guide shows common usage patterns for estimate-train-time.

## Using Bundled Examples

The package includes pre-configured examples for quick testing.

### List Available Examples

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

### View Example Configuration

```bash
estimate-train-time show-example llemma_7b_4_2_2_P
```

This displays the full YAML configuration, showing:
- GPU type and cluster
- Parallelism settings
- Model architecture parameters
- Operator list

### Run Prediction

```bash
estimate-train-time predict --example llemma_7b_4_2_2_P
```

Output:
```
Running prediction with config: /path/to/examples/llemma_7b_4_2_2_P.yml
----------------------------------------
Loading /path/to/regressors/NVIDIAA100-SXM4-80GB_embedding_fp16_fwd.json
Function:embedding_fp16_fwd    Input:[2, 4, 4096, 4096]    ...
...
Estimated time cost of current training config: 9480819.17 us
                                               = 9480.82 ms
                                               = 9.4808 s
```

## Interpreting Results

### Understanding the Output

The prediction output shows:

1. **Configuration path**: Which config file is being used
2. **Loading messages**: Each regressor model being loaded
3. **Per-operator predictions**: Individual timing for each operation
4. **Final estimate**: Total time in multiple units

### Time Units

| Unit | Description |
|------|-------------|
| us (microseconds) | Raw prediction value |
| ms (milliseconds) | us / 1,000 |
| s (seconds) | us / 1,000,000 |

### What the Time Represents

The predicted time is for **one complete training step**, which includes:
- Forward pass through all pipeline stages
- Backward pass with gradient computation
- Gradient synchronization (all-reduce)
- Optimizer step (parameter update)
- Parameter gathering (for ZeRO)

This corresponds to processing `batch_size × gradient_accumulation_steps` samples.

## Comparing Configurations

### Perlmutter vs Vista

Compare the same model on different clusters:

```bash
# A100 GPUs on Perlmutter
estimate-train-time predict -e llemma_7b_4_2_2_P

# GH200 GPUs on Vista
estimate-train-time predict -e llemma_7b_4_2_2_V
```

### Shell Script for Comparison

```bash
#!/bin/bash
# compare_examples.sh

echo "Comparing LLaMA 7B across clusters:"
echo "===================================="

for example in llemma_7b_4_2_2_P llemma_7b_4_2_2_V; do
    echo ""
    echo "Example: $example"
    estimate-train-time predict -e "$example" 2>/dev/null | tail -3
done
```

## Python Usage

### Basic Prediction

```python
from estimate_train_time import one_batch_predict

# Run prediction
time_us = one_batch_predict("config.yml")

# Display results
print(f"Time per step: {time_us:.2f} us")
print(f"             = {time_us/1000:.2f} ms")
print(f"             = {time_us/1e6:.4f} s")
```

### Using Bundled Examples

```python
from estimate_train_time import one_batch_predict
from estimate_train_time.data import get_examples_path
import os

# Get path to bundled examples
examples_dir = get_examples_path()
config_path = os.path.join(examples_dir, "llemma_7b_4_2_2_P.yml")

# Run prediction
time_us = one_batch_predict(str(config_path))
print(f"LLaMA 7B on Perlmutter: {time_us/1e6:.2f}s per step")
```

### Comparing Multiple Configurations

```python
from estimate_train_time import one_batch_predict
from estimate_train_time.data import get_examples_path
import os

examples_dir = get_examples_path()
examples = ['llemma_7b_4_2_2_P', 'llemma_7b_4_2_2_V']

print("Configuration Comparison")
print("=" * 50)

for name in examples:
    config_path = os.path.join(examples_dir, f"{name}.yml")
    time_us = one_batch_predict(str(config_path))
    print(f"{name}: {time_us/1e6:.2f}s per step")
```

## Training Time Estimation

### Calculate Full Training Time

```python
from estimate_train_time import one_batch_predict

# Configuration
config_path = "llama_7b_config.yml"

# Training parameters
total_tokens = 1_000_000_000_000  # 1T tokens (typical LLM pretraining)
batch_size = 4                     # Per-GPU batch size
sequence_length = 4096             # Context length
data_parallel = 2                  # DP degree
gradient_accumulation = 8          # Steps before update

# Calculate tokens per step
tokens_per_step = batch_size * sequence_length * data_parallel * gradient_accumulation
total_steps = total_tokens // tokens_per_step

print(f"Tokens per step: {tokens_per_step:,}")
print(f"Total steps: {total_steps:,}")

# Predict time per step
time_per_step_us = one_batch_predict(config_path)
time_per_step_s = time_per_step_us / 1e6

# Calculate total training time
total_seconds = time_per_step_s * total_steps
total_hours = total_seconds / 3600
total_days = total_hours / 24

print(f"\nTime per step: {time_per_step_s:.2f}s")
print(f"Total training time: {total_days:.1f} days ({total_hours:,.0f} hours)")
```

### GPU Hours Calculation

```python
from estimate_train_time import one_batch_predict

def calculate_gpu_hours(config_path, total_tokens, batch_size,
                        seq_length, dp, grad_accum, num_gpus):
    """Calculate total GPU hours for training."""

    # Tokens per step
    tokens_per_step = batch_size * seq_length * dp * grad_accum
    total_steps = total_tokens // tokens_per_step

    # Time prediction
    time_per_step_s = one_batch_predict(config_path) / 1e6

    # Total time
    total_hours = (time_per_step_s * total_steps) / 3600
    gpu_hours = total_hours * num_gpus

    return {
        'tokens_per_step': tokens_per_step,
        'total_steps': total_steps,
        'wall_clock_hours': total_hours,
        'gpu_hours': gpu_hours
    }

# Example usage
result = calculate_gpu_hours(
    config_path="llama_7b_config.yml",
    total_tokens=1e12,     # 1T tokens
    batch_size=4,
    seq_length=4096,
    dp=2,
    grad_accum=8,
    num_gpus=16            # pp * mp * dp = 4 * 2 * 2
)

print(f"Training requires {result['gpu_hours']:,.0f} GPU-hours")
print(f"Wall clock time: {result['wall_clock_hours']/24:.1f} days")
```

## Batch Processing

### Process Multiple Configs

```python
from estimate_train_time import one_batch_predict
from pathlib import Path
import json

def process_configs(config_dir, output_file):
    """Process all YAML configs in a directory."""

    results = []
    config_dir = Path(config_dir)

    for config_path in sorted(config_dir.glob("*.yml")):
        try:
            time_us = one_batch_predict(str(config_path))
            results.append({
                'config': config_path.name,
                'time_us': time_us,
                'time_s': time_us / 1e6,
                'status': 'success'
            })
            print(f"  {config_path.name}: {time_us/1e6:.2f}s")
        except Exception as e:
            results.append({
                'config': config_path.name,
                'error': str(e),
                'status': 'error'
            })
            print(f"  {config_path.name}: ERROR - {e}")

    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results

# Usage
results = process_configs("./configs/", "results.json")
```

### Export to CSV

```python
from estimate_train_time import one_batch_predict
from pathlib import Path
import csv

def export_results_csv(config_dir, output_csv):
    """Export prediction results to CSV."""

    config_dir = Path(config_dir)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'time_us', 'time_ms', 'time_s'])

        for config_path in sorted(config_dir.glob("*.yml")):
            try:
                time_us = one_batch_predict(str(config_path))
                writer.writerow([
                    config_path.name,
                    f"{time_us:.2f}",
                    f"{time_us/1000:.2f}",
                    f"{time_us/1e6:.4f}"
                ])
            except Exception as e:
                writer.writerow([config_path.name, 'ERROR', '', ''])

    print(f"Results saved to {output_csv}")

# Usage
export_results_csv("./configs/", "predictions.csv")
```

## Next Steps

- [Custom Configs](custom-configs.md) - Create your own configurations
- [Core Concepts](../concepts.md) - Understand the prediction model
- [Python API](../python-api.md) - Full API reference
