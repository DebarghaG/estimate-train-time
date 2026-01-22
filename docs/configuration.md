# Configuration Reference

This document describes all parameters in the configuration YAML files used by estimate-train-time.

## Configuration File Format

Configuration files use YAML syntax with Python-style comments. Example:

```yaml
{
  "gpu_name": "NVIDIAA100-SXM4-80GB",
  "operator_data_folder": "./regressors/Perlmutter/operator",
  "nccl_data_folder": "./regressors/Perlmutter/nccl",

  # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
  "comm_bucket": 1260000000,
  "encoders": 32,

  "function_list": ['embedding', 'RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add'],
  "encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'RMSlayernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add'],
  "layernorm_name": 'RMSlayernorm',

  "fwd_syncs": 2,
  "bwd_syncs": 2,
}
```

## Required Parameters

### gpu_name

**Type:** `string`

GPU identifier string matching the naming convention in your regressor data. This should match the output of `torch.cuda.get_device_name(0).replace(' ', '')`.

**Examples:**
- `"NVIDIAA100-SXM4-80GB"` - NVIDIA A100 SXM4 80GB
- `"NVIDIAGH200120GB"` - NVIDIA GH200 120GB

### operator_data_folder

**Type:** `string` (path)

Path to the directory containing compute operator regressors. Can be:
- Absolute path: `/path/to/regressors/operator`
- Relative to config file: `./regressors/Perlmutter/operator`
- Relative to bundled data: `./regressors/Perlmutter/operator` (automatically resolved)

This folder should contain:
- `{gpu_name}.csv` - Configuration file with best model parameters
- `{gpu_name}_{operator}_{precision}_{propagation}.json` or `.pkl` - Trained regressor models

### nccl_data_folder

**Type:** `string` (path)

Path to the directory containing communication (NCCL) operator regressors. Same path resolution rules as `operator_data_folder`.

This folder should contain:
- `{gpu_name}.csv` - Configuration file with best model parameters
- `{gpu_name}_{nccl_op}_{precision}_{nodes}_{gpus_per_node}.json` or `.pkl` - Trained regressor models

### training_config

**Type:** `list[int]` (9 elements)

Core training configuration as a list of 9 integers:

| Index | Parameter | Description |
|-------|-----------|-------------|
| 0 | `pp` | Pipeline parallelism degree |
| 1 | `mp` | Model (tensor) parallelism degree |
| 2 | `dp` | Data parallelism degree |
| 3 | `b` | Micro-batch size per GPU |
| 4 | `h` | Number of attention heads |
| 5 | `l` | Sequence length |
| 6 | `dim` | Hidden dimension size |
| 7 | `steps_per_update` | Gradient accumulation steps |
| 8 | `gpus_per_node` | GPUs per node in the cluster |

**Example:** `[4, 2, 2, 4, 32, 4096, 4096, 8, 4]`
- 4-way pipeline parallelism
- 2-way tensor parallelism
- 2-way data parallelism
- Batch size 4 per GPU
- 32 attention heads
- Sequence length 4096
- Hidden dimension 4096
- 8 gradient accumulation steps
- 4 GPUs per node

**Total GPUs:** `pp * mp * dp = 4 * 2 * 2 = 16 GPUs`

### comm_bucket

**Type:** `int`

Communication bucket size in bytes for gradient bucketing during data-parallel all-reduce. This affects how gradients are grouped for efficient communication.

**Typical values:**
- `1260000000` (1.26 GB) - Default for large models
- Smaller values for memory-constrained setups

### encoders

**Type:** `int`

Number of transformer encoder layers in the model.

**Examples:**
- LLaMA 7B: `32`
- LLaMA 13B: `40`
- LLaMA 70B: `80`

### function_list

**Type:** `list[string]`

List of all operators in the model that need timing prediction. These are profiled individually and combined to estimate total time.

**Available operators:**

| Operator | Description |
|----------|-------------|
| `embedding` | Token embedding lookup |
| `RMSlayernorm` | RMS Layer Normalization |
| `layernorm` | Standard Layer Normalization |
| `linear1` | QKV projection (attention) |
| `linear2` | Output projection (attention) |
| `linear3` | FFN up-projection |
| `linear4` | FFN down-projection |
| `linear_final` | Final output projection |
| `RoPE` | Rotary Position Embedding |
| `flash_atten` | Flash Attention |
| `gelu` | GELU activation |
| `res_add` | Residual addition |
| `parallel_cross_entropy_128` | Parallel cross-entropy loss |

### encoder_function_list

**Type:** `list[string]`

Ordered list of operators within a single transformer layer. This defines the computation graph for one encoder block.

**Example for LLaMA-style architecture:**
```yaml
['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2',
 'RMSlayernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add']
```

### layernorm_name

**Type:** `string`

The layer normalization variant used in the model.

**Values:**
- `'RMSlayernorm'` - RMS Layer Normalization (LLaMA, Mistral)
- `'layernorm'` - Standard Layer Normalization (GPT-2, BERT)

### fwd_syncs

**Type:** `int`

Number of synchronization points (all-reduce operations) during the forward pass per encoder layer. This accounts for tensor parallelism communication.

**Typical values:** `2` (one after attention, one after FFN)

### bwd_syncs

**Type:** `int`

Number of synchronization points during the backward pass per encoder layer.

**Typical values:** `2` (matching forward pass)

## Path Resolution

Data folder paths are resolved in this order:

1. **Absolute path**: If the path starts with `/` and exists, use it directly
2. **Relative to config**: `{config_directory}/{path}` if it exists
3. **Bundled data**: `{package_data_dir}/{path}` for bundled regressors

This allows configs to work both with local data and bundled package data.

## Complete Example

### LLaMA 7B on Perlmutter (A100)

```yaml
{
  "gpu_name": "NVIDIAA100-SXM4-80GB",
  "operator_data_folder": "./regressors/Perlmutter/operator",
  "nccl_data_folder": "./regressors/Perlmutter/nccl",

  # pp=4, mp=2, dp=2, b=4, h=32, l=4096, dim=4096, steps=8, gpus_per_node=4
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
  "comm_bucket": 1260000000,
  "encoders": 32,

  "function_list": ['embedding', 'RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add'],
  "encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'RMSlayernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add'],
  "layernorm_name": 'RMSlayernorm',

  "fwd_syncs": 2,
  "bwd_syncs": 2,
}
```

### LLaMA 7B on Vista (GH200)

```yaml
{
  "gpu_name": "NVIDIAGH200120GB",
  "operator_data_folder": "./regressors/Vista/operator",
  "nccl_data_folder": "./regressors/Vista/nccl",

  # pp=4, mp=2, dp=2, b=4, h=32, l=4096, dim=4096, steps=8, gpus_per_node=1
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 1],
  "comm_bucket": 1260000000,
  "encoders": 32,

  "function_list": ['embedding', 'RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add'],
  "encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'RMSlayernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add'],
  "layernorm_name": 'RMSlayernorm',

  "fwd_syncs": 2,
  "bwd_syncs": 2,
}
```

## Common Errors

### FileNotFoundError: Regressor data not found

The specified `operator_data_folder` or `nccl_data_folder` doesn't exist or doesn't contain the required files.

**Solutions:**
- Check path spelling and case sensitivity
- Ensure you're using bundled examples with `--example` flag
- For custom configs, verify regressor data exists

### KeyError: Unknown operator

An operator in `function_list` doesn't have a corresponding regressor.

**Solutions:**
- Check operator name spelling
- Use only operators that have been profiled
- See [Kernel Sampling](advanced/kernel-sampling.md) to add new operators

### Invalid training_config length

The `training_config` list must have exactly 9 elements.

**Solution:** Ensure all 9 parameters are specified: `[pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node]`

## See Also

- [CLI Reference](cli-reference.md) - Command-line options
- [Examples](examples/custom-configs.md) - Creating custom configurations
- [Advanced: Kernel Sampling](advanced/kernel-sampling.md) - Adding new operators
