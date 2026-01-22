# CLI Reference

Complete reference for the `estimate-train-time` command-line interface.

## Synopsis

```bash
estimate-train-time [--version] [--help] <command> [<args>]
```

## Global Options

### --version

Display the package version and exit.

```bash
estimate-train-time --version
```

Output:
```
estimate-train-time 0.1.0
```

### --help, -h

Display help information.

```bash
estimate-train-time --help
```

Output:
```
usage: estimate-train-time [-h] [--version] {predict,list-examples,show-example} ...

Distributed training time estimator for Large Language Models

positional arguments:
  {predict,list-examples,show-example}
                        Available commands
    predict             Run time estimation prediction
    list-examples       List available example configurations
    show-example        Show an example configuration

optional arguments:
  -h, --help            show this help message and exit
  --version             Show version and exit
```

## Commands

### predict

Run training time prediction with a configuration file.

```bash
estimate-train-time predict [--config PATH | --example NAME]
```

#### Options

| Option | Short | Type | Description |
|--------|-------|------|-------------|
| `--config` | `-c` | PATH | Path to a YAML configuration file |
| `--example` | `-e` | NAME | Name of a bundled example configuration |

**Note:** You must specify either `--config` or `--example`, but not both.

#### Examples

Using a custom config file:
```bash
estimate-train-time predict --config /path/to/my_config.yml
estimate-train-time predict -c ./configs/llama_70b.yml
```

Using a bundled example:
```bash
estimate-train-time predict --example llemma_7b_4_2_2_P
estimate-train-time predict -e llemma_7b_4_2_2_V
```

#### Output

The command outputs:
1. Path to the configuration being used
2. Loading messages for each regressor model
3. Per-operator predictions with input shapes
4. Final time estimate in microseconds, milliseconds, and seconds

```
Running prediction with config: /path/to/config.yml
----------------------------------------
Loading /path/to/regressors/NVIDIAA100-SXM4-80GB_embedding_fp16_fwd.json
Function:embedding_fp16_fwd    Input:[2, 4, 4096, 4096]    PredictorInput:[16384, 400, 4096]    Prediction:123.45
...

Estimated time cost of current training config: 9480819.17 us
                                               = 9480.82 ms
                                               = 9.4808 s
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (config not found, invalid config, prediction failed) |

### list-examples

List all available bundled example configurations.

```bash
estimate-train-time list-examples
```

#### Output

```
Available example configurations:
----------------------------------------
  llemma_7b_4_2_2_P
  llemma_7b_4_2_2_V

Use 'estimate-train-time show-example <name>' to view a configuration.
Use 'estimate-train-time predict --example <name>' to run prediction.
```

### show-example

Display the contents of a bundled example configuration.

```bash
estimate-train-time show-example <name>
```

#### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `name` | NAME | Name of the example configuration (without .yml extension) |

#### Example

```bash
estimate-train-time show-example llemma_7b_4_2_2_P
```

#### Output

```
# Configuration: llemma_7b_4_2_2_P
# Path: /path/to/examples/llemma_7b_4_2_2_P.yml
----------------------------------------
{
  "gpu_name": "NVIDIAA100-SXM4-80GB",
  "operator_data_folder": "./regressors/Perlmutter/operator",
  "nccl_data_folder": "./regressors/Perlmutter/nccl",

  # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
  ...
}
```

#### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Example not found |

## Usage Patterns

### Quick Evaluation

Test with bundled examples:

```bash
# See what's available
estimate-train-time list-examples

# Run prediction
estimate-train-time predict -e llemma_7b_4_2_2_P
```

### Custom Configuration

Use your own config file:

```bash
# Create config from template
estimate-train-time show-example llemma_7b_4_2_2_P > my_config.yml

# Edit my_config.yml to match your setup
# ...

# Run prediction
estimate-train-time predict -c my_config.yml
```

### Comparing Configurations

Run multiple predictions to compare:

```bash
# Compare Perlmutter vs Vista
estimate-train-time predict -e llemma_7b_4_2_2_P
estimate-train-time predict -e llemma_7b_4_2_2_V

# Compare different parallelism strategies
estimate-train-time predict -c configs/pp4_mp2_dp2.yml
estimate-train-time predict -c configs/pp2_mp4_dp2.yml
estimate-train-time predict -c configs/pp2_mp2_dp4.yml
```

### Scripting

Use in shell scripts:

```bash
#!/bin/bash
# compare_configs.sh

for config in configs/*.yml; do
    echo "Testing: $config"
    estimate-train-time predict -c "$config" 2>/dev/null | tail -3
    echo "---"
done
```

### Integration with Python

For more control, use the Python API instead:

```python
from estimate_train_time import one_batch_predict

configs = ['config_a.yml', 'config_b.yml', 'config_c.yml']
for config in configs:
    time_us = one_batch_predict(config)
    print(f"{config}: {time_us/1e6:.2f}s per step")
```

## Environment Variables

The CLI does not currently use environment variables. All configuration is done through command-line arguments and YAML config files.

## See Also

- [Getting Started](getting-started.md) - First steps with the tool
- [Configuration Reference](configuration.md) - Config file format
- [Python API](python-api.md) - Programmatic usage
