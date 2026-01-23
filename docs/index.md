# estimate-train-time Documentation

Welcome to the documentation for estimate-train-time, a distributed LLM training time estimator.

## What is estimate-train-time?

This tool predicts the wall-clock time required to train large language models across multiple GPUs. It helps you:

- **Plan capacity**: Know how long training will take before starting
- **Compare strategies**: Evaluate different parallelization configurations
- **Optimize costs**: Find the most efficient GPU allocation

The tool works by combining profiled operator timings with an analytical model of distributed training, supporting pipeline, tensor, and data parallelism (3D parallelism).

## Quick Navigation

### Getting Started

| Document | Description |
|----------|-------------|
| [Getting Started](getting-started.md) | Installation and first prediction |
| [Quick Start](#quick-start) | Minimal example to try now |

### Reference

| Document | Description |
|----------|-------------|
| [Configuration Reference](configuration.md) | Complete config file parameters |
| [CLI Reference](cli-reference.md) | Command-line options |
| [Python API](python-api.md) | Programmatic usage |

### Learning

| Document | Description |
|----------|-------------|
| [Core Concepts](concepts.md) | Deep dive into distributed training and the prediction model |

### Examples

| Document | Description |
|----------|-------------|
| [Basic Usage](examples/basic-usage.md) | Common usage patterns |
| [Custom Configs](examples/custom-configs.md) | Creating your own configurations |

### Advanced

| Document | Description |
|----------|-------------|
| [Kernel Sampling](advanced/kernel-sampling.md) | Profile GPU compute operators |
| [NCCL Sampling](advanced/nccl-sampling.md) | Profile multi-GPU communication |
| [Extending](advanced/extending.md) | Add new operators and GPU profiles |

## Quick Start

Install the package:

```bash
pip install estimate-train-time  # Coming soon to PyPI
```

**Note:** PyPI package is coming soon. For now, install directly from the repository:

```bash
git clone https://github.com/DebarghaG/estimate-train-time.git
cd estimate-train-time
pip install -e .
```

Run a prediction with a bundled example:

```bash
estimate-train-time predict --example llemma_7b_4_2_2_P
```

Or use Python:

```python
from estimate_train_time import one_batch_predict

time_us = one_batch_predict("config.yml")
print(f"One training step takes {time_us / 1e6:.2f} seconds")
```

## For Different User Types

### Data Scientists / ML Engineers

You want to estimate training time for capacity planning:

1. Start with [Getting Started](getting-started.md)
2. Use [bundled examples](examples/basic-usage.md) or create [custom configs](examples/custom-configs.md)
3. Read [Core Concepts](concepts.md) to understand the parallelism parameters

### Infrastructure Engineers

You want to profile new hardware or optimize cluster configurations:

1. Read [Core Concepts](concepts.md) for background
2. Follow [Kernel Sampling](advanced/kernel-sampling.md) to profile GPUs
3. Follow [NCCL Sampling](advanced/nccl-sampling.md) for communication benchmarks
4. See [Extending](advanced/extending.md) to add new cluster profiles

### Researchers

You want to understand or extend the prediction model:

1. Deep dive into [Core Concepts](concepts.md) for the mathematical model
2. Review the source code for prediction logic
3. See [Extending](advanced/extending.md) for adding new operators

## Bundled Examples

The package includes pre-configured examples:

| Example | GPU | Parallelism | Model |
|---------|-----|-------------|-------|
| `llemma_7b_4_2_2_P` | A100 (Perlmutter) | PP=4, MP=2, DP=2 | LLaMA 7B |
| `llemma_7b_4_2_2_V` | GH200 (Vista) | PP=4, MP=2, DP=2 | LLaMA 7B |

List available examples:
```bash
estimate-train-time list-examples
```

View an example:
```bash
estimate-train-time show-example llemma_7b_4_2_2_P
```

## Feature Overview

### Supported Parallelism

- **Pipeline Parallelism (PP)**: Split model layers across stages
- **Tensor Parallelism (MP)**: Split tensors within layers
- **Data Parallelism (DP)**: Split batches across replicas

### Supported Operators

Compute operators:
- Embedding, LayerNorm (standard and RMS)
- Linear projections (QKV, output, FFN up/down)
- Flash Attention
- GELU activation
- Cross-entropy loss

Communication operators:
- All-reduce (tensor parallel, data parallel)
- All-gather (ZeRO optimizer)
- Point-to-point (pipeline parallel)

### Bundled GPU Profiles

Pre-trained regressors included for:
- NVIDIA A100-SXM4-80GB (Perlmutter cluster)
- NVIDIA GH200-120GB (Vista cluster)

## Support

- **Issues**: [GitHub Issues](https://github.com/DebarghaG/estimate-train-time/issues)
- **Repository**: [GitHub](https://github.com/DebarghaG/estimate-train-time)

## Acknowledgements

This project is funded by the National Science Foundation (NSF) AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606).
