# estimate-train-time

[![PyPI version](https://badge.fury.io/py/estimate-train-time.svg)](https://badge.fury.io/py/estimate-train-time)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Predict distributed LLM training time before you run.** This tool estimates the wall-clock time for training large language models across multiple GPUs using 3D parallelism (pipeline, tensor, and data parallelism), helping you plan capacity and compare parallelization strategies without expensive trial runs.

## Installation

```bash
pip install estimate-train-time  # Coming soon to PyPI
```

**Note:** PyPI package is coming soon. For now, install directly from the repository:

```bash
git clone https://github.com/AI4CI/estimate-train-time.git
cd estimate-train-time
pip install -e .
```

## Quick Start

```bash
# List available example configurations
estimate-train-time list-examples

# Run prediction with a bundled example (Llama 7B on A100s)
estimate-train-time predict --example llemma_7b_4_2_2_P
```

Output:
```
Estimated time cost of current training config: 9480819.17 us
                                               = 9480.82 ms
                                               = 9.4808 s
```

## Features

- **3D Parallelism Support**: Pipeline, tensor (model), and data parallelism
- **Pre-trained Regressors**: Bundled models for NVIDIA A100 and GH200 GPUs
- **No GPU Required**: Predictions run on CPU using trained regressors
- **Extensible**: Add your own GPU profiles and cluster configurations

## Documentation

- [Getting Started](docs/getting-started.md) - Installation and first prediction
- [Core Concepts](docs/concepts.md) - Understanding distributed training estimation
- [Configuration Reference](docs/configuration.md) - Config file parameters
- [CLI Reference](docs/cli-reference.md) - Command-line options
- [Python API](docs/python-api.md) - Programmatic usage
- [Examples](docs/examples/) - Usage examples and custom configurations
- [Advanced](docs/advanced/) - Kernel sampling and extending the tool

## Python API

```python
from estimate_train_time import one_batch_predict

# Predict training time from a config file
time_us = one_batch_predict("path/to/config.yml")
print(f"One batch takes {time_us / 1e6:.2f} seconds")
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, pyyaml, ijson, joblib

For GPU sampling (optional): torch, flash-attn, deepspeed

## Acknowledgements

*National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)*

## License

MIT License - see [LICENSE](LICENSE) for details.
