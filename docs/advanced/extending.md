# Extending estimate-train-time

This guide explains how to add new capabilities to estimate-train-time, including new operators, GPU profiles, and cluster configurations.

## Overview

The tool can be extended in several ways:

1. **New Operators**: Add support for additional compute operations
2. **New GPU Types**: Profile and add support for different GPUs
3. **New Clusters**: Add network topology profiles for new clusters
4. **New Architectures**: Support different model architectures

## Adding New Operators

### Step 1: Implement the Operator

Create the operator function in `src/estimate_train_time/kernel_sampling/target_functions.py`:

```python
def my_new_operator(shapes, precision, device_num):
    """
    Profile a new compute operation.

    Args:
        shapes: Input shape parameters (e.g., [mp, b, l, dim])
        precision: 'fp16' or 'fp32'
        device_num: CUDA device index
    """
    mp, b, l, dim = shapes

    # Set dtype
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    device = f'cuda:{device_num}'

    # Create input tensor with gradient tracking
    x = torch.randn(b, l, dim, dtype=dtype, device=device, requires_grad=True)

    # Forward pass - your operation here
    y = my_custom_function(x)

    # Backward pass for gradient timing
    loss = y.sum()
    loss.backward()

    # Ensure GPU sync
    torch.cuda.synchronize()
```

### Step 2: Add Shape Transform Functions

In `src/estimate_train_time/estimator/encoder_config_to_layer_input.py`:

```python
def my_new_operator(input):
    """Map encoder config to operator input shape.

    Args:
        input: (mp, b, h, l, dim) - standard encoder config

    Returns:
        Shape array for the operator's input
    """
    mp, b, h, l, dim = input
    # Transform to operator-specific shape
    return np.array([mp, b, l, dim])
```

In `src/estimate_train_time/estimator/layer_input_to_predictor_input.py`:

```python
def my_new_operator(input):
    """Map operator input to predictor features.

    Args:
        input: Operator-specific input shape

    Returns:
        Feature array for the regressor
    """
    mp, b, l, dim = input
    # Transform to features that correlate with execution time
    # Common patterns: total elements, matrix dimensions
    return np.array([b * l, dim // mp])
```

### Step 3: Create Sampling Configuration

Create `src/estimate_train_time/data/configs/kernel_sampling/collect/my_new_operator.yml`:

```yaml
{
  "module_name": "target_functions",
  "kernel_name": "my_new_operator",
  "function_name": "my_new_operator",

  # Output columns: input params + timing
  "columns_name": ['mp', 'b', 'l', 'dim', 'F_dur(us)', 'B_dur(us)'],

  # Profiler trace targets - find these by running get_profiler mode
  # Multiple names per entry allow matching alternate kernel names
  "targets": [
    ['my_kernel_forward_name', 'aten::my_op'],  # Forward pass
    ['autograd::engine::evaluate_function: MyOpBackward']  # Backward pass
  ],

  "first_n_column": 4,

  # Parameter ranges
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

### Step 4: Run Profiling

```bash
cd src/estimate_train_time/kernel_sampling

# Test first
python sampling_controller.py \
    --config_path ../data/configs/kernel_sampling/collect/my_new_operator.yml \
    --precision fp16

# Full collection
python sampling_controller.py \
    --config_path ../data/configs/kernel_sampling/collect/my_new_operator.yml \
    --precision fp16 \
    --parts 4 --part 1
```

### Step 5: Update Prediction Config

Add the operator to your training config:

```yaml
{
  "function_list": [..., 'my_new_operator'],
  "encoder_function_list": [..., 'my_new_operator'],  # if used per-layer
}
```

## Adding New GPU Types

### Step 1: Profile Compute Operators

Profile all operators on the new GPU:

```bash
cd src/estimate_train_time/kernel_sampling

# Run for each operator
for op in embedding layernorm linear1 linear2 linear3 linear4 flash_atten gelu; do
    python sampling_controller.py \
        --config_path ../data/configs/kernel_sampling/collect/${op}.yml \
        --precision fp16
done
```

### Step 2: Profile Communication

```bash
cd src/estimate_train_time/nccl_sampling

# Intra-node configurations
torchrun --nnodes 1 --nproc_per_node 2 ... sampling_controller.py ...
torchrun --nnodes 1 --nproc_per_node 4 ... sampling_controller.py ...

# Inter-node configurations (if applicable)
# See NCCL Sampling guide for SLURM scripts
```

### Step 3: Organize Data

Create directory structure:

```
regressors/
└── MyNewGPU/
    ├── operator/
    │   ├── MyGPUName.csv           # Config with best model params
    │   ├── MyGPUName_embedding_fp16.csv
    │   ├── MyGPUName_embedding_fp16_fwd.json
    │   ├── MyGPUName_embedding_fp16_bwd.json
    │   └── ...
    └── nccl/
        ├── MyGPUName.csv
        ├── MyGPUName_allreduce_fp16_1_2.csv
        └── ...
```

### Step 4: Create Config File

Create `MyGPUName.csv` for operator folder:

```csv
Function,Precision,Propagation,Best_model,Best_config
embedding,fp16,fwd,xgboost,"{'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}"
embedding,fp16,bwd,rforest,"{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5}"
flash_atten,fp16,fwd,xgboost,"{'n_estimators': 150, 'max_depth': 6}"
...
```

### Step 5: Train Regressors

Regressors are trained automatically on first use. To pre-train:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pandas as pd
import joblib

def train_regressor(data_path, output_path):
    """Train and save optimal regressor."""
    df = pd.read_csv(data_path)

    X = df.iloc[:, :-2].values
    y_fwd = df.iloc[:, -2].values
    y_bwd = df.iloc[:, -1].values

    # Hyperparameter search
    models = {
        'rforest': (RandomForestRegressor(), {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
        }),
        'xgboost': (xgb.XGBRegressor(), {
            'n_estimators': [100, 150],
            'max_depth': [4, 6, 8],
        })
    }

    best_score = float('inf')
    best_model = None

    for name, (model, params) in models.items():
        grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error')
        grid.fit(X, y_fwd)

        if -grid.best_score_ < best_score:
            best_score = -grid.best_score_
            best_model = grid.best_estimator_

    # Save model
    if isinstance(best_model, xgb.XGBRegressor):
        best_model.save_model(f"{output_path}.json")
    else:
        joblib.dump(best_model, f"{output_path}.pkl")

    return best_model
```

### Step 6: Create Example Config

```yaml
{
  "gpu_name": "MyGPUName",  # Must match filename prefix
  "operator_data_folder": "./regressors/MyNewGPU/operator",
  "nccl_data_folder": "./regressors/MyNewGPU/nccl",
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
  ...
}
```

## Adding New Clusters

### Step 1: Characterize Network Topology

Document the cluster's network:
- Intra-node bandwidth (NVLink, PCIe)
- Inter-node bandwidth (InfiniBand, Ethernet)
- GPUs per node
- Nodes available

### Step 2: Profile Communication

Run NCCL sampling for all relevant topologies:

```bash
# Example for a cluster with 4 GPUs/node
# Intra-node
sbatch sample_1node_2gpu.sh
sbatch sample_1node_4gpu.sh

# Inter-node
sbatch sample_2node_1gpu.sh
sbatch sample_2node_2gpu.sh
sbatch sample_2node_4gpu.sh
```

### Step 3: Profile Compute (if different GPU)

If the cluster has a different GPU type, also run kernel sampling.

### Step 4: Create Cluster Directory

```
regressors/
└── MyCluster/
    ├── operator/
    │   └── ...  # Compute regressors
    └── nccl/
        └── ...  # Communication regressors
```

### Step 5: Add Example Configs

Create example configurations in `src/estimate_train_time/data/examples/`:

```yaml
# model_7b_mycluster.yml
{
  "gpu_name": "NVIDIAA100-SXM4-80GB",
  "operator_data_folder": "./regressors/MyCluster/operator",
  "nccl_data_folder": "./regressors/MyCluster/nccl",
  "training_config": [4, 2, 2, 4, 32, 4096, 4096, 8, 4],
  ...
}
```

## Supporting New Architectures

### Step 1: Identify New Operators

Compare the target architecture to supported operators:

| Architecture Feature | Existing Support | New Operator Needed |
|---------------------|------------------|---------------------|
| GQA (Grouped Query Attention) | No | Yes - `gqa_attention` |
| SwiGLU activation | No | Yes - `swiglu` |
| MoE (Mixture of Experts) | Partial | May need `moe_router` |

### Step 2: Implement and Profile

Follow the "Adding New Operators" section for each new operator.

### Step 3: Define Layer Structure

Create appropriate `encoder_function_list`:

```yaml
# Standard transformer
"encoder_function_list": ['layernorm', 'linear1', 'attention', 'linear2',
                          'layernorm', 'linear3', 'gelu', 'linear4', 'res_add']

# LLaMA-style (RMSNorm, SwiGLU)
"encoder_function_list": ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten',
                          'linear2', 'RMSlayernorm', 'res_add', 'linear3',
                          'swiglu', 'linear4', 'res_add']

# MoE architecture
"encoder_function_list": ['layernorm', 'attention', 'layernorm',
                          'moe_router', 'expert_linear1', 'activation',
                          'expert_linear2', 'res_add']
```

### Step 4: Adjust Parallelism Model

For significantly different architectures (e.g., MoE), you may need to modify `prediction.py` to handle:
- Expert parallelism
- Load balancing overhead
- Additional communication patterns

## Contributing Back

### Code Style

- Follow existing code patterns
- Add docstrings to new functions
- Include type hints where practical

### Testing

Test your additions:

```python
# Test operator shape transforms
from estimate_train_time.estimator import encoder_config_to_layer_input as e2l
from estimate_train_time.estimator import layer_input_to_predictor_input as l2p

encoder_config = (2, 4, 32, 4096, 4096)  # mp, b, h, l, dim
layer_input = e2l.my_new_operator(encoder_config)
predictor_input = l2p.my_new_operator(layer_input)
print(f"Layer input: {layer_input}")
print(f"Predictor input: {predictor_input}")
```

### Documentation

- Update relevant docs for new features
- Add examples showing usage
- Document any limitations

### Pull Request

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit PR with description

## Directory Structure Reference

```
estimate-train-time/
├── src/estimate_train_time/
│   ├── __init__.py              # Public API
│   ├── cli.py                   # CLI implementation
│   ├── estimator/
│   │   ├── prediction.py        # Core prediction logic
│   │   ├── mml_3d_prediction.py # 3D parallelism prediction
│   │   ├── predictor.py         # Predictor class
│   │   ├── encoder_config_to_layer_input.py
│   │   ├── layer_input_to_predictor_input.py
│   │   └── tools.py
│   ├── kernel_sampling/
│   │   ├── sampling_controller.py
│   │   ├── target_functions.py  # Operator implementations
│   │   └── megatron_moe/        # MoE implementations
│   ├── nccl_sampling/
│   │   ├── sampling_controller.py
│   │   └── nccl_functions.py    # Communication operations
│   └── data/
│       ├── configs/
│       │   ├── kernel_sampling/ # Kernel sampling configs
│       │   │   ├── collect/
│       │   │   ├── profilers/
│       │   │   └── test/
│       │   └── nccl_sampling/   # NCCL sampling configs
│       │       └── test/
│       ├── examples/            # Bundled example configs
│       └── regressors/          # Bundled regressor data
└── scripts/
    ├── kernel_sampling/         # Shell scripts for kernel profiling
    └── nccl_sampling/           # Shell scripts for NCCL profiling
```

## See Also

- [Kernel Sampling](kernel-sampling.md) - Detailed profiling guide
- [NCCL Sampling](nccl-sampling.md) - Communication profiling
- [Configuration Reference](../configuration.md) - Config parameters
- [Core Concepts](../concepts.md) - Understanding the prediction model
