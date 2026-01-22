# NCCL Sampling Guide

This guide explains how to profile multi-GPU communication operations using NCCL to create custom regressor data.

## Overview

NCCL (NVIDIA Collective Communications Library) sampling profiles communication operations:
- All-reduce (gradient and activation synchronization)
- All-gather (ZeRO parameter gathering)
- Reduce-scatter (ZeRO gradient distribution)
- Point-to-point (pipeline parallelism)

These profiles capture the relationship between message size, topology, and transfer time.

## Prerequisites

### Hardware

- Multi-GPU system (2+ GPUs)
- For inter-node: multiple nodes with InfiniBand/high-speed networking

### Software

```bash
pip install estimate-train-time[gpu]
```

Additional requirements:
- PyTorch with distributed support
- NCCL library (typically included with PyTorch)
- torchrun (distributed launcher)

### Cluster Environment

For SLURM-based clusters:
- Access to multiple nodes
- Ability to run `srun` and `torchrun`
- Network connectivity between nodes

## Sampling Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      NCCL Sampling Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│     Node 0                              Node 1                  │
│  ┌─────────────┐                     ┌─────────────┐           │
│  │   GPU 0     │◄──── Network ─────►│   GPU 0     │           │
│  │   GPU 1     │     (InfiniBand)   │   GPU 1     │           │
│  └─────────────┘                     └─────────────┘           │
│        │                                   │                    │
│        └──────────────┬──────────────────┘                    │
│                       ▼                                        │
│              ┌──────────────┐                                  │
│              │  Collective  │                                  │
│              │  Operation   │                                  │
│              │  + Profiling │                                  │
│              └──────────────┘                                  │
│                       │                                        │
│                       ▼                                        │
│              ┌──────────────┐                                  │
│              │   Timing     │                                  │
│              │   Results    │                                  │
│              └──────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Configuration Files

### Directory Structure

```
NCCL_sampling/
├── configs/
│   ├── collect/     # Full parameter sweeps
│   │   ├── allreduce.yml
│   │   ├── allgather.yml
│   │   ├── reducescatter.yml
│   │   └── p2p.yml
│   └── test/        # Quick tests
│       └── ...
├── nccl_functions.py
├── sampling_controller.py
└── tensor_shape_generator.py
```

### Configuration Structure

Example: `allreduce.yml`

```yaml
{
  # Module and function
  "module_name": "nccl_functions",
  "kernel_name": "allreduce",
  "function_name": "allreduce",

  # Output columns
  "columns_name": ['shape', 'nodes', 'gpus_per_node', 'dur(us)'],

  # Profiler targets
  "targets": [['ncclKernel_AllReduce']],

  # Parameters
  "first_n_column": 1,

  # Shape range (tensor size in elements)
  "starts": [20971520],        # ~20M elements
  "steps": [65536],            # Step size
  "ends": [134217728],         # ~134M elements
  "operators": ['add'],

  "warmup_shapes": [1048576],  # 1M elements for warmup

  "wait": 2,
  "warmup": 2,
  "active": 10,

  "shapes": [20971520],
  "run": "collecter"
}
```

### Shape Ranges

Different operations use different shape ranges:

| Operation | Start | End | Use Case |
|-----------|-------|-----|----------|
| allreduce (small) | 20M | 134M | Tensor parallel activation sync |
| allreduce (large) | 134M | 1.2B | Data parallel gradient sync |
| allgather | 134M | 1.2B | ZeRO parameter gathering |
| p2p | 20M | 134M | Pipeline parallelism |

## Running Sampling

### Intra-Node (Single Node, Multiple GPUs)

Test within one node:

```bash
cd NCCL_sampling

# 2 GPUs on 1 node
torchrun \
    --nnodes 1 \
    --nproc_per_node 2 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 \
    sampling_controller.py \
    --config_path ./configs/test/allreduce.yml \
    --precision fp16 \
    --parts 1 \
    --part 1

# 4 GPUs on 1 node
torchrun \
    --nnodes 1 \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 \
    sampling_controller.py \
    --config_path ./configs/test/allreduce.yml \
    --precision fp16 \
    --parts 1 \
    --part 1
```

### Inter-Node (Multiple Nodes)

For multi-node profiling with SLURM:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=6:00:00
#SBATCH --job-name=nccl_sampling

cd NCCL_sampling

# Get master node info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

NNODES=$(scontrol show hostname $SLURM_NODELIST | wc -l)

# Run with 4 GPUs per node across 2 nodes
srun --export=ALL torchrun \
    --nnodes $NNODES \
    --nproc_per_node 4 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    sampling_controller.py \
    --config_path ./configs/collect/allreduce.yml \
    --precision fp16 \
    --parts 1 \
    --part 1
```

### Point-to-Point (Pipeline Parallel Simulation)

P2P requires exactly 2 processes:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00:00

cd NCCL_sampling

# Get master info
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

NNODES=2

# Only GPU 0 on each node (simulates pipeline stages)
srun --export=ALL,CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes $NNODES \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    sampling_controller.py \
    --config_path ./configs/collect/p2p.yml \
    --precision fp16 \
    --parts 1 \
    --part 1
```

## Communication Operations

### All-Reduce

Sums tensors across all ranks. Used for:
- Tensor parallel activation synchronization
- Data parallel gradient synchronization

```python
# nccl_functions.py
def allreduce(shapes, precision):
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    local_rank = int(os.environ['LOCAL_RANK'])

    tensor = torch.rand(shapes, dtype=dtype, device=local_rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
```

### All-Gather

Gathers tensors from all ranks. Used for ZeRO optimizer parameter gathering.

```python
def allgather(shapes, precision):
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    tensor = torch.rand(shapes, dtype=dtype, device=local_rank)
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
```

### Reduce-Scatter

Reduces and scatters result. Used for ZeRO gradient distribution.

```python
def reducescatter(shapes, precision):
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    chunk_size = shapes[0] // world_size
    input_tensor = torch.rand(shapes, dtype=dtype, device=local_rank)
    output_tensor = torch.zeros([chunk_size], dtype=dtype, device=local_rank)
    input_list = list(input_tensor.chunk(world_size))

    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
```

### Point-to-Point

Direct send/receive between two ranks. Used for pipeline parallelism.

```python
def p2p(shapes, precision):
    dtype = torch.float16 if precision == 'fp16' else torch.float32
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    tensor = torch.rand(shapes, dtype=dtype, device=local_rank)

    if global_rank == 0:
        dist.send(tensor=tensor, dst=world_size-1)
    elif global_rank == world_size-1:
        dist.recv(tensor=tensor, src=0)
```

## Topology Configurations

### Sampling Matrix

Profile these topology combinations:

| Config | Nodes | GPUs/Node | Total GPUs | Use Case |
|--------|-------|-----------|------------|----------|
| 1-2 | 1 | 2 | 2 | Intra-node TP |
| 1-4 | 1 | 4 | 4 | Intra-node TP/DP |
| 1-8 | 1 | 8 | 8 | Full node |
| 2-1 | 2 | 1 | 2 | Inter-node PP |
| 2-2 | 2 | 2 | 4 | Inter-node TP |
| 2-4 | 2 | 4 | 8 | Multi-node training |

### Naming Convention

Output files follow this pattern:
```
{GPU_NAME}_{operation}_{precision}_{nodes}_{gpus_per_node}.csv
```

Example: `NVIDIAA100-SXM4-80GB_allreduce_fp16_2_4.csv`

## Output Files

### Raw Data

CSV files in `NCCL_sampling/sampling_data/`:

```csv
shape,nodes,gpus_per_node,dur(us)
20971520,2,4,1234.5
21037056,2,4,1240.2
...
```

### Progress Logs

Log files in `NCCL_sampling/sampling_log/`:

```
Kernel: allreduce_fp16 | Progress: 500/2000=25.00% | Speed: 0.45s/sample | Remaining: 0d0h11m15s
```

## Processing Results

### Merge Data Files

```python
import pandas as pd
import glob

# Merge all allreduce data for 2 nodes, 4 GPUs
pattern = "sampling_data/*_allreduce_fp16_2_4_*.csv"
files = glob.glob(pattern)

dfs = [pd.read_csv(f) for f in files]
merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates()
merged = merged.sort_values('shape')

merged.to_csv("NVIDIAA100-SXM4-80GB_allreduce_fp16_2_4.csv", index=False)
```

### Organize for Estimator

The estimator expects files in this structure:

```
regressors/
└── MyCluster/
    └── nccl/
        ├── NVIDIAA100-SXM4-80GB.csv          # Config file
        ├── NVIDIAA100-SXM4-80GB_allreduce_fp16_1_2.csv
        ├── NVIDIAA100-SXM4-80GB_allreduce_fp16_1_4.csv
        ├── NVIDIAA100-SXM4-80GB_allreduce_fp16_2_1.csv
        ├── NVIDIAA100-SXM4-80GB_allreduce_large_fp16_2_4.csv
        ├── NVIDIAA100-SXM4-80GB_allgather_large_fp16_2_4.csv
        └── NVIDIAA100-SXM4-80GB_p2p_fp16_2_1.csv
```

### Config File Format

Create `{GPU_NAME}.csv` with regressor configurations:

```csv
Function,Precision,Nodes,GPUsPerNode,Best_model,Best_config
allreduce,fp16,1,2,xgboost,"{'n_estimators': 100, 'max_depth': 5}"
allreduce,fp16,1,4,rforest,"{'n_estimators': 200, 'max_depth': 10}"
allreduce_large,fp16,2,4,xgboost,"{'n_estimators': 150, 'max_depth': 6}"
...
```

## Full Sampling Workflow

### Step 1: Intra-Node Sampling

```bash
# 1 node, 2 GPUs
sbatch scripts/sample_intra_1n2g.sh

# 1 node, 4 GPUs
sbatch scripts/sample_intra_1n4g.sh
```

### Step 2: Inter-Node Sampling

```bash
# 2 nodes, 1 GPU each
sbatch scripts/sample_inter_2n1g.sh

# 2 nodes, 4 GPUs each
sbatch scripts/sample_inter_2n4g.sh
```

### Step 3: Merge and Process

```python
# merge_nccl_data.py
import pandas as pd
import glob
import os

output_dir = "regressors/MyCluster/nccl"
os.makedirs(output_dir, exist_ok=True)

# Operations and topologies
operations = ['allreduce', 'allreduce_large', 'allgather_large', 'p2p']
topologies = [(1, 2), (1, 4), (2, 1), (2, 4)]

for op in operations:
    for nodes, gpus in topologies:
        pattern = f"sampling_data/*_{op}_fp16_{nodes}_{gpus}_*.csv"
        files = glob.glob(pattern)
        if files:
            dfs = [pd.read_csv(f) for f in files]
            merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
            output = f"{output_dir}/NVIDIAA100-SXM4-80GB_{op}_fp16_{nodes}_{gpus}.csv"
            merged.to_csv(output, index=False)
            print(f"Created: {output}")
```

### Step 4: Train Regressors

Regressors are trained automatically on first use, or pre-train with hyperparameter search.

## Troubleshooting

### Connection Timeout

```
RuntimeError: Timed out initializing process group
```

Solutions:
- Increase timeout: `NCCL_ASYNC_ERROR_HANDLING=1`
- Check network connectivity between nodes
- Verify firewall allows NCCL ports

### Wrong GPU Count

```
ValueError: Expected 4 GPUs but found 2
```

Check `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` match.

### NCCL Error

```
NCCL error: unhandled cuda error
```

Try:
- `export NCCL_DEBUG=INFO` for detailed logs
- Check CUDA/driver compatibility
- Verify GPU memory availability

## See Also

- [Kernel Sampling](kernel-sampling.md) - Compute operator profiling
- [Extending](extending.md) - Adding new capabilities
- [Core Concepts](../concepts.md) - Understanding communication patterns
