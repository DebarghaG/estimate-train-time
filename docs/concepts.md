# Core Concepts

This document provides a deep technical explanation of distributed LLM training and how estimate-train-time models training time.

## What is Distributed Training Time Estimation?

Training large language models requires distributing computation across many GPUs. The wall-clock time depends on:

1. **Compute time**: How long GPU kernels take to execute
2. **Communication time**: How long data transfers take between GPUs
3. **Parallelization overhead**: Pipeline bubbles, synchronization delays

This tool predicts training time by:
1. Profiling individual operators on target hardware (sampling phase)
2. Training regressors to predict operator time from input shapes
3. Composing operator times into full training step estimates

## 3D Parallelism

Modern LLM training uses three orthogonal parallelism strategies:

```
                    ┌─────────────────────────────────────┐
                    │         Full Model + Data           │
                    └─────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
   │  Pipeline   │          │   Tensor    │          │    Data     │
   │ Parallelism │          │ Parallelism │          │ Parallelism │
   │    (PP)     │          │    (MP)     │          │    (DP)     │
   └─────────────┘          └─────────────┘          └─────────────┘
   Split by layers          Split tensors            Split batches
```

Total GPUs = PP × MP × DP

### Pipeline Parallelism (PP)

Pipeline parallelism divides the model into sequential stages, each running on a different GPU group.

#### Layer Distribution

For a model with `L` encoder layers and `PP` pipeline stages, layers are distributed as:

```
Stage 0 (Head):   Embedding + ceil(L/PP) encoder layers
Stage 1..PP-2:    floor(L/PP) encoder layers each
Stage PP-1 (Tail): floor(L/PP) encoder layers + LayerNorm + Output
```

#### Pipeline Schedules

**All-Forward-All-Backward (AF-AB):**

```
Stage 0: F₀ F₁ F₂ F₃ │ B₀ B₁ B₂ B₃
Stage 1: ── F₀ F₁ F₂ F₃ │ B₀ B₁ B₂ B₃
Stage 2: ── ── F₀ F₁ F₂ F₃ │ B₀ B₁ B₂ B₃
Stage 3: ── ── ── F₀ F₁ F₂ F₃ │ B₀ B₁ B₂ B₃

         ◄─── Forward ───►│◄─── Backward ───►
```

Time formula:
```
T_AFAB = (num_microbatches + PP - 1) × (T_fwd + T_bwd)
```

**1F1B (One Forward One Backward):**

```
Stage 0: F₀ F₁ F₂ F₃ B₀ B₁ B₂ B₃
Stage 1: ── F₀ F₁ F₂ B₀ F₃ B₁ B₂ B₃
Stage 2: ── ── F₀ F₁ B₀ F₂ B₁ F₃ B₂ B₃
Stage 3: ── ── ── F₀ B₀ F₁ B₁ F₂ B₂ F₃ B₃
```

Time formula:
```
T_1F1B = T_warmup + (num_microbatches - 1) × max(T_fwd, T_bwd) + T_cooldown
```

#### Pipeline Bubble

The pipeline bubble is idle time when stages wait for data:

```
Bubble fraction = (PP - 1) / (num_microbatches + PP - 1)
```

For PP=4 and 8 microbatches:
```
Bubble = (4 - 1) / (8 + 4 - 1) = 3/11 ≈ 27%
```

#### Point-to-Point Communication

Pipeline stages communicate activations via P2P send/recv:

```
Activation size = batch × sequence_length × hidden_dim / MP
                = b × l × d / mp  (in elements)
                = b × l × d / mp × 2  (in bytes, for FP16)
```

### Tensor Parallelism (MP)

Tensor parallelism (also called model parallelism) splits individual tensors across GPUs within a layer.

#### Column Parallelism

For linear layers with weight `W ∈ R^{d×4d}` (e.g., FFN up-projection):

```
            GPU 0              GPU 1
         ┌─────────┐        ┌─────────┐
    X ──►│ W[:, :2d]│   X ──►│W[:, 2d:]│
         └────┬────┘        └────┬────┘
              │                  │
              ▼                  ▼
           Y[:, :2d]          Y[:, 2d:]
```

Each GPU computes a partition of the output columns.

#### Row Parallelism

For linear layers with weight `W ∈ R^{4d×d}` (e.g., FFN down-projection):

```
            GPU 0              GPU 1
         ┌─────────┐        ┌─────────┐
  X[:,:2d]│W[:2d, :]│  X[:,2d:]│W[2d:, :]│
         └────┬────┘        └────┬────┘
              │                  │
              └────────┬────────┘
                       │ All-Reduce
                       ▼
                       Y
```

Each GPU computes a partial output that must be summed via all-reduce.

#### Communication Pattern

Per encoder layer with tensor parallelism:

```
Forward pass:
  - All-reduce after attention output projection
  - All-reduce after FFN down projection
  Total: 2 × all-reduce(b × l × d)

Backward pass:
  - All-reduce for attention input gradients
  - All-reduce for FFN input gradients
  Total: 2 × all-reduce(b × l × d)
```

All-reduce volume per layer:
```
V_mp = 2 × (fwd_syncs + bwd_syncs) × b × l × d × 2  (bytes, FP16)
```

### Data Parallelism (DP)

Data parallelism replicates the model and splits batches across GPUs.

#### Gradient Synchronization

After backward pass, gradients must be synchronized:

```
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │
│  ∇W₀    │  │  ∇W₁    │  │  ∇W₂    │  │  ∇W₃    │
└────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘
     │            │            │            │
     └────────────┴────────────┴────────────┘
                         │
                    All-Reduce
                         │
     ┌────────────┬────────────┬────────────┐
     │            │            │            │
     ▼            ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ ∇W_avg  │  │ ∇W_avg  │  │ ∇W_avg  │  │ ∇W_avg  │
└─────────┘  └─────────┘  └─────────┘  └─────────┘
```

#### ZeRO Optimization

ZeRO-1 partitions optimizer states across DP ranks:

```
Standard DP:                    ZeRO-1:
┌─────────────────┐            ┌─────────────────┐
│ GPU 0: Full Opt │            │ GPU 0: Opt[0:P/4]│
│ GPU 1: Full Opt │   ──►      │ GPU 1: Opt[P/4:P/2]│
│ GPU 2: Full Opt │            │ GPU 2: Opt[P/2:3P/4]│
│ GPU 3: Full Opt │            │ GPU 3: Opt[3P/4:P]│
└─────────────────┘            └─────────────────┘
Memory: 4 × Opt_size           Memory: Opt_size (distributed)
```

Communication pattern:
1. **All-reduce**: Average gradients across all DP ranks
2. **All-gather**: Collect updated parameters after optimizer step

Communication volume:
```
V_dp_allreduce = 2 × P / DP × (DP - 1) / DP  (ring all-reduce)
V_dp_allgather = P / DP × (DP - 1)           (ring all-gather)
```

Where P = total model parameters.

## Prediction Pipeline

The estimator predicts training time through this pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SAMPLING PHASE (offline)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Kernel     │    │    NCCL      │    │  Regressor   │      │
│  │  Profiling   │───►│  Benchmarks  │───►│  Training    │      │
│  │  (GPU ops)   │    │  (comms)     │    │  (RF/XGB)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREDICTION PHASE (runtime)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Config     │    │   Operator   │    │  Composition │      │
│  │   Parsing    │───►│  Prediction  │───►│    Model     │      │
│  │              │    │              │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
│                                │                                 │
│                                ▼                                 │
│                      ┌──────────────┐                           │
│                      │  Total Time  │                           │
│                      │  Estimate    │                           │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Sampling Methodology

#### Kernel Profiling

Each compute operator is profiled using PyTorch's profiler:

1. Generate input shapes covering the parameter space
2. Run operator with warmup iterations
3. Record GPU kernel execution times
4. Extract timing from profiler traces

Shape parameters for operators:
```
flash_attention: (mp, b, h, l, dim)
linear:          (mp, b, l, dim)
layernorm:       (b, l, dim)
embedding:       (mp, b, l, dim)
```

#### NCCL Benchmarking

Communication operations are profiled across GPU topologies:

```
Intra-node:  1 node × {2, 4, 8} GPUs
Inter-node:  2 nodes × {1, 2, 4} GPUs per node
```

Operations profiled:
- `allreduce`: Sum tensors across all ranks
- `allgather`: Concatenate tensors from all ranks
- `reduce_scatter`: Reduce and scatter result
- `p2p`: Point-to-point send/receive

### Regressor Training

For each operator, a regressor predicts execution time from input shape:

```python
# Training data: (input_shape, measured_time)
X = [[b, l, h, dim], ...]  # Input shapes
y = [time_us, ...]         # Measured times

# Model selection via hyperparameter search
models = [RandomForestRegressor, XGBRegressor]
best_model = hyperparameter_search(models, X, y)
```

The tool performs hyperparameter search over:
- Random Forest (n_estimators, max_depth, min_samples_split)
- XGBoost (n_estimators, max_depth, learning_rate)

Best model is selected based on validation performance.

### Composition Model

Individual operator times are composed into total training time:

```python
# Per-layer forward time
encoder_fwd = (layernorm + linear1 + attention + linear2 +
               layernorm + linear3 + activation + linear4)
encoder_fwd += fwd_syncs * mp_allreduce

# Per-layer backward time (includes forward recompute)
encoder_bwd = encoder_fwd + gradient_computation

# Full model forward/backward
head_fwd = embedding + head_layers * encoder_fwd
middle_fwd = middle_layers * encoder_fwd
tail_fwd = tail_layers * encoder_fwd + final_layernorm + output_linear + loss

# Pipeline scheduling
if schedule == 'AFAB':
    compute_time = all_F_all_B(fwd_times, bwd_times, microbatches, pp)

# Add communication
total_time = compute_time + dp_allreduce + optimizer_step + dp_allgather
```

## Compute Operators

### Embedding

Token embedding lookup and position encoding.

```
Input: token_ids ∈ Z^{b×l}
Output: embeddings ∈ R^{b×l×d}

Time complexity: O(b × l × d)
```

### Linear Layers

Four linear layer types in transformer architecture:

| Layer | Input Shape | Output Shape | FLOPs |
|-------|-------------|--------------|-------|
| QKV Projection (linear1) | (b×l, d) | (b×l, 3d/mp) | 6×b×l×d²/mp |
| Output Projection (linear2) | (b×l, d/mp) | (b×l, d) | 2×b×l×d²/mp |
| FFN Up (linear3) | (b×l, d) | (b×l, 4d/mp) | 8×b×l×d²/mp |
| FFN Down (linear4) | (b×l, 4d/mp) | (b×l, d) | 8×b×l×d²/mp |

### Attention (Flash Attention)

Flash Attention computes scaled dot-product attention with memory efficiency:

```
Q, K, V ∈ R^{b×l×h×(d/h)}

Attention(Q, K, V) = softmax(QK^T / √(d/h)) V

Time complexity: O(b × l² × d)  (but with better memory access patterns)
```

### Normalization

Layer normalization (standard or RMS):

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
RMSNorm(x) = γ × x / √(mean(x²) + ε)

Input: x ∈ R^{b×l×d}
Time complexity: O(b × l × d)
```

### Activation Functions

GELU activation for FFN:

```
GELU(x) = x × Φ(x)  where Φ is standard Gaussian CDF

Input: x ∈ R^{b×l×4d/mp}
Time complexity: O(b × l × d / mp)
```

## Communication Operators

### All-Reduce

Sums tensors across all ranks:

```
Input:  T_i on rank i
Output: Σ T_i on all ranks

Ring all-reduce time: 2 × (N-1)/N × S / B
Where N = ranks, S = tensor size, B = bandwidth
```

### All-Gather

Gathers tensors from all ranks:

```
Input:  T_i on rank i
Output: [T_0, T_1, ..., T_{N-1}] on all ranks

Ring all-gather time: (N-1)/N × S × N / B = (N-1) × S / B
```

### Point-to-Point

Direct send/receive between two ranks:

```
Send(T, dst): rank src → rank dst
Recv(T, src): rank dst ← rank src

P2P time: S / B + latency
```

## When to Use This Tool

### Capacity Planning

Estimate training time before committing resources:

```python
# How long will training take?
time_per_step = one_batch_predict("config.yml")
total_steps = tokens_to_train / tokens_per_step
total_hours = time_per_step * total_steps / 3.6e9
```

### Parallelism Strategy Search

Compare different parallelization strategies:

```python
strategies = [
    (4, 2, 2),  # More pipeline stages
    (2, 4, 2),  # More tensor parallelism
    (2, 2, 4),  # More data parallelism
]

for pp, mp, dp in strategies:
    config = create_config(pp, mp, dp)
    time = one_batch_predict(config)
    efficiency = compute_only_time / time
    print(f"PP={pp}, MP={mp}, DP={dp}: {time:.2f}us, efficiency={efficiency:.1%}")
```

### Hardware Comparison

Compare different GPU types or clusters:

```python
clusters = ['Perlmutter', 'Vista']
for cluster in clusters:
    config = f"configs/{cluster}_llama_70b.yml"
    time = one_batch_predict(config)
    print(f"{cluster}: {time/1e6:.2f}s per step")
```

## Limitations

1. **Profiling dependency**: Predictions are only as accurate as the sampling data
2. **Architecture assumptions**: Current model assumes standard transformer architecture
3. **Schedule simplification**: Uses all-forward-all-backward scheduling model
4. **Memory not modeled**: Does not predict OOM conditions
5. **Topology simplification**: Assumes uniform network topology

## See Also

- [Configuration Reference](configuration.md) - Config parameters
- [Advanced: Kernel Sampling](advanced/kernel-sampling.md) - Profile your own GPUs
- [Advanced: NCCL Sampling](advanced/nccl-sampling.md) - Communication profiling
