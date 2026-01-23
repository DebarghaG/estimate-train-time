"""
Megatron MOE (Mixture of Experts) subpackage for MOE model training estimation.

This module provides MOE-specific components for kernel sampling and profiling.
Requires GPU dependencies (torch, megablocks).
"""

# Core MOE classes
from estimate_train_time.kernel_sampling.megatron_moe.moe import (
    ParallelDroplessMLP,
    ParallelDroplessMoE,
)

# Router implementations
from estimate_train_time.kernel_sampling.megatron_moe.router import (
    SinkhornRouter,
    TopKTokenChoiceRouter,
)

# MLP variants
from estimate_train_time.kernel_sampling.megatron_moe.moe_mlp import (
    ParallelGroupedMLP,
    ParallelGroupedLLaMAMLP,
)

# Utilities
from estimate_train_time.kernel_sampling.megatron_moe.activations import get_activation
from estimate_train_time.kernel_sampling.megatron_moe.moe_tools import get_moe_object, initial_moe
from estimate_train_time.kernel_sampling.megatron_moe.neoxargs import NeoXArgs

__all__ = [
    "ParallelDroplessMLP",
    "ParallelDroplessMoE",
    "SinkhornRouter",
    "TopKTokenChoiceRouter",
    "ParallelGroupedMLP",
    "ParallelGroupedLLaMAMLP",
    "get_activation",
    "get_moe_object",
    "initial_moe",
    "NeoXArgs",
]
