"""
Kernel sampling subpackage for GPU computation profiling.

This module requires GPU dependencies (torch, flash-attn, deepspeed).
Install with: pip install estimate-train-time[gpu]

Submodules:
    megatron_moe: MOE (Mixture of Experts) components for MOE model estimation
"""

from estimate_train_time.kernel_sampling import megatron_moe

__all__ = ["megatron_moe"]
