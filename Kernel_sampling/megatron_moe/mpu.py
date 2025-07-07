import torch
from typing import Optional

from . import globals

# import deepspeed
# import deepspeed.runtime.activation_checkpointing.checkpointing as checkpointing

#--------------------------------------------------------
def get_expert_token_counts_for_rank(tokens_per_expert: torch.Tensor, rank: Optional[int] = None):
    """
    Allow user to specify rank, fall back on this device
    """
    # TODO: add bounds checking of size is 1D for tokens_per_expert
    # should be (num_experts) long
    world_size = globals.MP_WORLD_SIZE
    if rank is None:
        rank = globals.MP_RANK

    return tokens_per_expert.chunk(world_size)[rank]
#--------------------------------------------------------

#--------------------------------------------------------
def get_expert_tokens_for_rank(routed_tokens: torch.Tensor, tokens_per_expert: torch.Tensor, rank: Optional[int] = None,):
    """
    Allow user to specify rank, fall back on this device
    """
    # Calculate cumulative sums of tokens_per_expert, ensure the shapes are correct
    world_size = globals.MP_WORLD_SIZE
    if rank is None:
        rank = globals.MP_RANK

    # TODO: is this check necessary here/what does it cost us to redundantly do it in multiple places?
    assert tokens_per_expert.shape[0] % world_size == 0

    cumulative_sums = torch.cumsum(tokens_per_expert, dim=0)
    assert cumulative_sums[-1] == routed_tokens.shape[0]

    # select the right starting and ending indices from the cumsum to figure out what tokens to select
    rank_expert_indices = cumulative_sums.chunk(world_size)
    start_index = rank_expert_indices[rank - 1][-1] if rank > 0 else 0
    end_index = rank_expert_indices[rank][-1]

    # Use indices to select the chunk of the tokens matrix
    selected_experts = routed_tokens[start_index:end_index]

    return selected_experts
#--------------------------------------------------------



#--------------------------------------------------------
def copy_to_expert_model_parallel_region(input_, tokens_per_expert):
    return _CopyToExpertModelParallelRegion.apply(input_, tokens_per_expert)


class _CopyToExpertModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, tokens_per_expert):
        # TODO: not sure if this is sufficient? not sure how this gets used downstream...
        return get_expert_tokens_for_rank(input_, tokens_per_expert)

    @staticmethod
    def forward(ctx, input_, tokens_per_expert):
        # Save tokens_per_expert in the context for later use in the backward pass
        ctx.save_for_backward(tokens_per_expert)

        return get_expert_tokens_for_rank(input_, tokens_per_expert)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the tokens_per_expert from the context
        (tokens_per_expert,) = ctx.saved_tensors

        # no grad for tokens_per_expert
        # return _dmoe_reduce(grad_output, tokens_per_expert), None
        return _dmoe_gather(grad_output, tokens_per_expert), None


def _dmoe_gather(input_: torch.Tensor, tokens_per_expert: torch.Tensor):
    """Gather tensors and concatinate along the first dimension)"""

    world_size = globals.MP_WORLD_SIZE
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Bf16 convert
    dt = input_.dtype

    # print(f'_dmoe_gather input shape: {input_.shape}')

    # if dt == torch.bfloat16 and get_fp32_allreduce():
    if dt == torch.bfloat16:
        input_ = input_.float()


    # Gather along first dimension
    gather_dim = 0
    rank = globals.MP_RANK

    # tokens_by_rank = [
    #     get_expert_token_counts_for_rank(tokens_per_expert, r)
    #     for r in range(world_size)
    # ]
    # # print(f"{torch.cuda.current_device()}: tokens_by_rank {tokens_by_rank}")
    # tensor_list = [
    #     torch.empty(sum(r), input_.shape[-1], device=input_.device, dtype=input_.dtype)
    #     for r in tokens_by_rank
    # ]
    # tensor_list[rank] = input_
    # torch.distributed.all_gather(tensor_list, input_, group=get_model_parallel_group())
    tensor_list = []
    tensor_list.append(input_)
    tensor_list.append(torch.randn((int(globals.SEQUENCE_SIZE*globals.BATCH_SIZE*globals.TOP_K - input_.shape[0]), input_.shape[-1]), device=input_.device, dtype=input_.dtype, requires_grad=True))

    # print(f'rank 0 shape:{input_.shape} \t total shape:{(globals.SEQUENCE_SIZE*globals.BATCH_SIZE*globals.TOP_K, input_.shape[-1])}')

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=gather_dim)

    # Bf16 convert
    # if dt == torch.bfloat16 and get_fp32_allreduce():
    if dt == torch.bfloat16:
        output = output.bfloat16()

    # print(f'_dmoe_gather output shape: {output.shape}')

    return output

    

#--------------------------------------------------------



#--------------------------------------------------------
def gather_from_expert_model_parallel_region(input_, tokens_per_expert):
    return _GatherFromExpertModelParallelRegion.apply(input_, tokens_per_expert)


class _GatherFromExpertModelParallelRegion(torch.autograd.Function):
    """Gather the input from expert model parallel region and concatinate.

    The major difference between this and _GatherFromModelParallelRegion is in the
    dMoE case, we need to gather & split along the first dimension, not the last
    """

    @staticmethod
    def symbolic(graph, input_, tokens_per_expert):
        # TODO: not sure if this is sufficient? not sure how this gets used downstream...
        return _dmoe_gather(input_, tokens_per_expert)

    @staticmethod
    def forward(ctx, input_, tokens_per_expert):
        # Save tokens_per_expert in the context for later use in the backward pass
        ctx.save_for_backward(tokens_per_expert)

        return _dmoe_gather(input_, tokens_per_expert)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the tokens_per_expert from the context
        (tokens_per_expert,) = ctx.saved_tensors

        # no grad for tokens_per_expert
        return _dmoe_split(grad_output, tokens_per_expert), None

def _dmoe_split(input_, tokens_per_expert):
    """Split the tensor along its first dimension according to where tokens
    were routed, keeping the corresponding slice."""

    world_size = globals.MP_WORLD_SIZE
    # print(f'_dmoe_split input shape: {input_.shape}')
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension, getting the expert tokens
    output = get_expert_tokens_for_rank(input_, tokens_per_expert)

    # print(f'_dmoe_split output shape: {output.shape}')

    return output
#--------------------------------------------------------


#--------------------------------------------------------
def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # with get_cuda_rng_tracker().fork():
    # with checkpointing.get_cuda_rng_tracker().fork():
    init_method(weight)
#--------------------------------------------------------


#--------------------------------------------------------
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
#--------------------------------------------------------