import torch
import torch.nn as nn

from megatron_moe.moe_tools import get_moe_object

def moe(shapes, precision, device_num):
    mp, b, l, dim, moe_num_experts, intermediate_size, top_k = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")
    if moe_num_experts % mp != 0:
        raise ValueError("experts is not divisible by mp!")
    if (mp * top_k) > moe_num_experts:
        raise ValueError("invalide mp and top_k values!")
    if precision != 'bf16':
        raise ValueError("moe only supports for bf16!")
    if intermediate_size > dim*4:
        raise ValueError("too large intermediate_size!")

    input = torch.randn((l, b, dim), device=0, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn((l, b, dim), device=0, dtype=torch.bfloat16, requires_grad=True)

    moe = get_moe_object(shapes).to(device_num)
    output, _ = moe(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()


if __name__ == "__main__":
    # shapes = [mp, b, l, dim, moe_num_experts, intermediate_size, top_k]
    shapes = [4, 4, 1024, 1024, 16, 2048, 4]
    precision = 'bf16'
    moe(shapes, precision, 0)