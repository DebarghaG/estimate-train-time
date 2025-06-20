
import torch

from megatron.fused_kernels import load
load()

import scaled_upper_triang_masked_softmax_cuda

print("Custom Kernel loading successfully!")


def ScaledUpperTriangMaskedSoftmax(shapes, precision, device_num):
    mp, b, h, l = shapes

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    inputs = torch.rand([b*h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)
    output_grads = torch.rand([b*h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)

    scale_t = torch.tensor([1])

    softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
        inputs, scale_t[0]
    )

    input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
        output_grads, softmax_results, scale_t[0]
    )



if __name__ == "__main__":
    shapes = [1, 4, 4, 1024]
    precision = 'fp16'
    ScaledUpperTriangMaskedSoftmax(shapes, precision, 0)