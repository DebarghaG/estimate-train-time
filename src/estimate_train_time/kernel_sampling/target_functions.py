import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


import math

from megatron.RoPE import (
    RotaryEmbedding,
    RoPE_CLASS,
    apply_rotary_pos_emb_torch,
    apply_rotary_pos_emb,
)

from megatron.fused_softmax import torch_softmax

from megatron.config import WORLD_SIZE

from megatron.cross_entropy import vocab_parallel_cross_entropy

from flash_attn import flash_attn_func

from flash_attn.ops.activations import swiglu

from megatron.norms import RMSNorm

from megatron.stage_simulator import FirstStage, MiddleStage, LastStage

from megatron_moe.moe_tools import get_moe_object

import deepspeed
from deepspeed.ops.adam import FusedAdam

from megatron.fused_kernels import load
load()

import scaled_upper_triang_masked_softmax_cuda



def layernorm(shapes, precision, device_num):
    b, l, dim = shapes

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    layernorm = nn.LayerNorm([dim])

    if precision == 'fp16':
        layernorm.weight.data = layernorm.weight.data.half()
        layernorm.bias.data = layernorm.weight.data.half()
    elif precision == 'bf16':
        layernorm.weight.data = layernorm.weight.data.bfloat16()
        layernorm.bias.data = layernorm.weight.data.bfloat16()

    layernorm.to(device_num)

    output = layernorm(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()



# def fillmask(shapes, precision, device_num):
#     b, h, l = shapes
#     if precision == 'fp16':
#         dtype = torch.float16
#     else:
#         dtype = torch.float32

#     target = torch.rand([b, h, l, l], dtype=dtype, device=device_num)

#     mask = torch.tril(torch.ones((1, l, l), device=device_num)).view(1, 1, l, l)
#     mask = mask < 0.5

#     target.masked_fill_(mask, 0)

#     # loss = target.sum()
#     # loss.backward()


def fillmask(shapes, precision, device_num):
    mp, b, h, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if dim % h != 0:
        raise ValueError("dim is not divisible by head!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    c = torch.empty([b*h//mp, l, l], dtype=dtype, device=device_num)
    q = torch.rand([b*h//mp, l, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    k = torch.rand([b*h//mp, dim//h, l], dtype=dtype, device=device_num, requires_grad=True)

    target = torch.rand([b, h//mp, l, l], dtype=dtype, device=device_num)

    alpha = 1.0
    beta = 0
    att_score = torch.baddbmm(c, q, k, beta=beta, alpha=alpha)
    att_score = att_score.view(b, h//mp, l, l)

    mask = torch.tril(torch.ones((1, l, l), device=device_num)).view(1, 1, l, l)
    mask = mask < 0.5

    att_score.masked_fill_(mask, 0)

    output = torch.nn.Softmax(dim=-1)(att_score)

    loss = nn.MSELoss()(output, target)
    loss.backward()


def softmax(shapes, precision, device_num):
    mp, b, h, l = shapes

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([b, h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([b, h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)

    softmax = nn.Softmax(dim=-1).to(device_num)
    output = softmax(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()


def baddbmm(shapes, precision, device_num):
    mp, b, h, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if dim % h != 0:
        raise ValueError("dim is not divisible by head!")


    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    c = torch.empty([b*h//mp, l, l], dtype=dtype, device=device_num)
    q = torch.rand([b*h//mp, l, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    k = torch.rand([b*h//mp, dim//h, l], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.randn([b*h//mp, l, l], dtype=dtype, device=device_num)

    alpha = 1.0
    beta = 0
    output = torch.baddbmm(c, q, k, beta=beta, alpha=alpha)
    loss = nn.MSELoss()(output, target)

    loss.backward()


def bmm(shapes, precision, device_num):
    mp, b, h, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if dim % h != 0:
        raise ValueError("dim is not divisible by head!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    socre = torch.rand([b*h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)
    v = torch.rand([b*h//mp, l, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.randn([b*h//mp, l, dim//h], dtype=dtype, device=device_num)

    output = torch.bmm(socre, v)
    loss = nn.MSELoss()(output, target)

    loss.backward()


def linear1(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    target = torch.rand([l, b, dim*3 // mp], dtype=dtype, device=device_num, requires_grad=True)

    if precision == 'fp16':
        linear = nn.Linear(dim, dim*3 // mp).to(device_num).half()
    elif precision == 'bf16':
        linear = nn.Linear(dim, dim*3 // mp).to(device_num).to(torch.bfloat16)
    else:
        linear = nn.Linear(dim, dim*3 // mp).to(device_num)
    
    output = linear(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()



def linear2(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")
   
    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim//mp], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    if precision == 'fp16':
        linear = nn.Linear(dim//mp, dim, bias=False).to(device_num).half()
    elif precision == 'bf16':
        linear = nn.Linear(dim//mp, dim, bias=False).to(device_num).to(torch.bfloat16)
    else:
        linear = nn.Linear(dim//mp, dim, bias=False).to(device_num)
    
    output = linear(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()


def linear3(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([l, b, dim*4//mp], dtype=dtype, device=device_num, requires_grad=True)

    if precision == 'fp16':
        linear = nn.Linear(dim, dim*4//mp, bias=False).to(device_num).half()
    elif precision == 'bf16':
        linear = nn.Linear(dim, dim*4//mp, bias=False).to(device_num).to(torch.bfloat16)
    else:
        linear = nn.Linear(dim, dim*4//mp, bias=False).to(device_num)
    
    output = linear(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()


def linear4(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim*4//mp], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    if precision == 'fp16':
        linear = nn.Linear(dim*4//mp, dim, bias=False).to(device_num).half()
    elif precision == 'bf16':
        linear = nn.Linear(dim*4//mp, dim, bias=False).to(device_num).to(torch.bfloat16)
    else:
        linear = nn.Linear(dim*4//mp, dim, bias=False).to(device_num)
    
    output = linear(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()



def gelu(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim*4//mp], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([l, b, dim*4//mp], dtype=dtype, device=device_num, requires_grad=True)

    # if precision == 'fp16':
    #     linear = nn.Linear(dim, dim*4, bias=False).to(device_num).half()
    # else:
    #     linear = nn.Linear(dim, dim*4, bias=False).to(device_num)
    
    output = F.gelu(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()


def RoPE(shapes, precision, device_num):
    # mp: model paralleism ways, b:batch, h:heads, l:length, dim:hidden_size
    mp, b, h, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if dim % h != 0:
        raise ValueError("dim is not divisible by head!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    # h: hidden size
    # n: number of attention heads
    # kv: number of key or value heads
    # p: number of model parallel partitions
    # np: n/p
    # kvp: kv/p
    # hp: h/p
    # hn: h/n
    # b: batch size
    # s: sequence length
    # l: number of layers

    # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
    # (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
    #     mixed_x_layer, 3
    # )
    
    #  [sq, b, np, hn] = [l, b, h, dim//h]
    query_layer = torch.rand([l, b, h//mp, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    key_layer = torch.rand([l, b, h//mp, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    value_layer = torch.rand([l, b, h//mp, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    
    target = torch.rand([l, b, h//mp, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    
    rotary_emb = RotaryEmbedding(dim//h, base=10000, precision=dtype)

    query_rot, key_rot = query_layer, key_layer

    apply_rotary_fn = (apply_rotary_pos_emb_torch if dtype == torch.bfloat16 else apply_rotary_pos_emb)

    seq_len = key_layer.shape[0]
    offset = 0
    # if exists(layer_past) and layer_past.numel() > 0:
    #     offset = layer_past[0].shape[0]
    #     seq_len += offset

    cos, sin = rotary_emb(value_layer, seq_len=seq_len)
    query_layer, key_layer = apply_rotary_fn(
        query_rot, key_rot, cos, sin, offset=offset
    )

    output = query_layer - key_layer
    loss = nn.MSELoss()(output, target)
    loss.backward()



def linear_final(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    input = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    target = torch.rand([l, b, partition_vocab_size], dtype=dtype, device=device_num, requires_grad=True)

    if precision == 'fp16':
        linear = nn.Linear(dim, partition_vocab_size, bias=False).to(device_num).half()
    elif precision == 'bf16':
        linear = nn.Linear(dim, partition_vocab_size, bias=False).to(device_num).to(torch.bfloat16)
    else:
        linear = nn.Linear(dim, partition_vocab_size, bias=False).to(device_num)
    
    output = linear(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()

# def RoPE(shapes, precision, device_num):
#     # b:batch, h:heads, l:length, dim:hidden_size
#     b, h, l, dim = shapes

#     if precision == 'fp16':
#         dtype = torch.half
#     else:
#         dtype = torch.float
    
#     # h: hidden size
#     # n: number of attention heads
#     # kv: number of key or value heads
#     # p: number of model parallel partitions
#     # np: n/p
#     # kvp: kv/p
#     # hp: h/p
#     # hn: h/n
#     # b: batch size
#     # s: sequence length
#     # l: number of layers

#     # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
#     # (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
#     #     mixed_x_layer, 3
#     # )
    
#     #  [sq, b, np, hn] = [l, b, h, dim//h]
#     query_layer = torch.rand([l, b, h, dim//h], dtype=dtype, device=device_num, requires_grad=True)
#     key_layer = torch.rand([l, b, h, dim//h], dtype=dtype, device=device_num, requires_grad=True)
#     value_layer = torch.rand([l, b, h, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    
#     target = torch.rand([l, b, h, dim//h], dtype=dtype, device=device_num, requires_grad=True)
    

#     rotary_emb = RoPE_CLASS(dim//h, dtype)
#     out_query_layer, out_key_layer = rotary_emb(query_layer, key_layer, value_layer)

#     output = out_query_layer - out_key_layer
#     loss = nn.MSELoss()(output, target)
#     loss.backward()


def parallel_cross_entropy_128(shapes, precision, device_num):
    mp, b, l = shapes

    WORLD_SIZE = mp

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    input = torch.rand([b, l, partition_vocab_size], dtype=dtype, device=device_num, requires_grad=True)
    
    target = torch.randint(0, vocab_size, (b, l), requires_grad=False).to(device_num)

    loss_mask = torch.randint(0, 2, (b, l), requires_grad=False).to(device_num)

    losses = vocab_parallel_cross_entropy(input, target)

    loss_mask = loss_mask.view(-1)

    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    loss.backward()



def embedding(shapes, precision, device_num):
    mp, b, l, dim = shapes

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    embedding_layer = nn.Embedding(num_embeddings=partition_vocab_size, embedding_dim=dim).to(device_num).to(dtype=dtype)

    input = torch.randint(0, partition_vocab_size, (b, l), requires_grad=False).to(device_num)
    
    target = torch.rand([b, l, dim], dtype=dtype, device=device_num, requires_grad=False)

    output = embedding_layer(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()



def flash_atten(shapes, precision, device_num):
    mp, b, h, l, dim = shapes

    if dim % mp != 0:
        raise ValueError("dim is not divisible by mp!")

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if dim % h != 0:
        raise ValueError("dim is not divisible by head!")

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # q: (batch_size, seqlen, nheads, headdim)
    # k: (batch_size, seqlen, nheads_k, headdim)
    # v: (batch_size, seqlen, nheads_k, headdim)

    Q = torch.randn(b, l, h//mp, dim//h, dtype=dtype, requires_grad=True).cuda(device_num)
    K = torch.randn(b, l, h//mp, dim//h, dtype=dtype, requires_grad=True).cuda(device_num)
    V = torch.randn(b, l, h//mp, dim//h, dtype=dtype, requires_grad=True).cuda(device_num)

    # Q = torch.randn(b, h//mp, l, dim, dtype=dtype, requires_grad=True).cuda(device_num)
    # K = torch.randn(b, h//mp, l, dim, dtype=dtype, requires_grad=True).cuda(device_num)
    # V = torch.randn(b, h//mp, l, dim, dtype=dtype, requires_grad=True).cuda(device_num)



    # mask = torch.ones(b, l).bool().cuda(device_num)
    # output, _ = flash_attn_func(Q, K, V, mask)

    output = flash_attn_func(Q, K, V)

    loss = output.sum()

    loss.backward()


# def swiglu_test(shapes, precision, device_num):
#     mp, b, l, dim = shapes

#     if dim % mp != 0:
#         raise ValueError("dim is not divisible by mp!")

#     if precision == 'fp16':
#         dtype = torch.float16
#     else:
#         dtype = torch.float32

#     input = torch.rand([l, b, dim*4//mp], dtype=dtype, device=device_num, requires_grad=True)
#     target = torch.rand([l, b, dim*4//mp], dtype=dtype, device=device_num, requires_grad=True)

#     # if precision == 'fp16':
#     #     linear = nn.Linear(dim, dim*4, bias=False).to(device_num).half()
#     # else:
#     #     linear = nn.Linear(dim, dim*4, bias=False).to(device_num)
    
#     output = swiglu(input, input)

#     loss = nn.MSELoss()(output, target)
#     loss.backward()



def RMSlayernorm(shapes, precision, device_num):
    b, l, dim = shapes

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)
    target = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    layernorm = RMSNorm(dim)

    # if precision == 'fp16':
    #     layernorm.weight.data = layernorm.weight.data.half()
    #     layernorm.bias.data = layernorm.weight.data.half()

    layernorm.to(device_num)

    output = layernorm(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()


def res_add(shapes, precision, device_num):
    b, l, dim = shapes

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tensor_a = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)
    tensor_b = torch.rand([l, b, dim], dtype=dtype, device=device_num, requires_grad=True)

    tensor_c = tensor_a + tensor_b


def ScaledUpperTriangMaskedSoftmax(shapes, precision, device_num):
    mp, b, h, l = shapes

    if h % mp != 0:
        raise ValueError("head is not divisible by mp!")
    
    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    inputs = torch.rand([b*h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)
    output_grads = torch.rand([b*h//mp, l, l], dtype=dtype, device=device_num, requires_grad=True)

    scale_t = torch.tensor([1])

    with record_function("custom_block_scaled_upper_triang_masked_softmax_cuda_forward"):
        softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
            inputs, scale_t[0]
        )

    with record_function("custom_block_scaled_upper_triang_masked_softmax_cuda_backward"):
        input_grads = scaled_upper_triang_masked_softmax_cuda.backward(
            output_grads, softmax_results, scale_t[0]
        )


def firstStage_optimizer(shapes, precision, device_num):
    mp, dim, encoders = shapes

    b = 2 
    l = 1024

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    input = torch.randint(0, partition_vocab_size, (b, l), requires_grad=False).to(device_num)

    target = torch.rand([b, l, dim], dtype=dtype, device=device_num, requires_grad=False)

    model = FirstStage(mp, dim, encoders, precision).to(device_num)

    output = model(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()

    optimizer = FusedAdam(
        model.parameters(),
        lr=0.001,            # Learning rate
        betas=(0.9, 0.999),  # Adam's beta coefficients
        eps=1e-8,            # Epsilon for numerical stability
        weight_decay=0       # Weight decay for regularization
    )

    optimizer.step()



def middleStage_optimizer(shapes, precision, device_num):
    mp, dim, encoders = shapes

    b = 2 
    l = 1024

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    input = torch.rand([b, l, dim], dtype=dtype, device=device_num, requires_grad=False)

    target = torch.rand([b, l, dim], dtype=dtype, device=device_num, requires_grad=False)

    model = MiddleStage(mp, dim, encoders, precision).to(device_num)

    output = model(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()

    optimizer = FusedAdam(
        model.parameters(),
        lr=0.001,            # Learning rate
        betas=(0.9, 0.999),  # Adam's beta coefficients
        eps=1e-8,            # Epsilon for numerical stability
        weight_decay=0       # Weight decay for regularization
    )

    optimizer.step() 



def lastStage_optimizer(shapes, precision, device_num):
    mp, dim, encoders = shapes

    b = 2 
    l = 1024

    if precision == 'fp16':
        dtype = torch.float16
    elif precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    input = torch.rand([b, l, dim], dtype=dtype, device=device_num, requires_grad=False)

    target = torch.rand([b, l, partition_vocab_size//mp], dtype=dtype, device=device_num, requires_grad=False)

    model = LastStage(mp, dim, encoders, precision).to(device_num)

    output = model(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()

    optimizer = FusedAdam(
        model.parameters(),
        lr=0.001,            # Learning rate
        betas=(0.9, 0.999),  # Adam's beta coefficients
        eps=1e-8,            # Epsilon for numerical stability
        weight_decay=0       # Weight decay for regularization
    )

    optimizer.step()



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

    input = torch.randn((l//mp, b, dim), device=0, dtype=torch.bfloat16, requires_grad=True)
    target = torch.randn((l//mp, b, dim), device=0, dtype=torch.bfloat16, requires_grad=True)

    moe = get_moe_object(shapes).to(device_num)
    output, _ = moe(input)

    loss = nn.MSELoss()(output, target)
    loss.backward()




# if __name__ == "__main__":
    # shapes = [1, 1, 4, 4]
    # precision = 'fp16'
    # swiglu_test(shapes, precision, 0)
  