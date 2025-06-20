
import numpy as np
import math


def baddbmm(input):
    # mp, b, h, l, dim
    mp, b, h, l, dim = input

    # map to h, m, k, n
    new_h = b * (h // mp)
    m = l
    k = dim // h
    n = l 

    return np.array([new_h, m, k, n])


def bmm(input):
    # mp, b, h, l, dim
    mp, b, h, l, dim = input

    # map to h, m, k, n
    new_h = b * (h // mp)
    m = l
    k = l 
    n = dim // h
    
    return np.array([new_h, m, k, n])


def embedding(input):
    # mp, b, l, dim 
    mp, b, l, dim = input

    vocab_size = 50257

    vocab = b * l

    emb_rows = math.ceil(vocab_size / (128 * mp)) * 128

    emb_columns = dim

    return np.array([vocab, emb_rows, emb_columns])


def fillmask(input):
    # mp, b, h, l, dim
    mp, b, h, l, dim = input

    # new_h = h // mp
    return np.array([b, h//mp, l, dim])


def flash_atten(input):
    # mp, b, h, l, dim
    mp, b, h, l, dim = input

    return np.array([b, l, h//mp, dim//h])


def gelu(input):
    mp, b, l, dim = input
    return np.array([l, b, dim*4//mp])


def layernorm(input):
    # b, l, dim = input
    return input


def res_add(input):
    # b, l, dim = input
    return input


def RMSlayernorm(input):
    # b, l, dim = input
    return input


def linear_final(input):
    mp, b, l, dim = input

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    return np.array([l * b, dim, partition_vocab_size])


def linear1(input):
    mp, b, l, dim = input

    return np.array([l * b, dim, dim*3 // mp])


def linear2(input):
    mp, b, l, dim = input
    
    return np.array([l * b, dim // mp, dim])


def linear3(input):
    mp, b, l, dim = input
    
    return np.array([l * b, dim, dim*4//mp])


def linear4(input):
    mp, b, l, dim = input
    
    return np.array([l * b, dim*4//mp, dim])


def parallel_cross_entropy_128(input):
    mp, b, l = input

    vocab_size = 50257

    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128

    return np.array([b, l, partition_vocab_size])


def RoPE(input):
    mp, b, h, l, dim = input
    
    return np.array([l, b, h//mp, dim//h])


def softmax(input):
    mp, b, h, l = input
    
    return np.array([b, h//mp, l, l])


def ScaledUpperTriangMaskedSoftmax(input):
    mp, b, h, l = input
    
    return np.array([b*h//mp, l, l])


def optimizer(input):
    return input


def firstStage_optimizer(input):
    return input


def allreduce_fp16(input):
    shape, nodes, GPUsPerNode = input
    return np.array([shape*2/(nodes*GPUsPerNode), 2*(nodes*GPUsPerNode-1)])
    return input


def allreduce_large_fp16(input):
    shape, nodes, GPUsPerNode = input
    return np.array([shape*2/(nodes*GPUsPerNode), 2*(nodes*GPUsPerNode-1)])


def allgather_large_fp16(input):
    shape, nodes, GPUsPerNode = input
    return np.array([shape*2/(nodes*GPUsPerNode), nodes*GPUsPerNode-1])


def p2p_fp16(input):
    shape, nodes, GPUsPerNode = input
    return np.array([shape*2])


def middleStage_optimizer(input):
    return input



def lastStage_optimizer(input):
    return input

