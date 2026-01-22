
import numpy as np
import math


def baddbmm(input):

    return input


def bmm(input):
    
    return input


def embedding(input):
    # mp, b, l, dim 
    mp, b, h, l, dim = input

    return np.array([mp, b, l, dim])


def fillmask(input):
    # # mp, b, h, l, dim
    # mp, b, h, l, dim = input

    # new_h = h // mp
    return input


def flash_atten(input):
    # mp, b, h, l, dim
    # mp, b, h, l, dim = input

    return input


def gelu(input):
    mp, b, h, l, dim = input
    return np.array([mp, b, l, dim])


def layernorm(input):
    mp, b, h, l, dim = input
    return np.array([b, l, dim])


def RMSlayernorm(input):
    mp, b, h, l, dim = input
    return np.array([b, l, dim])


def res_add(input):
    mp, b, h, l, dim = input
    return np.array([b, l, dim])


def linear_final(input):
    mp, b, h, l, dim = input

    return np.array([mp, b, l, dim])


def linear1(input):
    mp, b, h, l, dim = input

    return np.array([mp, b, l, dim])


def linear2(input):
    mp, b, h, l, dim = input

    return np.array([mp, b, l, dim])


def linear3(input):
    mp, b, h, l, dim = input

    return np.array([mp, b, l, dim])


def linear4(input):
    mp, b, h, l, dim = input

    return np.array([mp, b, l, dim])


def parallel_cross_entropy_128(input):
    mp, b, h, l, dim = input

    return np.array([mp, b, l])


def RoPE(input):
    
    return input


def softmax(input):
    mp, b, h, l, dim = input
    
    return np.array([mp, b, h, l])


def ScaledUpperTriangMaskedSoftmax(input):
    mp, b, h, l, dim = input
    
    return np.array([mp, b, h, l])


def optimizer(input):
    return input

def firstStage_optimizer(input):
    return input

def allreduce_fp16(input):
    shape, nodes, GPUsPerNode = input
    return np.array([shape*2/(nodes*GPUsPerNode), 2*(nodes*GPUsPerNode-1)])


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