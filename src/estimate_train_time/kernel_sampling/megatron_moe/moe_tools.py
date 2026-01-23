import torch
import torch.nn as nn

from .moe import ParallelDroplessMoE 
from .init_functions import small_init_init_method, wang_init_method
from .neoxargs import NeoXArgs
from . import globals


def get_moe_object(configs):
    # configs = [mp, b, l, dim, top_k, moe_num_experts, intermediate_size]
    mp, b, l, dim, moe_num_experts, intermediate_size, top_k = configs

    att_dict = {}
    att_dict['batch_size'] = b
    att_dict['seq_length'] = l 
    att_dict['hidden_size'] = dim 
    att_dict['moe_top_k'] = top_k

    att_dict['moe_num_experts'] = moe_num_experts
    att_dict['intermediate_size'] = None if intermediate_size == 0 else intermediate_size

    att_dict['activation'] = 'gelu'
    att_dict['params_dtype'] = torch.bfloat16
    att_dict['moe_router_type'] = "sinkhorn"
    att_dict['num_layers'] = 24

    globals.MP_WORLD_SIZE = mp   
    globals.MP_RANK = 0
    globals.HIIDDEN_SIZE = att_dict['hidden_size']
    globals.SEQUENCE_SIZE = att_dict['seq_length']
    globals.BATCH_SIZE = att_dict['batch_size']
    globals.TOP_K = att_dict['moe_top_k']

    return initial_moe(att_dict)


def initial_moe(att_dict):
    neox_args = NeoXArgs(att_dict)
    init_method = small_init_init_method(neox_args.hidden_size)
    output_layer_init_method = wang_init_method(neox_args.num_layers, neox_args.hidden_size)
    moe_layer = ParallelDroplessMoE(neox_args, init_method, output_layer_init_method).to(0)
    return moe_layer
