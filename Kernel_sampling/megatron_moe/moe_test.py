from .moe import ParallelDroplessMoE 

from .init_functions import small_init_init_method, wang_init_method

from .neoxargs import NeoXArgs

import torch
import torch.nn as nn

from . import globals

# def initial_neox_args(att_dict):
#     neox_args = NeoXArgs()
#     for key, value in att_dict.items():
#         setattr(neox_args, key, value)


def moe_test(att_dict):
    
    neox_args = NeoXArgs(att_dict)

    init_method = small_init_init_method(neox_args.hidden_size)

    output_layer_init_method = wang_init_method(neox_args.num_layers, neox_args.hidden_size)

    moe_layer = ParallelDroplessMoE(neox_args, init_method, output_layer_init_method).to(0)

    print(f'create the moe object sucessfully!')

    # create input data 
    x = torch.randn((neox_args.seq_length, neox_args.batch_size, neox_args.hidden_size), device=0, dtype=neox_args.params_dtype, requires_grad=True)
    target = torch.randn((neox_args.seq_length, neox_args.batch_size, neox_args.hidden_size), device=0, dtype=neox_args.params_dtype, requires_grad=True)

    # try fwd
    output, _ = moe_layer(x)
    print(f'fwd executes sucessfully!')
    
    # try bwd
    loss = nn.MSELoss()(output, target)
    loss.backward()

    print(f'bwd executes sucessfully!')
    return moe_layer


def test():
    att_dict = {}
    att_dict['activation'] = 'gelu'
    att_dict['moe_num_experts'] = 16
    att_dict['hidden_size'] = 1024
    att_dict['seq_length'] = 256
    att_dict['intermediate_size'] = None
    att_dict['params_dtype'] = torch.bfloat16
    att_dict['moe_router_type'] = "sinkhorn"
    att_dict['moe_top_k'] = 8
    att_dict['num_layers'] = 12
    att_dict['batch_size'] = 4

    # have to set mp size to 1. Useing seq_length//mp and moe_num_experts//mp to simulate the work load of each mp ways.
    # otherwise, if top_k > 1 it will cause shape errors.
    globals.MP_WORLD_SIZE = 2  
    globals.MP_RANK = 0
    globals.HIIDDEN_SIZE = att_dict['hidden_size']
    globals.SEQUENCE_SIZE = att_dict['seq_length']
    globals.BATCH_SIZE = att_dict['batch_size']
    globals.TOP_K = att_dict['moe_top_k']

    moe_test(att_dict)



if __name__ == "__main__":
    test()
