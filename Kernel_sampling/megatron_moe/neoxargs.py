class NeoXArgs:
    def __init__(self, dict):
        self.activation = dict['activation']
        self.moe_num_experts = dict['moe_num_experts']
        self.hidden_size = dict['hidden_size']
        self.seq_length = dict['seq_length']
        self.intermediate_size = dict['intermediate_size']
        self.params_dtype = dict['params_dtype']
        self.moe_router_type = dict['moe_router_type'] 
        self.moe_top_k = dict['moe_top_k']
        self.num_layers = dict['num_layers']
        self.batch_size = dict['batch_size']


"""
moe_mlp.py needs:
    neox_args.activation
    neox_args.moe_num_experts
    neox_args.hidden_size
    neox_args.intermediate_size
    neox_args.params_dtype

moe.py needs:
    neox_args.moe_router_type == "sinkhorn"
    neox_args.params_dtype

    needs to spicify 
        init_method,    
        output_layer_init_method,
        
    from init_funcgtion:
        # init methods
        "init_method": "small_init",
        "output_layer_init_method": "wang_init",

        from megatron.model.init_functions import get_init_methods
        self.init_method, self.output_layer_init_method = get_init_methods(self.neox_args)

    neox_args.moe_top_k


init_functions.py needs:
args.num_layers
args.hidden_size

"""