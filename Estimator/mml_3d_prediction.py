import math
import importlib
import tools
from predictor import Predictor


import argparse


def get_layer_input_shape(encoder_config, module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)(encoder_config)


def get_map_function(module_name, function_name):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name):

    pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node = training_config
    encoder_config = mp, b, h, l, dim
    head_encoder_layers, middle_encoder_layers, tail_encoder_layers = encoder_layers_list


    operator_config_folder = operator_data_folder
    nccl_config_folder = nccl_data_folder
    nccl_gpu_name = gpu_name

    columns_name = ['module', 'parameter', 'predition(us)']

    precision = 'fp16'

    propagation_list = ['fwd', 'bwd']

    run_time_dict = {}

    save_path = save_folder + '/' + save_name + '.csv'

    portion_save_path = save_folder + '/' + save_name + '_module_portion' + '.csv'

    portion_columns = ['module', 'portion']


    for function in function_list:
        for propagation in propagation_list:

            prediction = predictor.operator_statistic(operator_data_folder, operator_config_folder, gpu_name, [function], precision, encoder_config, propagation)
            
            writing_name = function + '_' + propagation
            module_name = 'encoder_config_to_layer_input'
            parameter_shape = get_layer_input_shape(encoder_config, module_name, function)
            
            map_function = get_map_function('layer_input_to_predictor_input', function)
            shape_new = map_function(parameter_shape)

            run_time_dict[writing_name] = prediction
            
            tools.write_one_result_to_csv(save_path, columns_name, [writing_name, shape_new, prediction])


    pp_p2p_cost = predictor.pp_p2p(nccl_data_folder, nccl_config_folder, nccl_gpu_name, b*l*dim, mp, dp, pp, gpus_per_node) if pp > 1 else 0
    run_time_dict['pp_p2p'] = pp_p2p_cost
    tools.write_one_result_to_csv(save_path, columns_name, ['pp_p2p', [b*l*dim//mp], pp_p2p_cost])

    mp_allreduce_cost = predictor.mp_allreduce(nccl_data_folder, nccl_config_folder, nccl_gpu_name, b*l*dim, mp, gpus_per_node) if mp > 1 else 0
    run_time_dict['mp_allreduce'] = mp_allreduce_cost
    tools.write_one_result_to_csv(save_path, columns_name, ['mp_allreduce', [b*l*dim], mp_allreduce_cost])

    head_parameters = get_embedding_parameters(encoder_config) + head_encoder_layers * get_encoder_parameters(encoder_config)
    middle_parameters = middle_encoder_layers * get_encoder_parameters(encoder_config)
    tail_parameters = tail_encoder_layers * get_encoder_parameters(encoder_config) + get_layer_norm_parameteres(encoder_config) + get_final_lnear_parameters(encoder_config)

    head_dp_allreduce_cost = dp_allreduce_cost(predictor, head_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    middle_dp_allreduce_cost = dp_allreduce_cost(predictor, middle_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    tail_dp_allreduce_cost = dp_allreduce_cost(predictor, tail_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    run_time_dict['dp_allreduce'] = [head_dp_allreduce_cost, middle_dp_allreduce_cost, tail_dp_allreduce_cost]
    tools.write_one_result_to_csv(save_path, columns_name, ['dp_allreduce', [head_parameters, middle_parameters, tail_parameters], [head_dp_allreduce_cost, middle_dp_allreduce_cost, tail_dp_allreduce_cost]])

    head_dp_allgather_cost = dp_allgather_cost(predictor, head_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    middle_dp_allgather_cost = dp_allgather_cost(predictor, middle_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    tail_dp_allgather_cost = dp_allgather_cost(predictor, tail_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    run_time_dict['dp_allgather'] = [head_dp_allgather_cost, middle_dp_allgather_cost, tail_dp_allgather_cost]
    tools.write_one_result_to_csv(save_path, columns_name, ['dp_allgather', [head_parameters//dp, middle_parameters//dp, tail_parameters//dp], [head_dp_allgather_cost, middle_dp_allgather_cost, tail_dp_allgather_cost]])

    head_dp_optimizer_cost = local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, head_encoder_layers, precision, dim, 'firstStage_optimizer') if dp > 1 else 0
    middle_dp_optimizer_cost = local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, middle_encoder_layers, precision, dim, 'middleStage_optimizer') if dp > 1 else 0
    tail_dp_optimizer_cost = local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, tail_encoder_layers, precision, dim, 'lastStage_optimizer') if dp > 1 else 0
    run_time_dict['optimizer'] = [head_dp_optimizer_cost, middle_dp_optimizer_cost, tail_dp_optimizer_cost]
    tools.write_one_result_to_csv(save_path, columns_name, ['optimizer', [head_parameters, middle_parameters, tail_parameters], run_time_dict['optimizer']])

    run_time_dict['update'] = [a + b for a, b in zip(run_time_dict['dp_allgather'], [head_dp_optimizer_cost, middle_dp_optimizer_cost, tail_dp_optimizer_cost])]
    tools.write_one_result_to_csv(save_path, columns_name, ['update', [head_parameters, middle_parameters, tail_parameters], run_time_dict['update']])
    

    # encoder_function_list = ['layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'layernorm', 'linear3', 'gelu', 'linear4']
    encoder_fwd = 0
    encoder_bwd = 0
    for function in encoder_function_list:
        encoder_fwd += run_time_dict.get(function + '_' + 'fwd')
        encoder_bwd += run_time_dict.get(function + '_' + 'bwd')

    encoder_fwd += fwd_syncs * mp_allreduce_cost
    encoder_bwd += bwd_syncs * mp_allreduce_cost

    encoder_bwd += encoder_fwd
    
    run_time_dict['encoder_fwd'] = encoder_fwd
    tools.write_one_result_to_csv(save_path, columns_name, ['encoder_fwd', encoder_config, encoder_fwd])
    run_time_dict['encoder_bwd'] = encoder_bwd
    tools.write_one_result_to_csv(save_path, columns_name, ['encoder_bwd', encoder_config, encoder_bwd])


    head_fwd = run_time_dict.get('embedding_fwd') + mp_allreduce_cost + run_time_dict.get('encoder_fwd') * head_encoder_layers
    middle_fwd = run_time_dict.get('encoder_fwd') * middle_encoder_layers
    tail_fwd = run_time_dict.get('encoder_fwd') * tail_encoder_layers + run_time_dict.get(layernorm_name + '_fwd') + run_time_dict.get('linear_final_fwd') + run_time_dict.get('parallel_cross_entropy_128_fwd')

    head_bwd =  run_time_dict.get('embedding_bwd') + run_time_dict.get('encoder_bwd') * head_encoder_layers
    middle_bwd = run_time_dict.get('encoder_bwd') * middle_encoder_layers
    tail_bwd = run_time_dict.get('encoder_bwd') * tail_encoder_layers + run_time_dict.get(layernorm_name + '_bwd') + run_time_dict.get('linear_final_bwd') + run_time_dict.get('parallel_cross_entropy_128_bwd')

    run_time_dict['fwd'] = [head_fwd, middle_fwd, tail_fwd]
    tools.write_one_result_to_csv(save_path, columns_name, ['stage_fwd', encoder_config, [head_fwd, middle_fwd, tail_fwd]])
    run_time_dict['bwd'] = [head_bwd, middle_bwd, tail_bwd]
    tools.write_one_result_to_csv(save_path, columns_name, ['stage_bwd', encoder_config, [head_bwd, middle_bwd, tail_bwd]])


    # update_noninterleaved_1F1B = noninterleaved_1F1B([head_fwd+pp_p2p_cost, middle_fwd+pp_p2p_cost, tail_fwd], [head_bwd, middle_bwd+pp_p2p_cost, tail_bwd+pp_p2p_cost], steps_per_update, pp) + head_dp_optimizer_cost + head_dp_allreduce_cost + head_dp_allgather_cost
    # tools.write_one_result_to_csv(save_path, columns_name, ['noninterleaved_1F1B', training_config, update_noninterleaved_1F1B])
    
    update_all_F_all_B_max_optimizer = all_F_all_B([head_fwd+pp_p2p_cost, middle_fwd+pp_p2p_cost, tail_fwd], [head_bwd, middle_bwd+pp_p2p_cost, tail_bwd+pp_p2p_cost], steps_per_update, pp) + head_dp_allreduce_cost + max(run_time_dict['update'])
    tools.write_one_result_to_csv(save_path, columns_name, ['all_F_all_B_max_optimizer', training_config, update_all_F_all_B_max_optimizer])


    # update_all_F_all_B_head = all_F_all_B([head_fwd+pp_p2p_cost, middle_fwd+pp_p2p_cost, tail_fwd], [head_bwd, middle_bwd+pp_p2p_cost, tail_bwd+pp_p2p_cost], steps_per_update, pp) +  + head_dp_optimizer_cost + head_dp_allreduce_cost + head_dp_allgather_cost
    # tools.write_one_result_to_csv(save_path, columns_name, ['all_F_all_B_head', training_config, update_all_F_all_B_head])






    # head_bwd_after_pp = head_dp_optimizer_cost + head_dp_allreduce_cost + head_dp_allgather_cost
    # last_middle_bwd_after_pp = middle_dp_optimizer_cost + middle_dp_allreduce_cost + middle_dp_allgather_cost - head_bwd 
    # tail_bwd_after_pp = tail_dp_optimizer_cost + tail_dp_allreduce_cost + tail_dp_allgather_cost - (pp-2)*(middle_bwd+pp_p2p_cost) - head_bwd 
    # tools.write_one_result_to_csv(save_path, columns_name, ['after_pp', training_config, [head_bwd_after_pp, last_middle_bwd_after_pp, tail_bwd_after_pp]])

    
    # max_after_pp= max([head_bwd_after_pp, last_middle_bwd_after_pp, tail_bwd_after_pp])
    # update_all_F_all_B_max = all_F_all_B([head_fwd+pp_p2p_cost, middle_fwd+pp_p2p_cost, tail_fwd], [head_bwd, middle_bwd+pp_p2p_cost, tail_bwd+pp_p2p_cost], steps_per_update, pp) + max_after_pp 
    # tools.write_one_result_to_csv(save_path, columns_name, ['all_F_all_B_max', training_config, update_all_F_all_B_max])



def local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, layers, precision, dim, operator_name):
    # num_layers,hidden_size,param_tensors,params
    optimizer_input = [mp, dim, layers]
    local_optimizer_cost = predictor.operator_statistic(operator_data_folder, operator_config_folder, gpu_name, [operator_name], precision, optimizer_input, 'dur')
    return local_optimizer_cost


def dp_allreduce_cost(predictor, parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name):
    # parameters_limit = 134217728
    parameters_limit = 0

    # communication cost ZERO1
    # gradients_allreduce
    allreduce_cost = 0
    allreduce_parameters = parameters
    while allreduce_parameters > parameters_limit:
        comm_parameters = comm_bucket if comm_bucket < allreduce_parameters else allreduce_parameters 
        temp_allreduce_cost = predictor.dp_allreduce(nccl_data_folder, nccl_config_folder, nccl_gpu_name, comm_parameters, mp, dp, gpus_per_node)
        allreduce_cost += temp_allreduce_cost
        allreduce_parameters -= comm_bucket

    return allreduce_cost


def dp_allgather_cost(predictor, parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name):
    # parameters_limit = 134217728
    parameters_limit = 0

    # parameters_allgather
    allgather_cost = 0
    allgather_parameters = parameters // dp
    while allgather_parameters > parameters_limit:
        comm_parameters = comm_bucket if comm_bucket < allgather_parameters else allgather_parameters 
        temp_allgather_cost = predictor.dp_allgather(nccl_data_folder, nccl_config_folder, nccl_gpu_name, comm_parameters, mp, dp, gpus_per_node)
        allgather_cost += temp_allgather_cost
        allgather_parameters -= comm_bucket
    return allgather_cost


def noninterleaved_1F1B(fwd_list, bwd_list, steps_per_update, pp):
    # list = [head, middle, tail]
    fwd_1 = fwd_list[0] + fwd_list[1] * (pp -2) + fwd_list[2]
    bwd_1 = bwd_list[0] + bwd_list[1] * (pp -2) + bwd_list[2]
    middle_cost = (fwd_list[2] + bwd_list[2]) * (steps_per_update - 1)
    return fwd_1 + middle_cost + bwd_1 


def all_F_all_B(fwd_list, bwd_list, steps_per_update, pp):
    cost = (steps_per_update - 1 + pp) * (max(fwd_list) + max(bwd_list))
    return cost 


# def all_F_all_B(fwd_list, bwd_list, steps_per_update, pp):
#     cost = (steps_per_update - 1) * (max(fwd_list) + max(bwd_list)) + (pp - 3) * (fwd_list[1] + bwd_list[1]) + (sum(fwd_list) + sum(bwd_list))
#     return cost 

def get_layer_norm_parameteres(encoder_config):
    mp, b, h, l, dim = encoder_config

    layer_norm = 2 * dim

    return layer_norm


def get_encoder_parameters(encoder_config):
    mp, b, h, l, dim = encoder_config

    layer_norm = 2 * dim

    linear1 = 3 * dim * (dim + 1) // mp

    linear2 = dim * (dim + 1) // mp

    linear3 = 4 * dim * (dim + 1) // mp

    linear4 = dim * (4 * dim + 1) // mp

    return 2*layer_norm + linear1 + linear2 + linear3 + linear4
  

def get_embedding_parameters(encoder_config):
    mp, b, h, l, dim = encoder_config
    vocab_size = 50257
    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128
    return partition_vocab_size*dim


def get_final_lnear_parameters(encoder_config):
    mp, b, h, l, dim = encoder_config
    vocab_size = 50257
    partition_vocab_size =  math.ceil(vocab_size / (128 * mp)) * 128
    return partition_vocab_size*dim





def GPT_20B_4_4_8_128(predictor, gpus_per_node, gpu_name, operator_data_folder, nccl_data_folder, save_folder, save_name):
    # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
    training_config = [4, 4, 8, 4, 64, 2048, 6144, 16, gpus_per_node]
    comm_bucket = 1260000000

    function_list = ['embedding', 'layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add']
    encoder_function_list = ['layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'layernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add']
    layernorm_name = 'layernorm'
    fwd_syncs = 1
    bwd_syncs = 2
    
    # Partitioning pipeline stages with method type:transformer|mlp
    encoder_layers_list = [11, 12, 9]
    get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name)



def GPT_20B_4_8_4_128(predictor, gpus_per_node, gpu_name, operator_data_folder, nccl_data_folder, save_folder, save_name):
    # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
    training_config = [4, 8, 4, 4, 64, 2048, 6144, 16, gpus_per_node]
    comm_bucket = 1260000000

    function_list = ['embedding', 'layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add']
    encoder_function_list = ['layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'layernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add']
    layernorm_name = 'layernorm'
    fwd_syncs = 1
    bwd_syncs = 2
    
    # Partitioning pipeline stages with method type:transformer|mlp
    encoder_layers_list = [11, 12, 9]
    get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name)



def GPT_20B_8_4_4_128(predictor, gpus_per_node, gpu_name, operator_data_folder, nccl_data_folder, save_folder, save_name):
    # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
    training_config = [8, 4, 4, 4, 64, 2048, 6144, 16, gpus_per_node]
    comm_bucket = 1260000000

    function_list = ['embedding', 'layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add']
    encoder_function_list = ['layernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'layernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add']
    layernorm_name = 'layernorm'
    fwd_syncs = 1
    bwd_syncs = 2
    
    # Partitioning pipeline stages with method type:transformer|mlp
    encoder_layers_list = [5, 6, 3]
    get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name)



def llama_13B_4_8_2_64(predictor, gpus_per_node, gpu_name, operator_data_folder, nccl_data_folder, save_folder, save_name):
    # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
    training_config = [4, 8, 2, 4, 40, 2048, 5120, 16, gpus_per_node]
    comm_bucket = 500000000

    function_list = ['embedding', 'RMSlayernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add']
    encoder_function_list = ['RMSlayernorm', 'linear1', 'RoPE', 'baddbmm', 'ScaledUpperTriangMaskedSoftmax', 'bmm', 'linear2', 'RMSlayernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add']
    layernorm_name = 'RMSlayernorm'

    fwd_syncs = 2
    bwd_syncs = 2

    #Partitioning pipeline stages with method type:transformer|mlp
    encoder_layers_list = [10, 11, 8]
    get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name)



def llemma_7B_4_2_2_16(predictor, gpus_per_node, gpu_name, operator_data_folder, nccl_data_folder, save_folder, save_name):
    # pp, mp, dp, b, h, l, dim, steps_per_update, gpus_per_node
    training_config = [4, 2, 2, 4, 32, 4096, 4096, 8, gpus_per_node]
    comm_bucket = 1260000000

    function_list = ['embedding', 'RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'linear3', 'gelu', 'linear4', 'linear_final', 'parallel_cross_entropy_128', 'res_add']
    encoder_function_list = ['RMSlayernorm', 'linear1', 'RoPE', 'flash_atten', 'linear2', 'RMSlayernorm', 'res_add', 'linear3', 'gelu', 'linear4', 'res_add']
    layernorm_name = 'RMSlayernorm'

    fwd_syncs = 2
    bwd_syncs = 2

    #Partitioning pipeline stages with method type:transformer|mlp
    encoder_layers_list = [8, 9, 6]
    get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='', type=str,
                        help="Path of config yml file")
    args = parser.parse_args()
    return args


def main(configs):
    predictor = Predictor()

    gpu_name = configs['gpu_name']
    operator_data_folder = configs['operator_data_folder']
    nccl_data_folder = configs['nccl_data_folder']

    training_config = configs['training_config']
    comm_bucket = configs['comm_bucket']

    encoders = configs['encoders']
    encoder_layers_list = pp_partitions(encoders, training_config[0])

    function_list = configs['function_list']
    encoder_function_list = configs['encoder_function_list']
    layernorm_name = configs['layernorm_name']

    fwd_syncs = configs['fwd_syncs']
    bwd_syncs = configs['bwd_syncs']

    save_folder = configs['save_folder']
    save_name = configs['save_name']

    get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs, save_folder, save_name)


def pp_partitions(encoders, pp):
    layers = encoders + 5
    return [math.ceil(layers/pp)-2, math.floor(layers/pp), math.floor(layers/pp)-3]



if __name__ == '__main__':
    args = parse_arguments() 
    configs = tools.config_decoder(args.config_path)
    main(configs)


# test_command
# python mml_3d_prediction.py --config_path ./target_config/llemma_7b_4_2_2_P.yml