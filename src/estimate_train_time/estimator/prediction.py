import math
import argparse
import os

from estimate_train_time.estimator import tools
from estimate_train_time.estimator.predictor import Predictor
from estimate_train_time.estimator import encoder_config_to_layer_input
from estimate_train_time.estimator import layer_input_to_predictor_input


def _resolve_data_path(path, config_dir):
    """Resolve a data path from config.

    Tries paths in order:
    1. Absolute path (if path is absolute and exists)
    2. Relative to config file directory
    3. Relative to bundled data directory
    """
    # If it's an absolute path and exists, use it
    if os.path.isabs(path) and os.path.exists(path):
        return path

    # Try relative to config directory
    config_relative = os.path.join(config_dir, path)
    if os.path.exists(config_relative):
        return config_relative

    # Try relative to bundled data directory
    bundled_data_dir = tools.get_bundled_data_path()
    # Strip leading ./ if present
    clean_path = path.lstrip('./')
    bundled_relative = os.path.join(bundled_data_dir, clean_path)
    if os.path.exists(bundled_relative):
        return bundled_relative

    # Return original path and let downstream code handle the error
    return path


def get_layer_input_shape(encoder_config, module_name, function_name):
    if module_name == 'encoder_config_to_layer_input':
        return getattr(encoder_config_to_layer_input, function_name)(encoder_config)
    raise ValueError(f"Unknown module: {module_name}")


def get_map_function(module_name, function_name):
    if module_name == 'layer_input_to_predictor_input':
        return getattr(layer_input_to_predictor_input, function_name)
    raise ValueError(f"Unknown module: {module_name}")


def get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs):

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
            

    pp_p2p_cost = predictor.pp_p2p(nccl_data_folder, nccl_config_folder, nccl_gpu_name, b*l*dim, mp, dp, pp, gpus_per_node) if pp > 1 else 0
    run_time_dict['pp_p2p'] = pp_p2p_cost

    mp_allreduce_cost = predictor.mp_allreduce(nccl_data_folder, nccl_config_folder, nccl_gpu_name, b*l*dim, mp, gpus_per_node) if mp > 1 else 0
    run_time_dict['mp_allreduce'] = mp_allreduce_cost

    head_parameters = get_embedding_parameters(encoder_config) + head_encoder_layers * get_encoder_parameters(encoder_config)
    middle_parameters = middle_encoder_layers * get_encoder_parameters(encoder_config)
    tail_parameters = tail_encoder_layers * get_encoder_parameters(encoder_config) + get_layer_norm_parameteres(encoder_config) + get_final_lnear_parameters(encoder_config)

    head_dp_allreduce_cost = dp_allreduce_cost(predictor, head_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    middle_dp_allreduce_cost = dp_allreduce_cost(predictor, middle_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    tail_dp_allreduce_cost = dp_allreduce_cost(predictor, tail_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    run_time_dict['dp_allreduce'] = [head_dp_allreduce_cost, middle_dp_allreduce_cost, tail_dp_allreduce_cost]

    head_dp_allgather_cost = dp_allgather_cost(predictor, head_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    middle_dp_allgather_cost = dp_allgather_cost(predictor, middle_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    tail_dp_allgather_cost = dp_allgather_cost(predictor, tail_parameters, mp, dp, gpus_per_node, comm_bucket, nccl_data_folder, nccl_config_folder, nccl_gpu_name) if dp > 1 else 0
    run_time_dict['dp_allgather'] = [head_dp_allgather_cost, middle_dp_allgather_cost, tail_dp_allgather_cost]

    head_dp_optimizer_cost = local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, head_encoder_layers, precision, dim, 'firstStage_optimizer') if dp > 1 else 0
    middle_dp_optimizer_cost = local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, middle_encoder_layers, precision, dim, 'middleStage_optimizer') if dp > 1 else 0
    tail_dp_optimizer_cost = local_update_cost(predictor, operator_data_folder, operator_config_folder, gpu_name, mp, tail_encoder_layers, precision, dim, 'lastStage_optimizer') if dp > 1 else 0
    run_time_dict['optimizer'] = [head_dp_optimizer_cost, middle_dp_optimizer_cost, tail_dp_optimizer_cost]

    run_time_dict['update'] = [a + b for a, b in zip(run_time_dict['dp_allgather'], [head_dp_optimizer_cost, middle_dp_optimizer_cost, tail_dp_optimizer_cost])]
    
    encoder_fwd = 0
    encoder_bwd = 0
    for function in encoder_function_list:
        encoder_fwd += run_time_dict.get(function + '_' + 'fwd')
        encoder_bwd += run_time_dict.get(function + '_' + 'bwd')

    encoder_fwd += fwd_syncs * mp_allreduce_cost
    encoder_bwd += bwd_syncs * mp_allreduce_cost

    encoder_bwd += encoder_fwd
    
    run_time_dict['encoder_fwd'] = encoder_fwd
    run_time_dict['encoder_bwd'] = encoder_bwd


    head_fwd = run_time_dict.get('embedding_fwd') + mp_allreduce_cost + run_time_dict.get('encoder_fwd') * head_encoder_layers
    middle_fwd = run_time_dict.get('encoder_fwd') * middle_encoder_layers
    tail_fwd = run_time_dict.get('encoder_fwd') * tail_encoder_layers + run_time_dict.get(layernorm_name + '_fwd') + run_time_dict.get('linear_final_fwd') + run_time_dict.get('parallel_cross_entropy_128_fwd')

    head_bwd =  run_time_dict.get('embedding_bwd') + run_time_dict.get('encoder_bwd') * head_encoder_layers
    middle_bwd = run_time_dict.get('encoder_bwd') * middle_encoder_layers
    tail_bwd = run_time_dict.get('encoder_bwd') * tail_encoder_layers + run_time_dict.get(layernorm_name + '_bwd') + run_time_dict.get('linear_final_bwd') + run_time_dict.get('parallel_cross_entropy_128_bwd')

    run_time_dict['fwd'] = [head_fwd, middle_fwd, tail_fwd]
    run_time_dict['bwd'] = [head_bwd, middle_bwd, tail_bwd]

    update_all_F_all_B_max_optimizer = all_F_all_B([head_fwd+pp_p2p_cost, middle_fwd+pp_p2p_cost, tail_fwd], [head_bwd, middle_bwd+pp_p2p_cost, tail_bwd+pp_p2p_cost], steps_per_update, pp) + head_dp_allreduce_cost + max(run_time_dict['update'])

    return update_all_F_all_B_max_optimizer



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


def pp_partitions(encoders, pp):
    layers = encoders + 5
    return [math.ceil(layers/pp)-2, math.floor(layers/pp), math.floor(layers/pp)-3]


def one_batch_predict(config_path):
    """Predict the time cost for one batch of training.

    Args:
        config_path: Path to a YAML configuration file

    Returns:
        Estimated time cost in microseconds
    """
    configs = tools.config_decoder(config_path)

    # Get config directory for resolving relative paths
    config_dir = os.path.dirname(os.path.abspath(config_path))

    predictor = Predictor()

    gpu_name = configs['gpu_name']

    # Resolve data folder paths
    operator_data_folder = _resolve_data_path(configs['operator_data_folder'], config_dir)
    nccl_data_folder = _resolve_data_path(configs['nccl_data_folder'], config_dir)

    training_config = configs['training_config']
    comm_bucket = configs['comm_bucket']

    encoders = configs['encoders']
    encoder_layers_list = pp_partitions(encoders, training_config[0])

    function_list = configs['function_list']
    encoder_function_list = configs['encoder_function_list']
    layernorm_name = configs['layernorm_name']

    fwd_syncs = configs['fwd_syncs']
    bwd_syncs = configs['bwd_syncs']

    return get_operator_statistics(predictor, gpu_name, operator_data_folder, nccl_data_folder, training_config, comm_bucket, encoder_layers_list, function_list, encoder_function_list, layernorm_name, fwd_syncs, bwd_syncs)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='', type=str,
                        help="Path of config yml file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments() 
    one_batch_timecost = one_batch_predict(args.config_path)
    print(f'Estimated timecost of current training configs is {one_batch_timecost} us.')

# test_command
# python mml_3d_prediction.py --config_path ./target_config/llemma_7b_4_2_2_P.yml