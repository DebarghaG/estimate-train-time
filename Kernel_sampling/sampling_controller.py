import argparse

from tensor_shape_generator import ShapeGenerator
import torch


with torch.profiler.profile() as profiler:
        pass


import torch.distributed as dist
import os
import time

import importlib

import sampling_tools

from datetime import datetime

import numpy as np

import traceback

# 20B: hidden_size = 6144, length = 2048, batch = 4, 50331648,  h = 64
# llama 13B: hidden_size = 5120, length = 2048, batch = 4, 41943040, h = 40
# llemma 7B: hidden_size = 4096, length = 4096, batch = 4, 67108864, h = 32
#  (llemma 7B 8 batch - llama 13B 2 batch) / 2000 ~= 65536

# mp [1 ~ 16]  mul 2, 6, 
# b [2 ~ 8] mul 2, 3 
# h [16 ~ 80] add 8, 8  
# l [1024 ~ 5120] add 512, 8
# dim [2048 ~ 8192] add 512, 12 
# 13,824 in total

# Using FP16

# Nodes settings:  intra: 1-2, 1-4, 1node 6h MH inter: 2-1, 2-2, 2-4 2nodes 9h  18h MH  = 24 MH  x3 = 72 MH 

# shape middle range start = [20971520] end = [134217728] step = [65536]   all_reduce and p2p (intra: 1-2 1node 3h, inter: 2-1, 2nodes 3h )

# shape large range  start = [134217728] end = [1200000000] step = [600000] reduce_scatter and all_gather and all_reduce

# 6 slurms in total

# Using Zero1

def get_shapeGenerator(configs, args):
    kernel_name = configs['kernel_name']
    starts = configs['starts']
    steps = configs['steps']
    ends = configs['ends']
    operators = configs['operators']
    first_n_column = configs['first_n_column']
    precision = args.precision

    part = args.part
    parts = args.parts

    raw_data_folder = './sampling_data'
    os.makedirs(raw_data_folder, exist_ok=True)
    raw_data_path = raw_data_folder + '/' + torch.cuda.get_device_name(0).replace(' ', '') + '_' + kernel_name + '_' + precision + '_' + str(parts) + '_' + str(part) + '.csv' 

    shapeGenerator = ShapeGenerator(kernel_name, precision, starts, steps, ends, operators, first_n_column, raw_data_path)
    return shapeGenerator


def get_all(configs, args):

    shapeGenerator = get_shapeGenerator(configs, args)

    raw_data_path = shapeGenerator.raw_data_path

    kernel_name = configs['kernel_name']
    precision = args.precision
    columns_name = configs['columns_name']

    part = args.part
    parts = args.parts

    log_folder = './sampling_log'
    os.makedirs(log_folder, exist_ok=True)
    log_path = log_folder + '/' + torch.cuda.get_device_name(0).replace(' ', '') + '_' + kernel_name + '_' + precision + '_' + str(parts) + '_' + str(part) + '.txt' 

    profiler_folder = './temp_profilings'
    os.makedirs(profiler_folder, exist_ok=True)


    start_time = time.time()
    shapes = True
    while shapes is not None:
        shapes = shapeGenerator.next_dims()
        if shapes is not None and shapeGenerator.used_dimes % parts == part - 1:
            shapes_to_str = '_'.join(map(str, shapes))
            profiler_folder_path = profiler_folder + '/' + torch.cuda.get_device_name(0).replace(' ', '') + '_' + kernel_name + '_' + precision + '_' + shapes_to_str
            # os.makedirs(profiler_folder_path, exist_ok=True)
            
            dur_list = data_collect_one(configs, args, shapeGenerator, profiler_folder_path)

            if dur_list is not None:

                record = shapes.copy()

                try:
                    record = np.concatenate((record.copy(), dur_list), axis=0)
                    sampling_tools.write_one_result_to_csv(raw_data_path, columns_name, record)
                except Exception as e:
                    traceback.print_exc()

                    
                # record the progress
                if save_to_txt_log(shapeGenerator, log_path, start_time, parts):
                    start_time = time.time()


def get_one(configs, args):
    shapeGenerator = get_shapeGenerator(configs, args)

    shapeGenerator.used_dimes = 2

    kernel_name = configs['kernel_name']
    precision = args.precision

    profiler_folder = './temp_profilings'
    os.makedirs(profiler_folder, exist_ok=True)

    shapes = configs['shapes']
   
    shapes_to_str = '_'.join(map(str, shapes))
    profiler_folder_path = profiler_folder + '/' + torch.cuda.get_device_name(0).replace(' ', '') + '_' + kernel_name + '_' + precision + '_' + shapes_to_str
    # os.makedirs(profiler_folder_path, exist_ok=True)
    
    shapeGenerator.current_dims = shapes

    dur_list = data_collect_one(configs, args, shapeGenerator, profiler_folder_path)

    if dur_list is not None:
        print(f'{kernel_name}:    input_shape:{shapes}     timecost(us) [fwd, bwd]: {dur_list}')
    else:
        print(f'{kernel_name} can not get runtime.')


def get_profiler(configs, args):
    shapeGenerator = get_shapeGenerator(configs, args)

    shapeGenerator.used_dimes = 0

    kernel_name = configs['kernel_name']
    precision = args.precision

    profiler_folder = './temp_profilings'
    os.makedirs(profiler_folder, exist_ok=True)

    shapes = configs['shapes']
   
    shapes_to_str = '_'.join(map(str, shapes))
    profiler_folder_path = profiler_folder + '/' + torch.cuda.get_device_name(0).replace(' ', '') + '_' + kernel_name + '_' + precision + '_' + shapes_to_str
    # os.makedirs(profiler_folder_path, exist_ok=True)
    
    shapeGenerator.current_dims = shapes

    profiler_flag = profiler_create_one(configs, args, shapeGenerator, profiler_folder_path)

    if profiler_flag:
        print('Profiling Done!')
    else:
        print('Profiling Fail!')


def get_function(configs):
    module_name = configs['module_name']
    function_name = configs['function_name']
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def run_function(configs, args, shapes):
    try:
        func = get_function(configs)
        # shapes = shapeGenerator.current_dims
        precision = args.precision
        device_num = args.device_num
        func(shapes, precision, device_num)
        return True
    except Exception as e:
        traceback.print_exc()
        print(f'Shaps: {shapes} \t Error type: {type(e)}')
        print(e)
        return False


def profiler_create_one(configs, args, shapeGenerator, profiler_folder_path):

    # self.setup()
    # initial profiler
    prof = torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=configs['wait'],
            warmup=configs['warmup'],
            active=configs['active']),
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_folder_path),
        record_shapes=True,
        profile_memory=True)

    prof.start()

    empty_loops = configs['wait'] + configs['warmup']

    # 2 wait and 3 warmup loops
    for i in range(empty_loops + configs['active']):
        if i >= empty_loops:
            func_return_flag = run_function(configs, args, shapeGenerator.current_dims)
        else:
            func_return_flag = run_function(configs, args, configs['warmup_shapes'])

        if func_return_flag:
            prof.step()
        else:
            prof.stop() 
            return False
    prof.stop()
    return True


def profiler_reader(profiler_folder_path, targets, shapeGenerator):
    profiler_file_name = sampling_tools.get_current_profiler_file_name(profiler_folder_path)
    # print(profiler_file_name)
    profiler_file_path = profiler_folder_path + '/' + profiler_file_name
    dur_lists_temp = []
    for target in targets:
        dur_list = sampling_tools.get_GPU_runtime_list(profiler_file_path, target)
        dur_list.sort()
        dur_lists_temp.append(dur_list)
        # print(f'the dur list is :{dur_list}')
    min_len = min([len(x) for x in dur_lists_temp])

    if min_len > 0:
        # dur_lists = [x[0:min_len - 5] for x in dur_lists_temp]
        dur_lists = [x[0:min_len] for x in dur_lists_temp]
        sampling_tools.delete_folder(profiler_folder_path)
        return np.array(dur_lists).T.tolist()
    else:
        sampling_tools.delete_folder(profiler_folder_path)
        return None


def data_collect_one(configs, args, shapeGenerator, profiler_folder_path):

    profiler_creater_flag = profiler_create_one(configs, args, shapeGenerator, profiler_folder_path)
    if profiler_creater_flag:
        dur_lists = profiler_reader(profiler_folder_path, configs['targets'], shapeGenerator)
        # print(dur_lists)
        if dur_lists is not None:
            # print(dur_lists)
            # print(np.var(dur_lists, axis=0))
            # print(np.var(dur_lists[2:7], axis=0))
            return np.mean(dur_lists[2:7], axis=0)
        else:
            return None
    else:
        sampling_tools.delete_folder(profiler_folder_path)
        return None


def save_to_txt_log(shapeGenerator, log_path, start_time, parts):
    log_per_iterations = 5
    if (shapeGenerator.get_collected_samples() // parts) % log_per_iterations == 0 or (shapeGenerator.get_collected_samples()) // parts == (shapeGenerator.total_samples // parts):
        time_cost = time.time() - start_time

        collected_samples, total_samples = shapeGenerator.collect_progress()

        collected_samples = collected_samples // parts
        total_samples = total_samples // parts

        current_date_string = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]'

        kernel_str = f"Kernel: {shapeGenerator.kernel_name}_{shapeGenerator.precision} | "

        progress_str = f'Progress: {int(collected_samples)}/{int(total_samples)}={collected_samples / total_samples * 100:.2f}% | '
        
        speed = time_cost / log_per_iterations
        speed_str = f'Speed: {speed:.2f}s/sample | '

        remains_time = (total_samples - collected_samples) * speed
        days, hours, minutes, seconds = sampling_tools.convert_seconds(int(remains_time))
        remains_str = f'Remaining: {days}d{hours}h{minutes}m{seconds}s'

        output_str = current_date_string + kernel_str + progress_str + speed_str + remains_str

        with open(log_path, "a") as file:
            # Write the content to the file
            file.write(output_str + '\n')
        return True
    else:
        return False


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='', type=str,
                        help="Path of config yml file")
    parser.add_argument("--precision", default='fp16', type=str,
                        help="Data type of tensor fp16 or fp32")
    parser.add_argument("--parts", default='1', type=int,
                        help="The work is breaked to how many parts")
    parser.add_argument("--part", default='1', type=int,
                        help="nth part of this running. (1 ~ gpus per node)")
    parser.add_argument("--device_num", default='0', type=int,
                        help="the device for sampling, 0, 1, 2, 3 ...")
    args = parser.parse_args()
    return args



def main(configs, args):
    func = globals()[configs["run"]]
    func(configs, args)


if __name__ == "__main__":
    args = parse_arguments() 
    configs = sampling_tools.config_decoder(args.config_path)
    main(configs, args)
