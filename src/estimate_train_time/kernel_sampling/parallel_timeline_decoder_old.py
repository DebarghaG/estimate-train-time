import json
# import ujson as json
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import time
import os
import numpy as np
import csv
import tools

# DATA = None
# CPU_OP_ITEMS = None
# CUDA_RUNTIME_ITEMS = None
# KERNEL_ITEMS = None
# NUM_THREADS = 4


class TimelineDecoder:
    def __init__(self, data_path, num_threads):
        self.data = None
        self.cpu_op_items = None
        self.cuda_runtime_items = None
        self.kernel_items = None
        self.num_threads = num_threads

        with open(data_path, 'r') as f:
            data = json.load(f)

        self.data = data["traceEvents"] 
        self.cpu_op_items = self.get_cpu_op_items(self.data)
        self.cuda_runtime_items = self.get_cuda_runtime_items(self.data)
        self.kernel_items = self.get_kernel_items(self.data)



    def get_nested_value(self, data, key_path):
        for key in key_path:
            if isinstance(data, list):
                # If data is a list, apply the rest of the path to each item in the list
                return [get_nested_value(item, key_path[key_path.index(key) + 1:]) for item in data]
            data = data.get(key, {})
        return data


    def search_item_nested_range_with_list(self, json_data, key_path, min_value, max_value):
        results = []
        for item in json_data:
            values = self.get_nested_value(item, key_path)

            if isinstance(values, list) and any(min_value < v < max_value for v in values if isinstance(v, (int, float))):
                results.append(item)
            elif isinstance(values, (int, float)) and min_value < values < max_value:
                results.append(item)
        return results


    def search_item_value_with_list(self, json_data, key_path, target_values):
        results = []
        for item in json_data:
            values = self.get_nested_value(item, key_path)

            if isinstance(values, list) and any(v in target_values for v in values if isinstance(v, (int, float, str))):
                results.append(item)
            elif isinstance(values, (int, float, str)) and values in target_values:
                results.append(item)
        return results



    def search_in_range_cpu_ops(self, json_data, target_value):
        results = []
        for item in json_data:
            value = item.get('ts') + item.get('dur')
            if value > target_value:
                results.append(item)
        return results


    def chunk_data(self, data, num_chunks):
        chunk_size = len(data) // num_chunks + 1
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


    def parallel_search_nested_range(self, data, key_path, min_value, max_value, num_threads=4):

        data_chunks = self.chunk_data(data, num_threads)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.search_item_nested_range_with_list, chunk, key_path, min_value, max_value)
                for chunk in data_chunks
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        
        return results


    def parallel_search_values(self, data, key_path, target_values, num_threads=4):

        data_chunks = self.chunk_data(data, num_threads)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.search_item_value_with_list, chunk, key_path, target_values)
                for chunk in data_chunks
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return results


    def parallel_search_in_range_cpu_ops(self, data, target_value, num_threads=4):

        data_chunks = self.chunk_data(data, num_threads)
        
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.search_in_range_cpu_ops, chunk, target_value)
                for chunk in data_chunks
            ]
            for future in as_completed(futures):
                results.extend(future.result())
        return sorted(results, key=lambda x: x.get('dur'))


    def get_cpu_op_items(self, data):
        key_path = ['cat']
        target_values = ['cpu_op', 'user_annotation']
        found_items = self.parallel_search_values(data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_cuda_runtime_items(self, data):
        key_path = ['cat']
        target_values = ['cuda_runtime']
        found_items = self.parallel_search_values(data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_kernel_items(self, data):
        key_path = ['cat']
        target_values = ['kernel', 'gpu_memcpy']
        found_items = self.parallel_search_values(data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_items_by_name(self, item_names):
        key_path = ['name']
        target_values = item_names
        found_items = self.parallel_search_values(self.data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_sub_items_by_name(self, item_names, higher_item):
        sub_data = self.get_sub_data_by_item(higher_item)
        key_path = ['name']
        target_values = item_names
        found_items = self.parallel_search_values(sub_data, key_path, target_values, num_threads=self.num_threads)
        return found_items


    def get_sub_data_by_item(self, item):
        start_time = item.get('ts')
        end_time = start_time + item.get('dur')
        key_path = ['ts']
        sub_data = self.parallel_search_nested_range(self.data, key_path, start_time, end_time, num_threads=self.num_threads)
        return sub_data


    def get_cuda_runtimes_by_item(self, item):
        start_time = item.get('ts')
        end_time = start_time + item.get('dur')
        key_path = ['ts']
        cuda_runtimes = self.parallel_search_nested_range(self.cuda_runtime_items, key_path, start_time, end_time, num_threads=self.num_threads)
        return sorted(cuda_runtimes, key=lambda x: x.get('ts'))


    def get_cpu_ops_by_item(self, item):
        start_time = item.get('ts')
        end_time = start_time + item.get('dur')
        key_path = ['ts']
        cpu_ops = self.parallel_search_nested_range(self.cpu_op_items, key_path, start_time, end_time, num_threads=self.num_threads)
        return cpu_ops


    def get_kernels_by_cuda_runtimes(self, cuda_runtimes):
        key_path = ['args', 'External id']
        target_values = [d.get('args').get('External id') for d in cuda_runtimes]
        kernels = self.parallel_search_values(self.kernel_items, key_path, target_values, num_threads=self.num_threads)
        kernels = sorted(kernels, key=lambda x: x.get('ts'))
        
        min_value = kernels[0].get('ts') - 1
        max_value = kernels[-1].get('ts') + 1 
        kernels = self.parallel_search_nested_range(self.kernel_items, ['ts'], min_value, max_value, num_threads=self.num_threads)

        return sorted(kernels, key=lambda x: x.get('ts'))


    def get_the_kernel_of_the_cuda_runtime(self, cuda_runtime, kernels):
        key_path = ['args', 'External id']
        target_values = [cuda_runtime.get('args').get('External id')]
        kernel_list = self.parallel_search_values(kernels, key_path, target_values, num_threads=self.num_threads)
        return kernel_list[0] if len(kernel_list) > 0 else None


    def get_the_cuda_runtime_by_kernel(self, kernel, cuda_runtimes):
        key_path = ['args', 'External id']
        target_values = [kernel.get('args').get('External id')]
        cuda_runtime_list = self.parallel_search_values(cuda_runtimes, key_path, target_values, num_threads=self.num_threads)
        return cuda_runtime_list[0] if len(cuda_runtime_list) > 0 else None


    def get_the_cpu_ops_of_the_cuda_runtime(self, cuda_runtime, cpu_ops):
        if cuda_runtime is None:
            return [None, None]
        cuda_runtime_start = cuda_runtime.get('ts')
        cuda_runtime_end = cuda_runtime_start + cuda_runtime.get('dur')
        left_cpu_ops = self.parallel_search_nested_range(cpu_ops, ['ts'], 0, cuda_runtime_start, num_threads=self.num_threads)
        if len(left_cpu_ops) > 0:
            in_range_cpu_ops = self.parallel_search_in_range_cpu_ops(left_cpu_ops, cuda_runtime_end, num_threads=self.num_threads)
        else:
            in_range_cpu_ops = [None]
        # highest cpu_op with biggest dur, lowest cpu_op with smallest dur 
        return [in_range_cpu_ops[-1], in_range_cpu_ops[0]] if len(in_range_cpu_ops) > 0 else [None, None]



    def get_gpu_runtime(self, item_names):
        results = []
        for item_name in item_names:
            target_items = self.get_items_by_name([item_name])
            print(f'total items: {len(target_items)}')
            temp_list = []
            for target_item in target_items:
                cuda_runtimes = self.get_cuda_runtimes_by_item(target_item)
                kernels = self.get_kernels_by_cuda_runtimes(cuda_runtimes)
                GPU_time = kernels[-1].get('ts') + kernels[-1].get('dur') - kernels[0].get('ts')
                temp_list.append(GPU_time)
                print(f'Item {len(results)+1}/{len(item_names)}: {len(temp_list)}/{len(target_items)} is done.')
            results.append(sum(temp_list) / len(temp_list))
        return results



    def get_statistic_by_item_name(self, item_names):
        target_items = self.get_items_by_name(item_names)
        print(f'total items: {len(target_items)}')

        results_list = []
        max_len = 0

        count = 0
        for target_item in target_items:
            cuda_runtimes = self.get_cuda_runtimes_by_item(target_item)
            kernels = self.get_kernels_by_cuda_runtimes(cuda_runtimes)
            cpu_ops = self.get_cpu_ops_by_item(target_item)
            
            # list of [highest cpu_op, lowest cpu_op, kernel]
            results = []
            for kernel in kernels:
                cuda_runtime = self.get_the_cuda_runtime_by_kernel(kernel, cuda_runtimes)
                highest_cpu_op, lowest_cpu_op = self.get_the_cpu_ops_of_the_cuda_runtime(cuda_runtime, cpu_ops)
                results.append([highest_cpu_op, lowest_cpu_op, kernel])
            
            if len(results) > max_len:
                max_len = len(results)
            results_list.append(results)
            count += 1
            print(f'{count}/{len(target_items)} done !')

        longgest_results_list = []
        avaiable_count = 0
        for results in results_list:
            if len(results) == max_len:
                avaiable_count += 1
                longgest_results_list.append(results)

        print(f'avaiable results: {avaiable_count}')

        merged_results = []

        for i in range(max_len):
            merged_item = {'highest_cpu_op_list': [], 'lowest_cpu_op_list': [], 'kernel_list': []}
            for results in longgest_results_list:
                merged_item.get('highest_cpu_op_list').append(results[i][0])
                merged_item.get('lowest_cpu_op_list').append(results[i][1])
                merged_item.get('kernel_list').append(results[i][2])
            merged_results.append(merged_item)

        return merged_results


    # write csv row
    def write_result_to_csv(self, path, columns_name, result):
        if os.path.exists(path):
            with open(path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(result)
        else:
            with open(path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(columns_name)   
                writer.writerow(result)  


    def merged_results_to_CSV(self, save_path, merged_kernel_list):
        columns_name = ['cpu_op_0', 'cpu_op_0_id', 'cpu_op_0_input_dim', 'cpu_op_1', 'cpu_op_1_id', 'cpu_op_1_input_dim', 'kernel', 'kernel_id', 'kernel_overhead(us)', 'kernel_dur(us)']
        
        result = []
        merged_kernel_list_length = len(merged_kernel_list)
        for i in range(merged_kernel_list_length):
            kernel = merged_kernel_list[i]

            highest_cpu_op_name = kernel.get('highest_cpu_op_list')[0].get('name')  if kernel.get('highest_cpu_op_list')[0] is not None else 'None'
            highest_cpu_op_id = kernel.get('highest_cpu_op_list')[0].get('args').get('External id') if kernel.get('highest_cpu_op_list')[0] is not None else 'None'
            highest_cpu_op_input_dim = kernel.get('highest_cpu_op_list')[0].get('args').get('Input Dims') if kernel.get('highest_cpu_op_list')[0] is not None else 'None'
            
            lowest_cpu_op_name = kernel.get('lowest_cpu_op_list')[0].get('name') if kernel.get('lowest_cpu_op_list')[0] is not None else 'None'
            lowest_cpu_op_id = kernel.get('lowest_cpu_op_list')[0].get('args').get('External id') if kernel.get('lowest_cpu_op_list')[0] is not None else 'None'
            lowest_cpu_op_input_dim = kernel.get('lowest_cpu_op_list')[0].get('args').get('Input Dims') if kernel.get('lowest_cpu_op_list')[0] is not None else 'None'
        
            kernel_name = kernel.get('kernel_list')[0].get('name')
            kernel_id = kernel.get('kernel_list')[0].get('args').get('External id')
            
            kernel_dur_count = 0
            for gpu_kernel in kernel.get('kernel_list'):
                kernel_dur_count += gpu_kernel.get('dur')
            kernel_dur_average = kernel_dur_count / len(kernel.get('kernel_list'))

            if i == 0:
                kernel_overhead_average = 0
            else:
                last_kernel_list = merged_kernel_list[i - 1]
                kernel_overhead_count = 0
                for j in range(len(last_kernel_list.get('kernel_list'))):
                    kernel_overhead_count += kernel.get('kernel_list')[j].get('ts') - (last_kernel_list.get('kernel_list')[j].get('ts') + last_kernel_list.get('kernel_list')[j].get('dur'))
                kernel_overhead_average = kernel_overhead_count / len(last_kernel_list)

            result = [highest_cpu_op_name, highest_cpu_op_id, highest_cpu_op_input_dim, lowest_cpu_op_name, lowest_cpu_op_id, lowest_cpu_op_input_dim, kernel_name, kernel_id, kernel_overhead_average, kernel_dur_average]

            result = np.asarray(result, dtype="object")
            
            self.write_result_to_csv(save_path, columns_name, result)   



# def test():
#     data_path = './profilers/GPT_20B_2_N1_GPN1/2239682_11608/PP1_MP1_DP1_ZERO0/0/aisct02_1549904.1730684232021.pt.trace.json'
#     timelineDecoder = TimelineDecoder(data_path, num_threads=16)
#     item_names = ['deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward', 'CheckpointFunctionBackward']  
#     results = timelineDecoder.get_gpu_runtime(item_names)
#     print(results)

# def test2():
#     data_path = './profilers/GPT_20B_2_N1_GPN1/2239682_11608/PP1_MP1_DP1_ZERO0/0/aisct02_1549904.1730684232021.pt.trace.json'
#     timelineDecoder = TimelineDecoder(data_path, num_threads=16)
#     name = ['CheckpointFunctionBackward']  
#     merged_results = timelineDecoder.get_statistic_by_item_name(name)
#     save_path = './test_GPT_20B_2_N1_GPN1_bwd.csv'
#     timelineDecoder.merged_results_to_CSV(save_path, merged_results)




def get_decoders(path_list, unm_threads):
    decoder_list = []
    for path in path_list:
        print(f'loading: {path}')
        decoder_list.append(TimelineDecoder(path, num_threads=unm_threads))
    return decoder_list


# def gpu_runtimes(decoder_list, module_list, target_list, columns_name, save_path):

#     results_dict = {}
#     for target in target_list:
#         results = []
#         for decoder in decoder_list:
#             results.append(decoder.get_gpu_runtime([target]))
#         results_dict[target] = results

#     # write results to csv
#     for i in range(len(module_list)):
#         result = [module_list[i], target_list[i], results_dict.get(target_list[i])]
#         tools.write_one_result_to_csv(save_path, columns_name, result)

#     # return results_dict


def gpu_runtimes(decoder_list, module_list, target_list, columns_name, save_path):

    for i in range(len(module_list)):
        module_name = module_list[i]
        target = target_list[i]
        results = []
        for decoder in decoder_list:
            print(f'{module_name}, GPU:{len(results)}')
            results.append(decoder.get_gpu_runtime([target])[0])
        list_for_write = [module_name, target, results]
        tools.write_one_result_to_csv(save_path, columns_name, list_for_write)


def operator_statistics(decoder_list, module_list, target_list, save_folder):

    encoder_index = 0
    for decoder in decoder_list:
        folder = save_folder + '/' + str(encoder_index) 
        os.makedirs(folder, exist_ok=True)
        for i in range(len(module_list)):
            module_name = module_list[i]
            target = target_list[i]
            merged_results = decoder.get_statistic_by_item_name([target])
            save_path = folder +  '/' + module_name + '.csv'
            decoder.merged_results_to_CSV(save_path, merged_results)
        encoder_index += 1


def modules_gpu_runtime(decoder_list, save_folder):
    module_list = ['encoder_fwd', 'encoder_bwd', 'reduce_grads', 'optimizer_step', 'fwd', 'bwd', 'update']
    target_list = ['deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward', 
                   'CheckpointFunctionBackward',
                   'deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads', 
                   'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step', 
                   'deepspeed/runtime/pipe/module.py(319): forward', 
                   'torch/autograd/__init__.py(103): backward', 
                   'deepspeed/runtime/pipe/engine.py(312): train_batch'
                   ]
    columns_name = ['module', 'function', 'gpu_runtime(us)']
    save_path = save_folder + '/module_times.csv'
    gpu_runtimes(decoder_list, module_list, target_list, columns_name, save_path)


def modules_operators_runtime(decoder_list, save_folder):
    module_list = ['encoder_fwd', 'encoder_bwd', 'reduce_grads', 'optimizer_step', 'fwd', 'bwd']
    target_list = ['deepspeed/runtime/activation_checkpointing/checkpointing.py(496): forward', 
                   'CheckpointFunctionBackward',
                   'deepspeed/runtime/pipe/engine.py(270): _exec_reduce_grads', 
                   'deepspeed/runtime/pipe/engine.py(1140): _exec_optimizer_step', 
                   'deepspeed/runtime/pipe/module.py(319): forward', 
                   'torch/autograd/__init__.py(103): backward'
                   ]
    operator_statistics(decoder_list, module_list, target_list, save_folder)


def decoding(path_list, save_folder, unm_threads):
    decoder_list = get_decoders(path_list, unm_threads)
    modules_gpu_runtime(decoder_list, save_folder)
    # modules_operators_runtime(decoder_list, save_folder)



# def test2():
#     # head = 'profilers/GPT_20B_2_N1_GPN1/2239682_11608/PP1_MP1_DP1_ZERO0/0/aisct02_1549904.1730684232021.pt.trace.json'
#     head = 'perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1/0/nid001412_560029.1728564975718.pt.trace.json'
#     # middle = 'perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1/32/nid001597_1888490.1728564983928.pt.trace.json'
#     # tail = 'perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1/127/nid003713_699016.1728564966214.pt.trace.json'
#     # path_list = [head, middle, tail]
#     path_list = [head]
#     # module_list = ['fwd', 'bwd', 'update']
#     # target_list = ['deepspeed/runtime/pipe/module.py(319): forward', 
#     #                'torch/autograd/__init__.py(103): backward', 
#     #                'deepspeed/runtime/pipe/engine.py(312): train_batch'
#     #                ]

#     module_list = ['update']
#     target_list = ['deepspeed/runtime/pipe/engine.py(312): train_batch'
#                    ]
#     columns_name = ['module', 'function', 'gpu_runtime(us)']
#     save_path = './test.csv'
#     unm_threads =  16
#     gpu_runtimes(path_list, module_list, target_list, columns_name, save_path, unm_threads)


def GPT_20B_4_4_8_32_4_decoding():
    head = 'perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1/0/nid001412_560029.1728564975718.pt.trace.json'
    middle = 'perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1/32/nid001597_1888490.1728564983928.pt.trace.json'
    tail = 'perlmutter/GPT_20B_4_4_8_32_4/31694544/PP4_MP4_DP8_ZERO1/127/nid003713_699016.1728564966214.pt.trace.json'
    path_list = [head, middle, tail]
    save_folder = './perlmutter/GPT_20B_4_4_8_32_4/31694544'
    unm_threads = 16
    decoding(path_list, save_folder, unm_threads)


def GPT_20B_4_8_4_32_4_decoding():
    head = 'perlmutter/GPT_20B_4_8_4_32_4/31697107/PP4_MP8_DP4_ZERO1/0/nid001441_730299.1728571320116.pt.trace.json'
    middle = 'perlmutter/GPT_20B_4_8_4_32_4/31697107/PP4_MP8_DP4_ZERO1/32/nid001868_2129062.1728571329593.pt.trace.json'
    tail = 'perlmutter/GPT_20B_4_8_4_32_4/31697107/PP4_MP8_DP4_ZERO1/127/nid008552_777562.1728571311051.pt.trace.json'
    path_list = [head, middle, tail]
    save_folder = 'perlmutter/GPT_20B_4_8_4_32_4/31697107'
    unm_threads = 16
    decoding(path_list, save_folder, unm_threads)


def GPT_20B_8_4_4_32_4_decoding():
    head = 'perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1/0/nid008276_673053.1728570631834.pt.trace.json'
    middle = 'perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1/16/nid008336_1201576.1728570638701.pt.trace.json'
    tail = 'perlmutter/GPT_20B_8_4_4_32_4/31697099/PP8_MP4_DP4_ZERO1/127/nid008684_2178173.1728570618001.pt.trace.json'
    path_list = [head, middle, tail]
    save_folder = 'perlmutter/GPT_20B_8_4_4_32_4/31697099'
    unm_threads = 16
    decoding(path_list, save_folder, unm_threads)


def llama_13B_4_8_2_16_4_decoding():
    head = 'perlmutter/llama_13B_4_8_2_16_4/31674706/PP4_MP8_DP2_ZERO1/0/nid001364_817966.1728536136897.pt.trace.json'
    middle = 'perlmutter/llama_13B_4_8_2_16_4/31674706/PP4_MP8_DP2_ZERO1/16/nid001372_1207264.1728536149209.pt.trace.json'
    tail = 'perlmutter/llama_13B_4_8_2_16_4/31674706/PP4_MP8_DP2_ZERO1/63/nid003209_1846364.1728536125178.pt.trace.json'
    path_list = [head, middle, tail]
    save_folder = 'perlmutter/llama_13B_4_8_2_16_4/31674706'
    unm_threads = 16
    decoding(path_list, save_folder, unm_threads)


def llemma_7B_4_2_2_4_4_decoding():
    head = 'perlmutter/llemma_7B_4_2_2_4_4/31664313/PP4_MP2_DP2_ZERO1/0/nid001876_1718219.1728507307150.pt.trace.json'
    middle = 'perlmutter/llemma_7B_4_2_2_4_4/31664313/PP4_MP2_DP2_ZERO1/4/nid002436_1846718.1728507313926.pt.trace.json'
    tail = 'perlmutter/llemma_7B_4_2_2_4_4/31664313/PP4_MP2_DP2_ZERO1/15/nid003893_1161181.1728507295680.pt.trace.json'
    path_list = [head, middle, tail]
    save_folder = 'perlmutter/llemma_7B_4_2_2_4_4/31664313'
    unm_threads = 16
    decoding(path_list, save_folder, unm_threads)


if __name__ == "__main__":
    # GPT_20B_4_4_8_32_4_decoding()
    # GPT_20B_4_8_4_32_4_decoding()
    # GPT_20B_8_4_4_32_4_decoding()
    # llama_13B_4_8_2_16_4_decoding()
    llemma_7B_4_2_2_4_4_decoding()

