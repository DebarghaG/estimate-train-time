import ijson
import numpy as np
import csv
import os
from datetime import datetime
import shutil
import random
import yaml


def get_current_time_string():
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    current_date_time_str = now.strftime("%Y%m%d%H%M%S")
    return current_date_time_str + str(random.randint(0, 1000))


def get_current_only_time_string():
    # Get the current date and time
    now = datetime.now()
    # Format the date and time as a string
    current_date_time_str = now.strftime("%Y%m%d%H%M%S")
    return current_date_time_str


# write csv row
def write_result_to_csv(path, columns_name, results):
    if os.path.exists(path):
        with open(path, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results)
    else:
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns_name)   
            writer.writerows(results)   


# write csv row
def write_one_result_to_csv(path, columns_name, result):
    if os.path.exists(path):
        with open(path, 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(result)
    else:
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns_name)   
            writer.writerow(result)   


# get profiler json file name
def get_current_profiler_file_name(folder_path):
    filename_list = os.listdir(folder_path)
    for filename in filename_list:
        if os.path.splitext(filename)[-1] == '.json':
            return(filename)
     

# delete the profiler folder
def delete_folder(folder_path):
    if os.path.exists(folder_path):
        # Removing the directory
        shutil.rmtree(folder_path)
        # print(f"The directory {folder_path} has been deleted.")


# convert seconds to days, hours, minutes, seconds
def convert_seconds(seconds):
    days = seconds // (24 * 3600)
    seconds %= (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return days, hours, minutes, seconds


def config_decoder(file_path):
    with open(file_path, 'r') as file:
        docs = yaml.safe_load(file)
        return docs


# ---------------------------------------------- Profiler decoder functions ----------------------------------------
def get_items_by_name(path, name):
    item_list = []
    with open(path, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'traceEvents.item')
        while True:
            try:
                item = objects.__next__()
                if name in item.get("name"):
                # if item.get("name") == name:
                    item_list.append(item)
            except StopIteration as e:
                break
    return sorted(item_list, key=lambda x: x.get('ts'))


# def get_items_by_names(path, names):
#     item_list = get_items_by_name(path, names[0])
#     i = 1
#     while i < len(names):
#         temp_item_list = []
#         for item in item_list:
#             sub_item_list = get_sub_item_list_of_item(path, names[i], item)
#             temp_item_list += sub_item_list
#         i += 1
#         item_list = temp_item_list
#     return sorted(item_list, key=lambda x: x.get('ts'))


def get_items_by_names(path, names):
    item_list = get_items_by_name(path, names[0])

    # item_name_list = [item.get('name') for item in item_list]
    # print(item_name_list)

    i = 1
    while i < len(names):
        if type(names[i]) == list:
            temp_item_list = []
            for item in item_list:
                temp_sub_item_list = []
                for sub_item_name in names[i]:
                    sub_item_list = get_sub_item_list_of_item(path, sub_item_name, item)
                    temp_sub_item_list += sub_item_list
                temp_item_list.append(sorted(temp_sub_item_list, key=lambda x: x.get('ts')))
                # print(len(temp_sub_item_list))
                # print('-----------------------')
            i += 1
            item_list = temp_item_list
        else: 
            temp_item_list = []
            for item in item_list:
                sub_item_list = get_sub_item_list_of_item(path, names[i], item)
                temp_item_list += sub_item_list
            i += 1
            item_list = sorted(temp_item_list, key=lambda x: x.get('ts'))
    return item_list


def get_sub_item_list_of_item(path, name, target_item):
    start_ts = target_item.get('ts')
    end_ts = start_ts + target_item.get('dur')
    cuda_runtime_list = []
    with open(path, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'traceEvents.item')
        while True:
            try:
                item = objects.__next__()
                if name in item.get("name"):
                    start_ts_temp = item.get('ts')
                    end_ts_temp = start_ts_temp + item.get('dur')
                    if start_ts_temp >= start_ts and end_ts_temp <= end_ts:
                        # print(item.get('name'))
                        # print(name)
                        # print(target_item.get('name'))
                        # print('---------------')
                        cuda_runtime_list.append(item)
            except StopIteration as e:
                break
    return sorted(cuda_runtime_list, key=lambda x: x.get('ts'))


def get_cuda_runtime_list_of_item(path, target_item):
    start_ts = target_item.get('ts')
    end_ts = start_ts + target_item.get('dur')
    cuda_runtime_list = []
    with open(path, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'traceEvents.item')
        while True:
            try:
                item = objects.__next__()
                if item.get("cat") == 'cuda_runtime':
                    start_ts_temp = item.get('ts')
                    end_ts_temp = start_ts_temp + item.get('dur')
                    if start_ts_temp >= start_ts and end_ts_temp <= end_ts:
                        cuda_runtime_list.append(item)
            except StopIteration as e:
                break
    return sorted(cuda_runtime_list, key=lambda x: x.get('ts'))


def get_cuda_runtime_list_of_item_list(path, target_item_list):
    cuda_runtime_list = []
    for target_item in target_item_list:
        temp_item_list = get_cuda_runtime_list_of_item(path, target_item)
        cuda_runtime_list += temp_item_list
    return sorted(cuda_runtime_list, key=lambda x: x.get('ts'))


def get_kernel_by_cuda_runtime_item(path, cuda_runtime_item):
    ext_id = cuda_runtime_item.get("args").get("External id")
    result = None
    with open(path, 'r', encoding='utf-8') as f:
        objects = ijson.items(f, 'traceEvents.item')
        while True:
            try:
                item = objects.__next__()
                if item.get("cat") == "kernel" or item.get("cat") == "gpu_memcpy": 
                    if item.get("args").get("External id") == ext_id:            
                        result = item
                        break
            except StopIteration as e:
                break
    return result


def get_kernel_list_by_cuda_runtime_item_list(path, cuda_runtime_item_list):
    kernel_item_list = []
    for cuda_runtime_item in cuda_runtime_item_list:
        kernel_item = get_kernel_by_cuda_runtime_item(path, cuda_runtime_item)
        if kernel_item is not None:
            kernel_item_list.append(kernel_item)
    if len(kernel_item_list) > 0:
        return sorted(kernel_item_list, key=lambda x: x.get('ts'))
    else:
        return kernel_item_list


def get_GPU_runtime_list(path, names):
    dur_list = []

    item_list = get_items_by_names(path, names)
    # item_list = get_items_by_name(path, name)

    if len(item_list) > 0:

        for item in item_list:
            if type(item) == list:
                cuda_runtime_item_list = get_cuda_runtime_list_of_item_list(path, item)
            else:
                cuda_runtime_item_list = get_cuda_runtime_list_of_item(path, item)

            kernel_item_list = get_kernel_list_by_cuda_runtime_item_list(path, cuda_runtime_item_list)


            if len(kernel_item_list) > 0:
                temp_kernel_first = kernel_item_list[0]
                temp_kernel_last = kernel_item_list[-1]
                dur = temp_kernel_last.get('ts') + temp_kernel_last.get('dur') - temp_kernel_first.get('ts')
                dur_list.append(dur)

    return dur_list


# -------------------------------------------End Profiler decoder functions ----------------------------------------



if __name__ == '__main__':
    path = './torch_profilers/NVIDIAA100-SXM4-80GB/baddbmm_fp32/20240624115807446/aisct01_1761762.1719244699346.pt.trace.json'
    name = 'aten::baddbmm'
    item_list = get_items_by_name(path, name)
    # list = get_GPU_runtime_list(path, name)
    # print(len(item_list))
    cuda_runtime_item_list = get_cuda_runtime_list_of_item(path, item_list[0])
    for item in cuda_runtime_item_list:
        print(item)