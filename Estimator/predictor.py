import pandas as pd
import numpy as np

import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestRegressor

import ast

import tools
import os
import importlib
import math


class Predictor:
    def __init__(self):
        self.operator_dict = {}
        self.nccl_dict = {}


    def rebuild_rforest(self, best_params, X_train, y_train):
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        return model

    
    def reload_rforest(self, model_path):
        # .pkl file
        return joblib.load(model_path)


    def rebuild_xgboost(self, best_params, X_train, y_train):
        model = xgb.XGBRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        return model


    def reload_xgboost(self, model_path):
        # .json file
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model


    def rebuild_nccl_model(self, gpu_name, model_name, model_dict, config_folder, data_folder):
        # GPU_Name = torch.cuda.get_device_name(0).replace(' ', '')
        config_path = config_folder + '/' + gpu_name + '.csv'
        
        df = pd.read_csv(config_path)
        df['Best_config'] = df['Best_config'].apply(ast.literal_eval)
        
        features = model_name.split('_')
        precision = features[-3]
        nodes = features[-2]
        gpus_per_node = features[-1]
        operator_name = '_'.join(features[0:-3])
        column_indexes = [0]

        data_path = data_folder + '/' + gpu_name + '_' + model_name + '.csv'
        model_path = data_folder + '/' + gpu_name + '_' + model_name

        if not os.path.isfile(data_path) and (not os.path.isfile(model_path+'.json') or not os.path.isfile(model_path+'.pkl')):
            data_path = data_folder + '/' + gpu_name + '_' + operator_name + '_' + precision + '.csv'
            model_path = data_folder + '/' + gpu_name + '_' + operator_name + '_' + precision
            column_indexes = [0, 1, 2]
            dict_key_name = operator_name + '_' + precision
            if dict_key_name in model_dict:
                return model_dict[dict_key_name]
            target_i = df[(df['Function'] == operator_name) & (df['Precision'] == precision) & (df['Nodes'] == 0) & (df['GPUsPerNode'] == 0)].index[0]
        else:
            dict_key_name = model_name
            target_i = df[(df['Function'] == operator_name) & (df['Precision'] == precision) & (df['Nodes'] == int(nodes)) & (df['GPUsPerNode'] == int(gpus_per_node))].index[0]

        model_type = df.iloc[target_i]['Best_model']

        if model_type == 'xgboost':
            model_path = model_path + '.json'
        else:
            model_path = model_path + '.pkl'

        if not os.path.isfile(model_path):
            best_params = df.iloc[target_i]['Best_config']

            data = pd.read_csv(data_path).to_numpy()
            
            X_train = data[:, column_indexes]
            y_train = data[:, -1]
            print(f'Rebuilding {data_path}')
            
            if model_type == 'xgboost':
                model = self.rebuild_xgboost(best_params, X_train, y_train)
                model.save_model(model_path)
            else:
                model = self.rebuild_rforest(best_params, X_train, y_train)
                joblib.dump(model, model_path)
        else:
            print(f'Loading {model_path}')
            if model_type == 'xgboost':
                model = self.reload_xgboost(model_path)
            else:
                model = self.reload_rforest(model_path)

        model_dict[dict_key_name] = model

        return model


    def predict_nccl_operator(self, gpu_name, model_name, shape, model_dict, config_folder, data_folder):
        data_path = data_folder + '/' + gpu_name + '_' + model_name + '.csv'
        if os.path.isfile(data_path):
            dict_key_name = model_name
        else:
            dict_key_name = '_'.join(model_name.split('_')[0:-2])

        if dict_key_name in model_dict:
            model = model_dict[dict_key_name]
        else:
            model = self.rebuild_nccl_model(gpu_name, model_name, model_dict, config_folder, data_folder)
        
        if os.path.isfile(data_path):
            predict_input = [[shape]]
            predict = model.predict(predict_input)
        else:
            features = model_name.split('_')
            nodes = features[-2]
            gpus_per_node = features[-1]
            predict_input = [[shape, int(nodes), int(gpus_per_node)]]
            predict = model.predict(predict_input)

        print(f'Function:{model_name}\t Input:{shape}\t Prediction:{predict[0]}')

        return predict[0]


    def mp_allreduce(self, nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, gpu_per_node):
        nccl_function = 'allreduce_fp16'
        mp_gpu_per_node = mp if mp // gpu_per_node == 0 else gpu_per_node
        nodes = 1 if mp // mp_gpu_per_node == 0 else mp // mp_gpu_per_node
        nccl_function = '_'.join([nccl_function, str(nodes), str(mp_gpu_per_node)])

        # print(nccl_function)

        timecost = self.predict_nccl_operator(gpu_name, nccl_function, shape, self.nccl_dict, nccl_config_folder, nccl_data_folder)
        return timecost


    def dp_allreduce(self, nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, gpu_per_node):
        nccl_function = 'allreduce_large_fp16'
        dp_gpu_per_node = 1 if gpu_per_node // mp ==0 else gpu_per_node // mp
        nodes = 1 if dp // dp_gpu_per_node == 0 else dp // dp_gpu_per_node
        nccl_function = '_'.join([nccl_function, str(nodes), str(dp_gpu_per_node)])
        timecost = self.predict_nccl_operator(gpu_name, nccl_function, shape, self.nccl_dict, nccl_config_folder, nccl_data_folder)
        return timecost


    def dp_allgather(self, nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, gpu_per_node):
        nccl_function = 'allgather_large_fp16'
        dp_gpu_per_node = 1 if gpu_per_node // mp ==0 else gpu_per_node // mp
        nodes = 1 if dp // dp_gpu_per_node == 0 else dp // dp_gpu_per_node
        nccl_function = '_'.join([nccl_function, str(nodes), str(dp_gpu_per_node)])
        timecost = self.predict_nccl_operator(gpu_name, nccl_function, shape, self.nccl_dict, nccl_config_folder, nccl_data_folder)
        return timecost


    def pp_p2p(self, nccl_data_folder, nccl_config_folder, gpu_name, shape, mp, dp, pp, gpu_per_node):
        nccl_function = 'p2p_fp16'
        if mp*dp*pp >= gpu_per_node:
            nodes = 2
            pp_gpu_per_node = 1
        else:
            nodes = 1
            pp_gpu_per_node = 2
        nccl_function = '_'.join([nccl_function, str(nodes), str(pp_gpu_per_node)])
        timecost = self.predict_nccl_operator(gpu_name, nccl_function, shape//mp, self.nccl_dict, nccl_config_folder, nccl_data_folder)
        return timecost



    def rebuild_operator_model(self, gpu_name, model_name, model_dict, config_folder, data_folder):
        # GPU_Name = torch.cuda.get_device_name(0).replace(' ', '')
        config_path = config_folder + '/' + gpu_name + '.csv'
        
        df = pd.read_csv(config_path)
        df['Best_config'] = df['Best_config'].apply(ast.literal_eval)
        
        precision_str, propagation_str = model_name.split('_')[-2:]
        function_str = '_'.join(model_name.split('_')[0:-2])

        target_i = df[(df['Function'] == function_str) & (df['Precision'] == precision_str) & (df['Propagation'] == propagation_str)].index[0]

        model_type = df.iloc[target_i]['Best_model']
        
        data_path = data_folder + '/' + gpu_name + '_' + function_str + '_' + precision_str + '.csv'

        model_path = data_folder + '/' + gpu_name + '_' + function_str + '_' + precision_str + '_' + propagation_str

        if model_type == 'xgboost':
            model_path = model_path + '.json'
        else:
            model_path = model_path + '.pkl'

        if not os.path.isfile(model_path):
            best_params = df.iloc[target_i]['Best_config']

            data = pd.read_csv(data_path).to_numpy()

            if propagation_str == 'dur':
                X_train = data[:, 0:-1]
            else:    
                X_train = data[:, 0:-2]

            map_function = self.get_map_function('layer_input_to_predictor_input', function_str)
            X_train = np.apply_along_axis(map_function, axis=1, arr=X_train)
            
            if propagation_str == 'fwd':
                y_train = data[:, -2]
            else:
                y_train = data[:, -1]

            print(f'Rebuilding {data_path}')
            if model_type == 'xgboost':
                model = self.rebuild_xgboost(best_params, X_train, y_train)
                model.save_model(model_path)
            else:
                model = self.rebuild_rforest(best_params, X_train, y_train)
                joblib.dump(model, model_path)
        else:
            print(f'Loading {model_path}')
            if model_type == 'xgboost':
                model = self.reload_xgboost(model_path)
            else:
                model = self.reload_rforest(model_path)

        model_dict[model_name] = model

        return model


    def predict_operator(self, gpu_name, function, precision, propagation, shape, model_dict, config_folder, data_folder):
        model_name = function + '_' + precision + '_' + propagation
        if model_name in model_dict:
            model = model_dict[model_name]
        else:
            model = self.rebuild_operator_model(gpu_name, model_name, model_dict, config_folder, data_folder)

        map_function = self.get_map_function('layer_input_to_predictor_input', function)
        shape_new = map_function(shape)
        
        predict = model.predict([shape_new])

        print(f'Function:{model_name}\t Input:{shape}\t PredictorInput:{shape_new}\t Prediction:{predict[0]}')

        return predict[0]


    def get_layer_input_shape(self, encoder_config, module_name, function_name):
        module = importlib.import_module(module_name)
        return getattr(module, function_name)(encoder_config)


    def get_map_function(self, module_name, function_name):
        module = importlib.import_module(module_name)
        return getattr(module, function_name)


    def operator_statistic(self, data_folder, config_folder, gpu_name, function_list, precision, encoder_config, propagation):
        time_cost = 0

        for function in function_list:
            module_name = 'encoder_config_to_layer_input'
            shape = self.get_layer_input_shape(encoder_config, module_name, function)
            result = self.predict_operator(gpu_name, function, precision, propagation, shape, self.operator_dict, config_folder, data_folder)
            time_cost += result
            
        return time_cost



# if __name__ == '__main__':
#     # mp_allreduce_test()
#     # dp_allreduce_test()
#     # dp_allgather_test()
#     # pp_p2p_test()