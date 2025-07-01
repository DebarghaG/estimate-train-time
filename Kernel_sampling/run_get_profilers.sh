#!/bin/bash

root_path="/mnt/rds/VipinRDS/VipinRDS/users/bxz297/distributed_training_estimator_of_LLM/Kernel_sampling"

python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/baddbmm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/bmm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/crossentropy.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/embedding.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/fillmask.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/gelu.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/layernorm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/linear_final.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/linear1.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/linear2.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/linear3.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/linear4.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/RoPE.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/softmax.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/flash_atten.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/RMSlayernorm.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/res_add.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/firstStage_optimizer.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/middleStage_optimizer.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/lastStage_optimizer.yml --precision fp16
python $root_path/sampling_controller.py --config_path $root_path/configs/profilers/ScaledUpperTriangMaskedSoftmax.yml --precision fp16