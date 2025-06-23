# Distributed Training Estimator of LLMs
Time cost estimator of LLM's distributed training

- AI4CI


# Tutorials

### Predictor
In order to run Predictor, the training configurations, computing and communication operators' sampling data are required. In the "example_training_config" folder, there are two example configuration YAML files. The "regressors" folder contains the required data obtained from two real clusters as examples.
   ```bash
   cd Estimator
   python mml_3d_prediction.py --config_path <path_to_config.yml>
   ```



### Operator Sampling
The computing operator sampling module requires the configuration of each operator in the form of a YAML file. The "/configs/collect" and "/configs/test" directories provide details about the configuration files.
   ```bash
   cd Kernel_sampling
   ## For example sampling the baddbmm with fp16 
   python sampling_controller.py --config_path ./configs/collect/baddbmm.yml --precision fp16 
   ```
The files "run_collection.sh" and "run_test.sh" contain details about how to test and collect the sampling data for each operator. The "--parts" option specifies how many parts the sampling work should be split into, and the "--part" option specifies which part of the work is being processed on current GPU.


### Communication Sampling
This part, like the Operator Sampling, also requires the configuration of each communication operator in the form of a YAML file. The "/configs/collect" and "/configs/test" directories provide details about the configuration files. The example shows how to collect P2P communication between two nodes, with only one GPU being active on each node.
   ```bash
    # Get master address and port
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    # Get the number of nodes
    NNODES=$(scontrol show hostname $SLURM_NODELIST | wc -l)

    # Active GPU 0 of each node
    srun --export=ALL,CUDA_VISIBLE_DEVICES=0 torchrun \
    --nnodes $NNODES \
    --nproc_per_node 1 \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    sampling_controller.py \
    --config_path ./configs/test/p2p.yml \
    --precision fp16 \
    --parts 1 \
    --part 1
   ```





### Acknowledgements

*National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)*
