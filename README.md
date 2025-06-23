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
The computing operator sampling moudule needs the configurations of each operator as a yaml file. "/configs/collect" or "/configs/test" provide detials about the config file.
   ```bash
   cd Kernel_sampling
   ## For example sampling the baddbmm with fp16 
   python sampling_controller.py --config_path ./configs/collect/baddbmm.yml --precision fp16 
   ```
The file "run_collection.sh" and "run_test.sh" have detials about how to test and collection the sampling data of each operators. "--parts" means how many parts do you want to split the sampling work and "--part" measn the process doing the job of which part of parts.


### Communication Sampling
As same as the Operator Sampling, this part also needs the configurations of each communication operator as a yaml file. "/configs/collect" or "/configs/test" provide detials about the config file. The example of collecting p2p communication among two nodes and only active one GPU of each node.
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
