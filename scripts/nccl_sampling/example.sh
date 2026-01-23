# Get master address and port
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Get the number of nodes
NNODES=$(scontrol show hostname $SLURM_NODELIST | wc -l)


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