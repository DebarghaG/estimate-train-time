import os
import torch
import torch.distributed as dist


def allreduce(shapes, precision):

    if precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    local_rank = int(os.environ['LOCAL_RANK'])

    tensor = torch.rand(shapes, dtype=dtype, device=local_rank, requires_grad=False)

    # Synchronize all processes before the all_reduce operation
    # dist.barrier()

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # world_size = int(os.environ['WORLD_SIZE'])
    # local_size = int(os.environ['LOCAL_WORLD_SIZE'])
    # global_rank = int(os.environ['RANK'])
    # print(f'world size:{world_size} local size:{local_size} global rank:{global_rank} local rank:{local_rank}')


def p2p(shapes, precision):

    if precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    local_rank = int(os.environ['LOCAL_RANK'])

    tensor = torch.rand(shapes, dtype=dtype, device=local_rank, requires_grad=False)

    global_rank = int(os.environ['RANK'])

    world_size = int(os.environ['WORLD_SIZE'])

    # Synchronize all processes before next function
    # dist.barrier()

    if global_rank == 0:
        # Send tensor from rank 0 to rank 1
        dist.send(tensor=tensor, dst=world_size-1)
    elif global_rank == world_size-1:
        # Receive tensor on rank 1 from rank 0
        dist.recv(tensor=tensor, src=0)


def allgather(shapes, precision):

    if precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    local_rank = int(os.environ['LOCAL_RANK'])

    tensor = torch.rand(shapes, dtype=dtype, device=local_rank, requires_grad=False)

    gather_list = [torch.zeros_like(tensor) for _ in range(int(os.environ['WORLD_SIZE']))]

    # Synchronize all processes before next function
    # dist.barrier()

    # Perform all_reduce operation
    dist.all_gather(gather_list, tensor)

    # cleanup()


def reducescatter(shapes, precision):

    if precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32

    chunk_size = shapes[0] // int(os.environ['WORLD_SIZE'])

    local_rank = int(os.environ['LOCAL_RANK'])

    input_tensor = torch.rand(shapes, dtype=dtype, device=local_rank, requires_grad=False)

    output_tensor = torch.zeros([chunk_size], dtype=dtype, device=local_rank, requires_grad=False)

    input_list = list(input_tensor.chunk(int(os.environ['WORLD_SIZE'])))

    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)


