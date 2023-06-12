import torch
import os 
import torch.distributed as dist
import argparse as args
def init_distributed_mode():
    '''initilize DDP 
    '''

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    print(os.environ["RANK"],os.environ["WORLD_SIZE"],os.environ["LOCAL_RANK"])
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    
    return args.world_size