import torch
import os 
import torch.distributed as dist
import argparse as args
def init_distributed_mode():
    '''initilize DDP 
    '''
    # os.environ["WORLD_SIZE"]='2'
    # os.environ["LOCAL_RANK"]='2'
    # os.environ["RANK"] = '2'
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpu = int(os.environ["LOCAL_RANK"])
    print(os.environ["RANK"],os.environ["WORLD_SIZE"],os.environ["LOCAL_RANK"])
        # print(os.environ["LOCAL_RANK"])
    # elif "SLURM_PROCID" in os.environ:
    #     args.rank = int(os.environ["SLURM_PROCID"])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # elif hasattr(args, "rank"):
    #     pass
    # else:
    #     print("Not using distributed mode")
    #     args.distributed = False
    #     return

    args.distributed = True
    # print(args.gpu)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    # print(args.gpu)
    print(f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    
    return args.world_size