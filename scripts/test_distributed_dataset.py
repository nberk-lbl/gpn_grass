import os
import datetime
import torch
import torch.distributed as dist

def main():
    # --- Perlmutter-Specific Setup ---
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["MASTER_ADDR"] = os.environ["SLURM_JOB_NODELIST"].split(",")[0].split("-")[0]
    os.environ["MASTER_PORT"] = "29500"
    
    # --- GPU Mapping ---
    gpu_ids = os.environ["SLURM_JOB_GPUS"].split(",")
    local_id = int(os.environ["SLURM_LOCALID"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[local_id]
    
    # --- NCCL Init ---
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(minutes=5),
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"])
    )
    
    try:
        # --- CUDA Device Fix ---
        torch.cuda.set_device(0)  # Always use first (and only) visible GPU
        print(f"Rank {dist.get_rank()}: Using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
        
        # --- AllReduce Test ---
        tensor = torch.tensor([1.0], device=torch.cuda.current_device())
        dist.all_reduce(tensor)
        print(f"Rank {dist.get_rank()}: AllReduce sum = {tensor.item()}")
        
    finally:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()