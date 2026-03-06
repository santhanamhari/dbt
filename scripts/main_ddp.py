#!/usr/bin/env python
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add Mirai to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# scripts/main_ddp.py (top-level)
import os, time, datetime
import torch
import torch.distributed as dist
import math
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from operator import itemgetter
from typing import Optional

from torch.utils.data import Dataset, Sampler, DistributedSampler


class DatasetFromSampler(Dataset):
    def __init__(self, sampler: Sampler):
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        if self.sampler_list is None:  # we don't instantiate the list in __init__ because want to shuffle first (happens in DistributedSamplerWrapper.__iter__)
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):

        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

# === SLURM Environment Setup ===-------------------------------------------------------------------------------------------------------------------
def setup_slurm_env():
    """Map SLURM variables to PyTorch distributed variables"""
    if "SLURM_PROCID" in os.environ:
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))
        global_rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 1))
        
        os.environ["RANK"] = str(global_rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        
        return local_rank, global_rank, world_size
    else:
        return 0, 0, 1

local_rank, global_rank, world_size = setup_slurm_env()

# Set CUDA device BEFORE any imports that use CUDA
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)

# Now import Mirai modules
import onconet.datasets.factory as dataset_factory
import onconet.models.factory as model_factory
from onconet.learn import train
import onconet.transformers.factory as transformer_factory
import onconet.utils.parsing as parsing
import onconet.learn.state_keeper as state

def setup_distributed():
    """Initialize distributed training"""
    print(f"[DEBUG][rank {global_rank}][host {os.uname().nodename}] MASTER_ADDR={os.environ.get('MASTER_ADDR')} MASTER_PORT={os.environ.get('MASTER_PORT')} WORLD_SIZE={os.environ.get('WORLD_SIZE')}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=global_rank
    )
    
    if global_rank == 0:
        print(f"Initialized distributed training:")
        print(f"  World size: {world_size}")
        print(f"  Nodes: {os.environ.get('SLURM_NODELIST', 'unknown')}")
        print(f"  Master: {os.environ.get('MASTER_ADDR')}:{os.environ.get('MASTER_PORT')}")

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def main():
    # Parse arguments
    args = parsing.parse_args()
    
    # Override args for distributed training
    args.cuda = True
    args.device = f'cuda:{local_rank}'
    args.local_rank = local_rank
    args.global_rank = global_rank
    args.world_size = world_size
    
    # Adjust batch size for distributed training
    # Each GPU processes batch_size / world_size samples
    args.batch_size = args.batch_size // world_size

    if args.batch_size < 1:
        args.batch_size = 1
    
    # Only rank 0 prints/saves
    args.is_master = (global_rank == 0)
    
    if args.is_master:
        print("\n" + "="*60)
        print("DISTRIBUTED TRAINING CONFIGURATION")
        print("="*60)
        print(f"Global rank: {global_rank}/{world_size}")
        print(f"Local rank: {local_rank}")
        print(f"Device: {args.device}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print("="*60 + "\n")
    
    # Initialize distributed
    setup_distributed()
    
    # Load data
    if args.is_master:
        print("\nLoading data...")
    
    transformers = transformer_factory.get_transformers(
        args.image_transformers, args.tensor_transformers, args)
    test_transformers = transformer_factory.get_transformers(
        args.test_image_transformers, args.test_tensor_transformers, args)
    
    train_data, dev_data, test_data = dataset_factory.get_dataset(
        args, transformers, test_transformers)


    if args.class_bal or args.year_weighted_class_bal:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=train_data.weights,
            num_samples=len(train_data),
            replacement=True
        )
        train_sampler = DistributedSamplerWrapper(sampler, num_replicas=world_size, rank=global_rank, shuffle=True)
    
    else:
        # Wrap datasets with DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=True
        )
    
    dev_sampler = torch.utils.data.distributed.DistributedSampler(
        dev_data,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False
    )
    
    # Override dataloaders in training code (we'll modify train.py)
    args.train_sampler = train_sampler
    args.dev_sampler = dev_sampler
    
    # Load model
    if args.is_master:
        print("\nLoading model...")
    
    if args.snapshot is None:
        model = model_factory.get_model(args)
    else:
        model = model_factory.load_model(args.snapshot, args)
    
    # Move model to correct device
    model = model.to(args.device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True  # Set True if you get errors
    )
    
    if args.is_master:
        print(f"\nModel wrapped with DistributedDataParallel")
        print(f"Model device: {next(model.parameters()).device}\n")
    
    # Override save_dir to avoid conflicts

    if not args.is_master:
        args.save_dir = f"{args.save_dir}_rank{global_rank}"
        os.makedirs(args.save_dir, exist_ok=True)
   
    # Train
    if args.train:
        if args.is_master:
            print("Starting training...\n")
        
        epoch_stats, model = train.train_model(train_data, dev_data, model, args)
        args.epoch_stats = epoch_stats
    
    # Evaluate (only on rank 0)
    if args.dev and args.is_master:
        print("\n" + "="*60)
        print("EVALUATING ON DEV SET")
        print("="*60)
        args.dev_stats = train.eval_model(dev_data, model, args)

        # Save results
        import pickle
        save_path = args.results_path
        print(f"\nSaving results to {save_path}")
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))

    if args.test and args.is_master:
        print("\n" + "="*60)
        print("EVALUATING ON TEST SET")
        print("="*60)
        args.test_stats = train.eval_model(test_data, model, args)
        
        # Save results
        import pickle
        save_path = args.results_path
        print(f"\nSaving results to {save_path}")
        args_dict = vars(args)
        pickle.dump(args_dict, open(save_path, 'wb'))
    
    # Cleanup
    cleanup_distributed()
    
    if args.is_master:
        print("\nTraining completed successfully!")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[Rank {global_rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        cleanup_distributed()
        sys.exit(1)