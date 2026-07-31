"""
Distributed training utilities for Q-GAD system.

This module provides GPU parallel training support:
- DataParallel for single-node multi-GPU
- DistributedDataParallel for multi-node training
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from typing import Optional, List
import os


def setup_distributed(backend: str = "nccl") -> bool:
    """
    Setup distributed training environment.

    Args:
        backend: Distributed backend ('nccl', 'gloo', or 'mpi')

    Returns:
        True if distributed setup is successful, False otherwise
    """
    if not torch.cuda.is_available():
        print("CUDA not available, cannot setup distributed training")
        return False

    try:
        # Check if NCCL is available (recommended for GPU)
        if backend == "nccl" and not torch.cuda.nccl.is_available(torch.cuda.current_device()):
            print("NCCL not available, falling back to gloo")
            backend = "gloo"

        # Initialize process group
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

            torch.distributed.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size
            )

            # Set device for this process
            torch.cuda.set_device(rank)

            print(f"Distributed training initialized: rank={rank}/{world_size}")
            return True
        else:
            print("Distributed environment variables not set (RANK, WORLD_SIZE)")
            return False

    except Exception as e:
        print(f"Failed to setup distributed training: {e}")
        return False


class DistributedDataParallel(nn.Module):
    """
    Wrapper for DistributedDataParallel with special handling for GBS circuits.

    Since GBS circuits are created dynamically per sample, standard DDP
    requires special handling.
    """

    def __init__(self, model: nn.Module, device_ids: Optional[List[int]] = None):
        """
        Initialize DDP wrapper.

        Args:
            model: Model to parallelize
            device_ids: List of GPU device IDs. If None, use all available GPUs.
        """
        super().__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) > 1:
            # Use DistributedDataParallel for multi-GPU
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=device_ids,
                output_device=device_ids[0],
                find_unused_parameters=True
            )
            self.is_parallel = True
            print(f"Using DistributedDataParallel on GPUs: {device_ids}")
        else:
            self.model = model
            self.is_parallel = False
            print(f"Single GPU mode: {device_ids[0] if device_ids else 'cpu'}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def train(self, mode: bool = True):
        """Set training mode."""
        if self.is_parallel:
            self.model.train(mode)
        else:
            self.model.train(mode)

    def eval(self):
        """Set evaluation mode."""
        if self.is_parallel:
            self.model.eval()
        else:
            self.model.eval()


class SimpleDataParallel(nn.Module):
    """
    Simple DataParallel wrapper for single-node multi-GPU training.

    This is easier to use than DDP and sufficient for single-node training.
    """

    def __init__(self, model: nn.Module, device_ids: Optional[List[int]] = None):
        """
        Initialize DataParallel wrapper.

        Args:
            model: Model to parallelize
            device_ids: List of GPU device IDs
        """
        super().__init__()

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids=device_ids)
            self.is_parallel = True
            print(f"Using DataParallel on GPUs: {device_ids}")
        else:
            self.model = model
            self.is_parallel = False

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def get_parallel_model(model: nn.Module, use_ddp: bool = False) -> nn.Module:
    """
    Get a parallel version of the model.

    Args:
        model: Original model
        use_ddp: Whether to use DistributedDataParallel (vs DataParallel)

    Returns:
        Parallelized model or original model if only one GPU
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        print(f"Single GPU available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        return model

    print(f"Multi-GPU training: {n_gpus} GPUs detected")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    print(f"GPU devices: {gpu_names}")

    if use_ddp:
        # Use DistributedDataParallel (requires process spawning)
        return DistributedDataParallel(model)
    else:
        # Use simple DataParallel (easier to use)
        return SimpleDataParallel(model)


def adjust_batch_size(batch_size: int, n_gpus: int) -> int:
    """
    Adjust batch size to be divisible by number of GPUs.

    Args:
        batch_size: Original batch size
        n_gpus: Number of GPUs

    Returns:
        Adjusted batch size
    """
    if batch_size % n_gpus != 0:
        new_batch_size = ((batch_size // n_gpus) + 1) * n_gpus
        print(f"Adjusted batch size: {batch_size} -> {new_batch_size} (divisible by {n_gpus})")
        return new_batch_size
    return batch_size


def get_device_for_rank() -> torch.device:
    """
    Get the appropriate device for the current process in distributed training.

    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            return torch.device(f"cuda:{rank}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    print("Testing distributed utilities...")

    # Test GPU detection
    n_gpus = torch.cuda.device_count()
    print(f"GPUs available: {n_gpus}")

    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Test batch size adjustment
    original_batch = 16
    adjusted_batch = adjust_batch_size(original_batch, n_gpus if n_gpus > 0 else 1)
    print(f"Batch size adjustment: {original_batch} -> {adjusted_batch}")
