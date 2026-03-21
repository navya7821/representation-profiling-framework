import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # enforce deterministic behavior (important for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_tensor(x):
    """
    Ensures input is a torch tensor
    """
    if isinstance(x, torch.Tensor):
        return x
    raise TypeError("Input must be a torch.Tensor")


def to_device(x, device):
    """
    Move tensor to device
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x