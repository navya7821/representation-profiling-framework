import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    """
    Set all random seeds for reproducibility
    """
    torch.manual_seed(seed)

    #  Safe CUDA handling
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    # enforce deterministic behavior (important for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_tensor(x):
    """
    Ensures input is a torch tensor.
    Converts numpy arrays to tensors.
    """
    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)

    raise TypeError(f"Unsupported input type: {type(x)}. Expected torch.Tensor or numpy.ndarray.")


def to_device(x, device):
    """
    Move tensor to device.
    If input is not a tensor, it is returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)

    return x