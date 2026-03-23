import torch


def l2_normalized(f1, f2):
    """
    Normalized L2 distance per layer
    """
    results = {}

    for layer in f1:
        x = f1[layer].view(f1[layer].size(0), -1)
        y = f2[layer].view(f2[layer].size(0), -1)

        l2 = torch.norm(x - y, p=2, dim=1).mean()
        norm = torch.norm(x, p=2, dim=1).mean()

        results[layer] = (l2 / norm).item()

    return results