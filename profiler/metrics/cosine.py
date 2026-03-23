import torch
import torch.nn.functional as F


def cosine_similarity(f1, f2):
    """
    Layer-wise cosine similarity between feature dictionaries
    """
    results = {}

    for layer in f1:
        x = f1[layer].view(f1[layer].size(0), -1)
        y = f2[layer].view(f2[layer].size(0), -1)

        sim = F.cosine_similarity(x, y, dim=1).mean().item()
        results[layer] = sim

    return results