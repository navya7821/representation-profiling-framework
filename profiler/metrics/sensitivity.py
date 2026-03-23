from profiler.metrics.linear import linear_similarity
from profiler.metrics.l2 import l2_normalized


def compute_sensitivity(f1, f2, top_k=2):
    """
    Composite sensitivity using CKA + normalized L2
    """
    cka = linear_similarity(f1, f2)
    l2 = l2_normalized(f1, f2)

    sensitivity = {}

    for layer in f1:
        score = (1 - cka[layer]) + l2[layer]

        sensitivity[layer] = {
            "cka": cka[layer],
            "l2_normalised": l2[layer],
            "score": score,
        }

    # sort layers
    sorted_layers = sorted(
        sensitivity.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    top_layers = [name for name, _ in sorted_layers[:top_k]]

    return sensitivity, top_layers