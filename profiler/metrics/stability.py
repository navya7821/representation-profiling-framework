from profiler.metrics.cosine import cosine_similarity
from profiler.metrics.gram import gram_similarity


def compute_stability(f1, f2, layers):
    """
    Stability on selected layers using cosine + gram
    """
    cos = cosine_similarity(f1, f2)
    gram = gram_similarity(f1, f2)

    stability = {}

    for layer in layers:
        stability[layer] = {
            "cosine": cos[layer],
            "gram": gram[layer],
        }

    return stability