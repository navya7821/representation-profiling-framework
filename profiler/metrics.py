import torch
import torch.nn.functional as F


# -------------------- CKA --------------------

def linear_CKA(X, Y):
    X = X - X.mean(dim=1, keepdim=True)
    Y = Y - Y.mean(dim=1, keepdim=True)

    X = F.normalize(X, dim=1)
    Y = F.normalize(Y, dim=1)

    return (X * Y).sum(dim=1).mean().item()


# -------------------- GRAM MATRIX --------------------

def gram_matrix(Fmap):
    B, C, H, W = Fmap.shape
    Fmap = Fmap.view(B, C, -1)
    G = torch.bmm(Fmap, Fmap.transpose(1, 2))
    return G / (C * H * W)


# -------------------- SENSITIVITY --------------------

def compute_sensitivity(features, features_aug, top_k=2):
    """
    Returns:
        sensitivity_dict: {layer: score}
        top_layers: list of most sensitive layers
    """
    sensitivity = {}

    for layer in features:
        f1 = features[layer]
        f2 = features_aug[layer]

        # flatten assumed already (from extractor)
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)

        # CKA
        cka_score = linear_CKA(f1_flat, f2_flat)

        # L2 (normalized)
        l2 = torch.norm(f1_flat - f2_flat, p=2, dim=1).mean()
        l2_norm = (l2 / torch.norm(f1_flat, p=2, dim=1).mean()).item()

        # composite score
        score = (1 - cka_score) + l2_norm

        sensitivity[layer] = {
            "cka": cka_score,
            "l2_normalised": l2_norm,
            "score": score,
        }

    # sort by score
    sorted_layers = sorted(
        sensitivity.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    top_layers = [name for name, _ in sorted_layers[:top_k]]

    return sensitivity, top_layers


# -------------------- STABILITY --------------------

def compute_stability(features, features_aug, layers):
    """
    Computes cosine + gram similarity for selected layers
    """
    stability = {}

    for layer in layers:
        f1 = features[layer]
        f2 = features_aug[layer]

        # cosine similarity
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)

        cos_sim = F.cosine_similarity(f1_flat, f2_flat, dim=1).mean().item()

        # gram similarity (requires 4D features ideally)
        if len(f1.shape) == 4:
            G1 = gram_matrix(f1)
            G2 = gram_matrix(f2)

            gram_sim = F.cosine_similarity(
                G1.view(G1.size(0), -1),
                G2.view(G2.size(0), -1),
                dim=1
            ).mean().item()
        else:
            gram_sim = None  # not applicable if flattened

        stability[layer] = {
            "cosine": cos_sim,
            "gram": gram_sim,
        }

    return stability


# -------------------- EMBEDDING ROBUSTNESS --------------------

def compute_embedding_robustness(model, x, x_aug):
    """
    Final embedding-level robustness
    """

    with torch.no_grad():
        emb = model(x)
        emb_aug = model(x_aug)

    emb = emb.view(emb.size(0), -1)
    emb_aug = emb_aug.view(emb_aug.size(0), -1)

    cos_sim = F.cosine_similarity(emb, emb_aug, dim=1).mean()

    l2 = torch.norm(emb - emb_aug, p=2, dim=1).mean()
    l2_norm = l2 / torch.norm(emb, p=2, dim=1).mean()

    R = cos_sim * torch.exp(-l2_norm)

    return {
        "cosine": cos_sim.item(),
        "l2_normalised": l2_norm.item(),
        "score": R.item(),
    }