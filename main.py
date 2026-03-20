import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import random
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights


# -------------------- SET SEED --------------------

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# -------------------- SAFE CKA --------------------

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


# -------------------- FEATURE EXTRACTOR --------------------

class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: list):
        self.model = model
        self.layers = layers
        self.features = {}
        self.handles = []
        self._register_hooks()

    def _get_layer_dict(self):
        return dict(self.model.named_modules())

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
        return hook

    def _register_hooks(self):
        layer_dict = self._get_layer_dict()

        for name in self.layers:
            handle = layer_dict[name].register_forward_hook(
                self._hook_fn(name)
            )
            self.handles.append(handle)

    def clear(self):
        self.features = {}

    def __call__(self, x):
        self.clear()
        _ = self.model(x)
        return self.features


# -------------------- AUGMENTATION --------------------

class AugmentationPipeline:
    def __init__(self):
        self.transforms = nn.Sequential(
            K.RandomRotation(degrees=15.0, p=1.0),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0),
            K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
        )

    def __call__(self, x):
        return self.transforms(x)


# -------------------- MAIN --------------------

if __name__ == "__main__":
    set_seed(42)

    # model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    # layers
    layers = ["layer1", "layer2", "layer3", "layer4"]
    extractor = FeatureExtractor(model, layers)

    # augmentation
    augment = AugmentationPipeline()

    # input
    x = torch.rand(1, 3, 224, 224)
    x_aug = augment(x)

    # forward
    features = extractor(x)
    features_aug = extractor(x_aug)

    # -------------------- PRINT FEATURES --------------------

    print("\n--- ORIGINAL FEATURES ---")
    for k, v in features.items():
        print(k, v.shape)

    print("\n--- AUGMENTED FEATURES ---")
    for k, v in features_aug.items():
        print(k, v.shape)

    # -------------------- LAYER SENSITIVITY (CKA + L2) --------------------

    print("\n--- LAYER SENSITIVITY (CKA + L2) ---")

    sensitivity = {}

    for layer in features:
        f1 = features[layer]
        f2 = features_aug[layer]

        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)

        # CKA
        cka_score = linear_CKA(f1_flat, f2_flat)

        # L2
        l2 = torch.norm(f1_flat - f2_flat, p=2, dim=1).mean()
        l2_norm = (l2 / torch.norm(f1_flat, p=2, dim=1).mean()).item()

        # combined score
        score = (1 - cka_score) + l2_norm
        sensitivity[layer] = score

        print(f"{layer}: CKA={cka_score:.4f}, L2={l2_norm:.4f}, score={score:.4f}")

    # -------------------- SELECT SENSITIVE LAYERS --------------------

    k = 2
    sorted_layers = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
    sensitive_layers = [name for name, _ in sorted_layers[:k]]

    print("\n--- MOST SENSITIVE LAYERS ---")
    for name in sensitive_layers:
        print(name)

    # -------------------- REPRESENTATION STABILITY (Cosine + Gram) --------------------

    print("\n--- REPRESENTATION STABILITY (Cosine + Gram) ---")

    for layer in sensitive_layers:
        f1 = features[layer]
        f2 = features_aug[layer]

        # cosine
        f1_flat = f1.view(f1.size(0), -1)
        f2_flat = f2.view(f2.size(0), -1)
        cos_sim = F.cosine_similarity(f1_flat, f2_flat, dim=1).mean().item()

        # gram
        G1 = gram_matrix(f1)
        G2 = gram_matrix(f2)

        gram_sim = F.cosine_similarity(
            G1.view(G1.size(0), -1),
            G2.view(G2.size(0), -1),
            dim=1
        ).mean().item()

        print(f"{layer}: cosine={cos_sim:.4f}, gram={gram_sim:.4f}")

    # -------------------- EMBEDDING ROBUSTNESS --------------------

    print("\n--- EMBEDDING ROBUSTNESS ---")

    with torch.no_grad():
        emb = model(x)
        emb_aug = model(x_aug)

    emb = emb.view(emb.size(0), -1)
    emb_aug = emb_aug.view(emb_aug.size(0), -1)

    cos_sim = F.cosine_similarity(emb, emb_aug, dim=1).mean()

    l2 = torch.norm(emb - emb_aug, p=2, dim=1).mean()
    l2_norm = l2 / torch.norm(emb, p=2, dim=1).mean()

    R = cos_sim * torch.exp(-l2_norm)

    print(f"cosine: {cos_sim.item():.4f}")
    print(f"l2_norm: {l2_norm.item():.4f}")
    print(f"Robustness Score R: {R.item():.4f}")