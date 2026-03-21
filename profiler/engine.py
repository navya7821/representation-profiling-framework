import torch

from augment import AugmentationPipeline
from hooks import FeatureExtractor

from profiler.metrics import (
    compute_sensitivity,
    compute_stability,
    compute_embedding_robustness,
)


def model_profile_under_augmentation(model, config):
    """
    Core pipeline:
    - apply augmentations
    - extract features
    - compute metrics
    - return structured results
    """

    model.eval()

    # -------------------- CONFIG --------------------

    layers = config["layers"]
    augment_config = config.get("augmentations", None)
    mode = config.get("mode", "individual")
    top_k = config.get("top_k", 2)
    processing = config.get("processing", "flatten")

    # -------------------- COMPONENTS --------------------

    extractor = FeatureExtractor(model, layers, processing=processing)
    augmenter = AugmentationPipeline(augment_config, mode=mode)

    # -------------------- INPUT --------------------

    x = config["input"]  # tensor expected

    # -------------------- ORIGINAL FEATURES --------------------

    features = extractor(x)

    # -------------------- AUGMENTED FEATURES --------------------

    aug_outputs = augmenter(x)

    results = {}

    # -------------------- LOOP OVER AUGMENTATIONS --------------------

    for aug_name, x_aug in aug_outputs.items():

        features_aug = extractor(x_aug)

        # ---------- SENSITIVITY ----------
        sensitivity, sensitive_layers = compute_sensitivity(
            features, features_aug, top_k=top_k
        )

        # ---------- STABILITY ----------
        stability = compute_stability(
            features, features_aug, sensitive_layers
        )

        # ---------- EMBEDDING ROBUSTNESS ----------
        robustness = compute_embedding_robustness(model, x, x_aug)

        # ---------- STORE ----------
        results[aug_name] = {
            "sensitivity": sensitivity,
            "top_sensitive_layers": sensitive_layers,
            "stability": stability,
            "embedding_robustness": robustness,
        }

    return results