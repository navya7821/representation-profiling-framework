import torch

from augment import AugmentationPipeline
from profiler.model_profiler import ModelProfiler


def model_profile_under_augmentation(model, config):
    """
    Wrapper API:
    Keeps your original behavior but uses the new generic profiler internally
    """

    model.eval()

    # -------------------- CONFIG --------------------

    layers = config.get("layers", None)
    augment_config = config.get("augmentations", None)
    mode = config.get("mode", "individual")
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    x = config["input"]

    # -------------------- AUGMENTATION --------------------

    augmenter = AugmentationPipeline(augment_config, mode=mode)
    aug_outputs = augmenter(x)

    # -------------------- PROFILING --------------------

    with ModelProfiler(model, layers=layers, processing=processing) as p:

        #  Ensure 1-to-1 pairing and track augmentation names
        for aug_name, x_aug in aug_outputs.items():

            # original → group_a (no tag needed)
            p(x=x, group="group_a", tag=None)

            # augmented → group_b (attach augmentation name)
            p(x=x_aug, group="group_b", tag=aug_name)

        # compute metrics
        p.compute(
            metrics=metrics,
            groups=["group_a", "group_b"]
        )

        return {
            "raw_results": p.results,
            "dataframe": p.df,
            "num_augmentations": len(aug_outputs),
        }