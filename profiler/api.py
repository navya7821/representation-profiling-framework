import json

from profiler.model_profiler import ModelProfiler
from augment import AugmentationPipeline


# -------------------- CONFIG LOADER --------------------

def _load_config(config):
    """
    Supports:
    - dict (direct use)
    - str (path to JSON file)
    """
    if config is None:
        return {}

    if isinstance(config, str):
        with open(config, "r") as f:
            return json.load(f)

    return config


# -------------------- CORE GENERIC API --------------------

def model_profiling_under_input_changes(
    model,
    input_a,
    input_b,
    config=None,
    output=None,
):
    """
    Generic representation comparison between two inputs

    Args:
        model: PyTorch model
        input_a: original input
        input_b: transformed input
        config: dict or JSON path
        output: optional file path to save report

    Returns:
        ModelProfiler instance (with results + df)
    """

    config = _load_config(config)

    layers = config.get("layers", None)
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    with ModelProfiler(model, layers=layers, processing=processing) as p:

        # group A → original
        p(x=input_a, group="group_a", tag="input_a")

        # group B → transformed
        p(x=input_b, group="group_b", tag="input_b")

        # compute metrics
        p.compute(
            metrics=metrics,
            groups=["group_a", "group_b"]
        )

        # output handling
        if output:
            p.save_as_report(output)
        else:
            p.print()

        return p


# -------------------- AUGMENTATION WRAPPER --------------------

def model_profile_under_augmentation(
    model,
    config,
    output=None,
):
    """
    Wrapper over generic API for augmentation-based evaluation
    """

    config = _load_config(config)

    layers = config.get("layers", None)
    augment_config = config.get("augmentations", None)
    mode = config.get("mode", "individual")
    processing = config.get("processing", "flatten")
    metrics = config.get("metrics", ["cosine", "linear"])

    input_a = config["input"]

    augmenter = AugmentationPipeline(augment_config, mode=mode)
    aug_outputs = augmenter(input_a)

    with ModelProfiler(model, layers=layers, processing=processing) as p:

        for aug_name, input_b in aug_outputs.items():

            # replicate input_a to ensure 1:1 pairing with augmented inputs
            p(x=input_a, group="group_a", tag="original")

            # augmented input
            p(x=input_b, group="group_b", tag=aug_name)

        p.compute(
            metrics=metrics,
            groups=["group_a", "group_b"]
        )

        if output:
            p.save_as_report(output)
        else:
            p.print()

        return p