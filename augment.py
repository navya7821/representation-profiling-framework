import torch
import kornia.augmentation as K


AUGMENTATION_REGISTRY = {
    "rotation": lambda cfg: K.RandomRotation(
        degrees=cfg.get("degrees", 15.0), p=1.0
    ),
    "blur": lambda cfg: K.RandomGaussianBlur(
        kernel_size=cfg.get("kernel_size", (3, 3)),
        sigma=cfg.get("sigma", (0.1, 2.0)),
        p=1.0,
    ),
    "brightness": lambda cfg: K.RandomBrightness(
        brightness=cfg.get("brightness", (0.8, 1.2)),
        p=1.0,
    ),
}


class AugmentationPipeline:
    def __init__(self, augmentations_config=None, mode="sequential"):
        """
        augmentations_config: list of dicts
            Example:
            [
                {"name": "rotation", "params": {"degrees": 15}},
                {"name": "blur", "params": {"kernel_size": (3,3)}}
            ]

        mode:
            "sequential" → apply all augmentations in sequence
            "individual" → apply each augmentation separately
        """
        self.mode = mode

        # 🔹 store (name, module) pairs for consistency
        self.augmentations = []

        if augmentations_config is None:
            augmentations_config = [
                {"name": "rotation", "params": {}},
                {"name": "blur", "params": {}},
                {"name": "brightness", "params": {}},
            ]

        for aug in augmentations_config:
            name = aug["name"]
            params = aug.get("params", {})

            if name not in AUGMENTATION_REGISTRY:
                raise ValueError(f"Unknown augmentation: {name}")

            module = AUGMENTATION_REGISTRY[name](params)
            self.augmentations.append((name, module))

        # 🔹 sequential pipeline (modules only)
        if self.mode == "sequential":
            modules = [module for _, module in self.augmentations]
            self.pipeline = torch.nn.Sequential(*modules)

    def __call__(self, x):
        """
        Applies augmentations to input tensor x
        Assumes x is already on correct device
        """

        if self.mode == "sequential":
            # 🔹 create meaningful combined name
            combined_name = "+".join([name for name, _ in self.augmentations])
            return {combined_name: self.pipeline(x)}

        elif self.mode == "individual":
            outputs = {}

            for name, aug in self.augmentations:
                outputs[name] = aug(x)

            return outputs

        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# -------------------- MAIN TEST --------------------

if __name__ == "__main__":
    x = torch.rand(1, 3, 224, 224)

    config = [
        {"name": "rotation", "params": {"degrees": 20}},
        {"name": "blur", "params": {}},
    ]

    augment = AugmentationPipeline(config, mode="individual")
    outputs = augment(x)

    for k, v in outputs.items():
        print(f"{k}: shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}")