import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: list, processing: str = "none"):
        """
        model: PyTorch model
        layers: list of layer names (strings) to hook
        processing:
            "none"   → raw features
            "flatten" → flatten to (B, -1)
            "gap"    → global average pooling to (B, C)
        """
        self.model = model
        self.layers = layers
        self.processing = processing

        self.features = {}
        self.handles = []

        self._register_hooks()

    def _get_layer_dict(self):
        return dict(self.model.named_modules())

    def _process(self, x):
        if self.processing == "none":
            return x
        elif self.processing == "flatten":
            return x.view(x.size(0), -1)
        elif self.processing == "gap":
            return F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        else:
            raise ValueError(f"Unknown processing: {self.processing}")

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = self._process(output.detach())
        return hook

    def _register_hooks(self):
        layer_dict = self._get_layer_dict()

        for name in self.layers:
            if name not in layer_dict:
                raise ValueError(f"Layer {name} not found in model")

            handle = layer_dict[name].register_forward_hook(
                self._hook_fn(name)
            )
            self.handles.append(handle)

    def clear(self):
        self.features = {}

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, x):
        self.clear()
        _ = self.model(x)
        return self.features


# -------------------- MAIN TEST --------------------

if __name__ == "__main__":
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    layers = ["layer1", "layer2", "layer3", "layer4"]

    # try different processing modes
    extractor = FeatureExtractor(model, layers, processing="flatten")

    x = torch.randn(1, 3, 224, 224)
    features = extractor(x)

    for k, v in features.items():
        print(k, v.shape)

    extractor.remove_hooks()