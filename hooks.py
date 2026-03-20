import torch
import torch.nn as nn


class FeatureExtractor:
    def __init__(self, model: nn.Module, layers: list):
        """
        model: PyTorch model
        layers: list of layer names (strings) to hook
        """
        self.model = model
        self.layers = layers
        self.features = {}
        self.handles = []

        self._register_hooks()

    def _get_layer_dict(self):
        """Map layer names to modules"""
        return dict(self.model.named_modules())

    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name] = output.detach()
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
        """Clear stored features before forward pass"""
        self.features = {}

    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __call__(self, x):
        """
        Run forward pass and collect features
        """
        self.clear()
        _ = self.model(x)
        return self.features
    
def main():
    import torchvision.models as models
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    model.eval()
    layers = ["layer1", "layer2", "layer3", "layer4"]
    extractor = FeatureExtractor(model, layers)
    x = torch.randn(1, 3, 224, 224)
    features = extractor(x)
    for k, v in features.items():
        print(k, v.shape)
        
main()