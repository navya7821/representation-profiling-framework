import torch
from torchvision.models import resnet18, ResNet18_Weights

from profiler.engine import model_profile_under_augmentation
from profiler.report import build_report, print_report, save_report
from profiler.utils import set_seed


def main():
    # -------------------- SETUP --------------------
    set_seed(42)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.eval()

    # -------------------- CONFIG --------------------
    config = {
        "layers": ["layer1", "layer2", "layer3", "layer4"],
        "augmentations": [
            {"name": "rotation", "params": {"degrees": 15}},
            {"name": "blur", "params": {}},
            {"name": "brightness", "params": {}},
        ],
        "mode": "individual",   # IMPORTANT
        "top_k": 2,
        "processing": "flatten",
        "input": torch.rand(1, 3, 224, 224),
    }

    # -------------------- RUN PROFILER --------------------
    results = model_profile_under_augmentation(model, config)

    # -------------------- BUILD REPORT --------------------
    report = build_report(results)

    # -------------------- OUTPUT --------------------
    print_report(report)
    save_report(report, "report.json")


if __name__ == "__main__":
    main()