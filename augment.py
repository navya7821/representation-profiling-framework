import torch
import kornia.augmentation as K


class AugmentationPipeline:
    def __init__(self):
        self.transforms = torch.nn.Sequential(
            K.RandomRotation(degrees=15.0, p=1.0),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=1.0),
            K.RandomBrightness(brightness=(0.8, 1.2), p=1.0),
        )

    def __call__(self, x):
        return self.transforms(x)


# -------------------- MAIN TEST --------------------

if __name__ == "__main__":
    x = torch.rand(1, 3, 224, 224)
    augment = AugmentationPipeline()
    x_aug = augment(x)

    print("Original shape:", x.shape)
    print("Augmented shape:", x_aug.shape)

    print("Original stats:", x.min().item(), x.max().item())
    print("Augmented stats:", x_aug.min().item(), x_aug.max().item())