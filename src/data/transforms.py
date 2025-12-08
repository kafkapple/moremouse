"""
Data Transforms for MoReMouse
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Resize:
    """Resize image to target size."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            img.unsqueeze(0),
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)


class Normalize:
    """Normalize to [0, 1] or ImageNet stats."""

    def __init__(self, mean: Tuple = None, std: Tuple = None):
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            return (img - mean) / std
        return img


class RandomHorizontalFlip:
    """Random horizontal flip with pose adjustment."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < self.p:
            return TF.hflip(img)
        return img


class ColorJitter:
    """Random color jittering."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # ColorJitter expects [C, H, W]
        return self.transform(img)


class RandomRotation:
    """Random rotation within specified range."""

    def __init__(self, degrees: float = 15.0):
        self.degrees = degrees

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        angle = (torch.rand(1).item() * 2 - 1) * self.degrees
        return TF.rotate(img, angle)


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img


def get_transforms(
    mode: str = "train",
    image_size: int = 378,
    augment: bool = True,
) -> Compose:
    """
    Get transforms for training or evaluation.

    Args:
        mode: "train" or "eval"
        image_size: Target image size
        augment: Whether to apply augmentation (train only)

    Returns:
        Transform pipeline
    """
    transforms = [Resize(image_size)]

    if mode == "train" and augment:
        transforms.extend([
            ColorJitter(brightness=0.1, contrast=0.1),
            # Note: Horizontal flip requires pose adjustment
            # RandomHorizontalFlip(p=0.5),
        ])

    return Compose(transforms)
