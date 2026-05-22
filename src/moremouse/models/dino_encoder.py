"""DINOv2 image encoder wrapper with an offline fallback."""

import torch
import torch.nn.functional as functional
from urllib.error import URLError


class DinoImageEncoder(torch.nn.Module):
    """Use DINOv2 when available, otherwise keep the same contract with a CNN fallback."""

    def __init__(self, output_dim: int, pretrained: bool = True) -> None:
        """Initialize encoder and projection head."""
        super().__init__()
        self.output_dim = int(output_dim)
        self.backbone = self._load_dinov2() if pretrained else None
        if self.backbone is None:
            self.backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                torch.nn.GELU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(64, output_dim, kernel_size=3, stride=2, padding=1),
                torch.nn.GELU(),
                torch.nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.projection = torch.nn.Identity()
        else:
            self.projection = torch.nn.Linear(768, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return 16 spatial tokens."""
        inputs = functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        features = self.backbone(inputs)
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 2:
            return self.projection(features).unsqueeze(1).repeat(1, 16, 1)
        raise ValueError(f"Unsupported DINO feature rank: {features.ndim}")

    def _load_dinov2(self) -> torch.nn.Module | None:
        """Load torch-hub DINOv2 if the local environment already supports it."""
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
        except (RuntimeError, OSError, URLError):
            return None
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model
