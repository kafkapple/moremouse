"""Small single-view image-to-mesh PCA model."""

import torch


class SingleViewMeshMvp(torch.nn.Module):
    """Predict mesh PCA coefficients from one RGB image."""

    def __init__(self, components: int, hidden_dim: int) -> None:
        """Initialize the image encoder and coefficient head."""
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
            torch.nn.Flatten(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, components),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict PCA coefficient vectors."""
        return self.head(self.encoder(images))
