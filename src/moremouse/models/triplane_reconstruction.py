"""Transformer-triplane single-view mouse reconstruction model."""

import torch

from moremouse.models.dino_encoder import DinoImageEncoder


class MoReMouseTriplane(torch.nn.Module):
    """Predict mesh PCA coefficients and triplane tokens from one RGB image."""

    def __init__(self, components: int, hidden_dim: int, plane_channels: int,
                 plane_size: int, layers: int, heads: int, use_dino: bool = False) -> None:
        """Initialize image tokenizer, transformer, triplane head, and mesh head."""
        super().__init__()
        self.plane_channels = int(plane_channels)
        self.plane_size = int(plane_size)
        self.encoder = DinoImageEncoder(hidden_dim, pretrained=True) if use_dino else torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            torch.nn.GELU(),
            torch.nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.position = torch.nn.Parameter(torch.zeros(1, 16, hidden_dim))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers=layers)
        self.coeff_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 16),
            torch.nn.Linear(hidden_dim * 16, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, components),
        )
        self.plane_head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 16),
            torch.nn.Linear(hidden_dim * 16, 3 * plane_channels * plane_size * plane_size),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return normalized PCA coefficients and triplane tensors."""
        encoded = self.encoder(images)
        tokens = (encoded if encoded.ndim == 3 else encoded.flatten(2).transpose(1, 2)) + self.position
        tokens = self.transformer(tokens)
        features = tokens.flatten(1)
        triplane = self.plane_head(features)
        triplane = triplane.view(images.shape[0], 3, self.plane_channels, self.plane_size, self.plane_size)
        return {"coeffs": self.coeff_head(features), "triplanes": triplane}
