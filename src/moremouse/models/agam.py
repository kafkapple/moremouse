"""Anchor-based AGAM proxy model for author-level MoReMouse reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from moremouse.data.agam import AgamTemplate, TorchGaussianAvatar
from moremouse.models.dino_encoder import DinoImageEncoder


@dataclass(frozen=True)
class AgamPrediction:
    """Model output for a batch of Gaussian avatars."""

    avatar: TorchGaussianAvatar
    rotations: torch.Tensor
    latent: torch.Tensor


class AgamAvatarModel(torch.nn.Module):
    """Predict anchor-wise Gaussian avatar parameters from a monocular image."""

    def __init__(
        self,
        template: AgamTemplate,
        hidden_dim: int,
        use_dino: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
    ) -> None:
        super().__init__()
        self.anchor_count = int(template.anchor_indices.shape[0])
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
        block = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = torch.nn.TransformerEncoder(block, num_layers=transformer_layers)
        self.shared = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim * 16),
            torch.nn.Linear(hidden_dim * 16, hidden_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.GELU(),
        )
        self.center_head = torch.nn.Linear(hidden_dim, self.anchor_count * 3)
        self.color_head = torch.nn.Linear(hidden_dim, self.anchor_count * 3)
        self.scale_head = torch.nn.Linear(hidden_dim, self.anchor_count)
        self.opacity_head = torch.nn.Linear(hidden_dim, self.anchor_count)
        self.rotation_head = torch.nn.Linear(hidden_dim, self.anchor_count * 4)

        self.register_buffer("base_centers", torch.from_numpy(template.centers))
        self.register_buffer("base_colors", torch.from_numpy(template.colors))
        self.register_buffer("base_scales", torch.from_numpy(template.scales))
        self.register_buffer("base_opacities", torch.from_numpy(template.opacities))

    def forward(self, images: torch.Tensor) -> AgamPrediction:
        """Predict Gaussian avatar parameters for a batch of images."""
        tokens = self.encode(images)
        return self.decode_tokens(tokens)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into spatial tokens."""
        features = self.encoder(images)
        if features.ndim == 4:
            return features.flatten(2).transpose(1, 2)
        if features.ndim == 3:
            return features
        if features.ndim == 2:
            return features.unsqueeze(1).repeat(1, 16, 1)
        raise ValueError(f"Unsupported encoder feature rank: {features.ndim}")

    def decode_tokens(self, tokens: torch.Tensor) -> AgamPrediction:
        """Decode image tokens into Gaussian avatar parameters."""
        if tokens.shape[1] != 16:
            raise ValueError(f"Expected 16 tokens, got {tokens.shape[1]}")
        tokens = self.transformer(tokens + self.position)
        latent = self.shared(tokens.flatten(1))
        centers = self.base_centers.unsqueeze(0) + 0.15 * torch.tanh(self.center_head(latent).view(-1, self.anchor_count, 3))
        colors = torch.sigmoid(self.color_head(latent).view(-1, self.anchor_count, 3))
        base_scale = torch.log(self.base_scales.clamp_min(1e-4)).unsqueeze(0)
        scales = functional.softplus(self.scale_head(latent).view(-1, self.anchor_count) + base_scale)
        opacities = torch.sigmoid(self.opacity_head(latent).view(-1, self.anchor_count))
        rotations = functional.normalize(self.rotation_head(latent).view(-1, self.anchor_count, 4), dim=-1)
        avatar = TorchGaussianAvatar(
            centers=centers,
            colors=colors,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
        )
        return AgamPrediction(avatar=avatar, rotations=rotations, latent=latent)
