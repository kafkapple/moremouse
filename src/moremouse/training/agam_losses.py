"""Losses for the anchor-based AGAM proxy."""

from __future__ import annotations

import torch

from moremouse.data.agam import TorchGaussianAvatar


def gaussian_avatar_loss(prediction: TorchGaussianAvatar, target: TorchGaussianAvatar) -> dict[str, torch.Tensor]:
    """Return the component losses used by the AGAM regression proxy."""
    center = torch.nn.functional.smooth_l1_loss(prediction.centers, target.centers)
    color = torch.nn.functional.mse_loss(prediction.colors, target.colors)
    scale = torch.nn.functional.smooth_l1_loss(prediction.scales, target.scales)
    opacity = torch.nn.functional.binary_cross_entropy(prediction.opacities.clamp(1e-4, 1 - 1e-4),
                                                        target.opacities.clamp(1e-4, 1 - 1e-4))
    rotation = quaternion_identity_regularizer(prediction.rotations)
    total = center + color + scale + opacity + 0.05 * rotation
    return {
        "total": total,
        "center": center,
        "color": color,
        "scale": scale,
        "opacity": opacity,
        "rotation": rotation,
    }


def quaternion_identity_regularizer(rotations: torch.Tensor) -> torch.Tensor:
    """Penalize rotations away from identity-like unit quaternions."""
    if rotations.ndim != 3 or rotations.shape[-1] != 4:
        raise ValueError("rotations must have shape [batch, anchors, 4]")
    identity = torch.zeros_like(rotations)
    identity[..., 0] = 1.0
    dot = torch.abs((rotations * identity).sum(dim=-1)).clamp(0.0, 1.0)
    return 1.0 - dot.mean()
