"""
Depth Loss

Depth consistency loss within object regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    """
    Depth consistency loss.

    Only computed within object mask region.
    Uses scale-invariant depth loss to handle depth ambiguity.
    """

    def __init__(
        self,
        scale_invariant: bool = True,
        reduction: str = "mean",
    ):
        super().__init__()
        self.scale_invariant = scale_invariant
        self.reduction = reduction

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_depth: Predicted depth [B, H, W]
            target_depth: Target depth [B, H, W]
            mask: Valid region mask [B, H, W]
        """
        if mask is None:
            mask = torch.ones_like(pred_depth)

        mask = mask.float()

        if self.scale_invariant:
            # Scale-invariant depth loss
            # Normalize predictions to same scale as target within mask
            pred_masked = pred_depth * mask
            target_masked = target_depth * mask

            # Compute optimal scale
            scale = (pred_masked * target_masked).sum() / (
                (pred_masked ** 2).sum() + 1e-8
            )

            pred_scaled = pred_depth * scale
            diff = (pred_scaled - target_depth) * mask

        else:
            diff = (pred_depth - target_depth) * mask

        loss = diff ** 2

        if self.reduction == "mean":
            return loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DepthSmoothLoss(nn.Module):
    """
    Depth smoothness loss.

    Encourages smooth depth transitions, weighted by image gradients.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        depth: torch.Tensor,
        image: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            depth: Depth map [B, H, W]
            image: RGB image [B, H, W, 3] for edge-aware weighting
            mask: Valid region [B, H, W]
        """
        # Compute depth gradients
        grad_x = torch.abs(depth[:, :, :-1] - depth[:, :, 1:])
        grad_y = torch.abs(depth[:, :-1, :] - depth[:, 1:, :])

        if image is not None:
            # Edge-aware weighting
            if image.dim() == 4 and image.shape[-1] == 3:
                image = image.mean(dim=-1)  # To grayscale

            img_grad_x = torch.abs(image[:, :, :-1] - image[:, :, 1:])
            img_grad_y = torch.abs(image[:, :-1, :] - image[:, 1:, :])

            weight_x = torch.exp(-img_grad_x)
            weight_y = torch.exp(-img_grad_y)

            grad_x = grad_x * weight_x
            grad_y = grad_y * weight_y

        loss = grad_x.mean() + grad_y.mean()

        return loss
