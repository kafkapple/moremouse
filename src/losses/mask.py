"""
Mask Loss

Binary cross-entropy for opacity/mask prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskLoss(nn.Module):
    """
    Binary cross-entropy loss for mask prediction.

    Compares predicted alpha/opacity with ground truth mask.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred_alpha: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_alpha: Predicted opacity [B, H, W] in [0, 1]
            target_mask: Target mask [B, H, W] in {0, 1}
        """
        # Ensure same shape
        if pred_alpha.shape != target_mask.shape:
            target_mask = F.interpolate(
                target_mask.unsqueeze(1).float(),
                size=pred_alpha.shape[-2:],
                mode='nearest',
            ).squeeze(1)

        # Convert to logits for AMP compatibility
        # Use binary_cross_entropy_with_logits (safe for autocast)
        pred_clamped = pred_alpha.clamp(1e-7, 1 - 1e-7)
        pred_logits = torch.log(pred_clamped / (1 - pred_clamped))  # inverse sigmoid

        loss = F.binary_cross_entropy_with_logits(
            pred_logits,
            target_mask.float(),
            reduction=self.reduction,
        )

        return loss


class DiceLoss(nn.Module):
    """
    Dice loss for mask segmentation.

    Complements BCE by handling class imbalance.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        pred_alpha: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_alpha: Predicted opacity [B, H, W] in [0, 1]
            target_mask: Target mask [B, H, W] in {0, 1}
        """
        pred = pred_alpha.flatten()
        target = target_mask.flatten().float()

        intersection = (pred * target).sum()
        dice = (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1 - dice
