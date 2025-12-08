"""
Reconstruction Loss Functions

Standard image reconstruction losses for training.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted values
            target: Target values
            mask: Optional mask (1 = valid, 0 = ignore)
        """
        loss = (pred - target) ** 2

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() * pred.shape[-1] + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class L1Loss(nn.Module):
    """L1 (Mean Absolute Error) loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.abs(pred - target)

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() * pred.shape[-1] + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss.

    Penalizes large RGB discrepancies more than small ones.
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(pred, target, beta=self.beta, reduction='none')

        if mask is not None:
            loss = loss * mask.unsqueeze(-1)
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() * pred.shape[-1] + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss.

    Higher SSIM = more similar -> loss = 1 - SSIM
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channel: int = 3,
        data_range: float = 1.0,
    ):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range

        # Create Gaussian window
        self.register_buffer(
            "window",
            self._create_gaussian_window(window_size, sigma, channel)
        )

    def _create_gaussian_window(
        self,
        window_size: int,
        sigma: float,
        channel: int,
    ) -> torch.Tensor:
        """Create Gaussian kernel for SSIM computation."""
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()

        window_1d = g.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channel, 1, window_size, window_size).contiguous()

        return window

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, H, W, C] or [B, C, H, W] predicted images
            target: [B, H, W, C] or [B, C, H, W] target images
            mask: Optional [B, H, W] mask
        """
        # Convert to BCHW if needed
        if pred.dim() == 4 and pred.shape[-1] == self.channel:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        B, C, H, W = pred.shape
        window = self.window.to(pred.device)

        # Constants
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        pad = self.window_size // 2

        # Means
        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_target = F.conv2d(target, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # Variances
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pred_target

        # SSIM
        ssim = (
            (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        ) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )

        # Average over channels
        ssim = ssim.mean(dim=1)  # [B, H, W]

        if mask is not None:
            ssim = ssim * mask
            return 1 - ssim.sum() / (mask.sum() + 1e-8)

        return 1 - ssim.mean()


class LPIPSLoss(nn.Module):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) loss.

    Uses VGG features for perceptual similarity.
    """

    def __init__(self, net: str = "vgg"):
        super().__init__()

        try:
            import lpips
            self.lpips = lpips.LPIPS(net=net)
            self._available = True
        except ImportError:
            print("Warning: lpips not installed. LPIPS loss will return 0.")
            self._available = False

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: [B, H, W, 3] or [B, 3, H, W] predicted images in [0, 1]
            target: [B, H, W, 3] or [B, 3, H, W] target images in [0, 1]
            mask: Optional mask (not fully supported by LPIPS)
        """
        if not self._available:
            return torch.tensor(0.0, device=pred.device)

        # Convert to BCHW
        if pred.dim() == 4 and pred.shape[-1] == 3:
            pred = pred.permute(0, 3, 1, 2)
            target = target.permute(0, 3, 1, 2)

        # LPIPS expects [-1, 1]
        pred = pred * 2 - 1
        target = target * 2 - 1

        # Move LPIPS to correct device
        self.lpips = self.lpips.to(pred.device)

        loss = self.lpips(pred, target)
        return loss.mean()
