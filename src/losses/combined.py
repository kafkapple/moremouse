"""
Combined Loss Module

Combines all individual losses for MoReMouse training.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .reconstruction import MSELoss, L1Loss, SSIMLoss, LPIPSLoss, SmoothL1Loss
from .mask import MaskLoss
from .depth import DepthLoss
from .geodesic import GeodesicLoss


class MoReMouseLoss(nn.Module):
    """
    Combined loss for MoReMouse training.

    Total loss = λ_mse * L_mse + λ_lpips * L_lpips + λ_mask * L_mask
                 + λ_smooth * L_smooth + λ_depth * L_depth + λ_geo * L_geo

    Default weights from paper:
    - λ_mse = 1.0
    - λ_lpips = 1.0
    - λ_mask = 0.3
    - λ_smooth = 0.2
    - λ_depth = 0.2
    - λ_geo = 0.1

    Args:
        lambda_mse: Weight for MSE loss
        lambda_lpips: Weight for LPIPS loss
        lambda_mask: Weight for mask loss
        lambda_smooth: Weight for smooth L1 loss
        lambda_depth: Weight for depth loss
        lambda_geo: Weight for geodesic embedding loss
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_lpips: float = 1.0,
        lambda_mask: float = 0.3,
        lambda_smooth: float = 0.2,
        lambda_depth: float = 0.2,
        lambda_geo: float = 0.1,
    ):
        super().__init__()

        # Loss weights
        self.lambda_mse = lambda_mse
        self.lambda_lpips = lambda_lpips
        self.lambda_mask = lambda_mask
        self.lambda_smooth = lambda_smooth
        self.lambda_depth = lambda_depth
        self.lambda_geo = lambda_geo

        # Loss functions
        self.mse_loss = MSELoss()
        self.lpips_loss = LPIPSLoss()
        self.mask_loss = MaskLoss()
        self.smooth_loss = SmoothL1Loss()
        self.depth_loss = DepthLoss()
        self.geo_loss = GeodesicLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            pred: Dictionary with:
                - rgb: [B, H, W, 3] predicted RGB
                - alpha: [B, H, W] predicted alpha
                - depth: [B, H, W] predicted depth (optional)
                - embedding: [B, H, W, 3] predicted embedding (optional)

            target: Dictionary with:
                - rgb: [B, H, W, 3] target RGB
                - mask: [B, H, W] target mask
                - depth: [B, H, W] target depth (optional)
                - embedding: [B, H, W, 3] target embedding (optional)

        Returns:
            Dictionary with:
                - total: Combined loss
                - mse: MSE loss value
                - lpips: LPIPS loss value
                - mask: Mask loss value
                - smooth: Smooth L1 loss value
                - depth: Depth loss value
                - geo: Geodesic loss value
        """
        losses = {}
        total_loss = 0.0

        # Get mask for valid regions
        mask = target.get("mask", None)

        # MSE loss
        if self.lambda_mse > 0:
            mse = self.mse_loss(pred["rgb"], target["rgb"], mask)
            losses["mse"] = mse
            total_loss = total_loss + self.lambda_mse * mse

        # LPIPS loss
        if self.lambda_lpips > 0:
            lpips = self.lpips_loss(pred["rgb"], target["rgb"], mask)
            losses["lpips"] = lpips
            total_loss = total_loss + self.lambda_lpips * lpips

        # Mask loss
        if self.lambda_mask > 0 and "alpha" in pred and mask is not None:
            mask_loss = self.mask_loss(pred["alpha"], mask)
            losses["mask"] = mask_loss
            total_loss = total_loss + self.lambda_mask * mask_loss

        # Smooth L1 loss
        if self.lambda_smooth > 0:
            smooth = self.smooth_loss(pred["rgb"], target["rgb"], mask)
            losses["smooth"] = smooth
            total_loss = total_loss + self.lambda_smooth * smooth

        # Depth loss
        if self.lambda_depth > 0 and "depth" in pred and "depth" in target:
            depth = self.depth_loss(pred["depth"], target["depth"], mask)
            losses["depth"] = depth
            total_loss = total_loss + self.lambda_depth * depth

        # Geodesic embedding loss
        if self.lambda_geo > 0 and "embedding" in pred and "embedding" in target:
            geo = self.geo_loss(pred["embedding"], target["embedding"])
            losses["geo"] = geo
            total_loss = total_loss + self.lambda_geo * geo

        losses["total"] = total_loss

        return losses


class AvatarTrainingLoss(nn.Module):
    """
    Loss for Gaussian Avatar training.

    L = L1 + λ_ssim * L_ssim + λ_lpips * L_lpips

    Reference: MoReMouse paper - λ_ssim = 0.2, λ_lpips = 0.1
    """

    def __init__(
        self,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.1,
    ):
        super().__init__()

        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips

        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
        self.lpips_loss = LPIPSLoss()

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_rgb: [B, H, W, 3] predicted RGB
            target_rgb: [B, H, W, 3] target RGB

        Returns:
            Dictionary of loss values
        """
        losses = {}

        # L1 loss
        l1 = self.l1_loss(pred_rgb, target_rgb)
        losses["l1"] = l1

        # SSIM loss
        ssim = self.ssim_loss(pred_rgb, target_rgb)
        losses["ssim"] = ssim

        # LPIPS loss
        lpips = self.lpips_loss(pred_rgb, target_rgb)
        losses["lpips"] = lpips

        # Total
        total = l1 + self.lambda_ssim * ssim + self.lambda_lpips * lpips
        losses["total"] = total

        return losses
