"""
Evaluation Metrics for MoReMouse
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted image [B, H, W, 3] or [B, 3, H, W]
        target: Target image
        mask: Optional mask
        data_range: Data range (default 1.0 for [0, 1])

    Returns:
        PSNR value (higher is better)
    """
    mse = ((pred - target) ** 2)

    if mask is not None:
        mse = mse * mask.unsqueeze(-1)
        mse = mse.sum() / (mask.sum() * pred.shape[-1] + 1e-8)
    else:
        mse = mse.mean()

    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))
    return psnr


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    Compute Structural Similarity Index.

    Args:
        pred: Predicted image [B, H, W, 3] or [B, 3, H, W]
        target: Target image
        window_size: Window size for local statistics
        data_range: Data range

    Returns:
        SSIM value (higher is better, max 1.0)
    """
    # Convert to BCHW if needed
    if pred.dim() == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    pad = window_size // 2

    # Use average pooling as approximation to Gaussian
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=pad) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pred_target

    ssim = (
        (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    ) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim.mean()


def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    lpips_fn=None,
) -> torch.Tensor:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        pred: Predicted image [B, H, W, 3] or [B, 3, H, W]
        target: Target image
        lpips_fn: LPIPS model instance

    Returns:
        LPIPS value (lower is better)
    """
    if lpips_fn is None:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='vgg').to(pred.device)
        except ImportError:
            return torch.tensor(0.0, device=pred.device)

    # Convert to BCHW
    if pred.dim() == 4 and pred.shape[-1] == 3:
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    # LPIPS expects [-1, 1]
    pred = pred * 2 - 1
    target = target * 2 - 1

    return lpips_fn(pred, target).mean()


def compute_iou(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Compute Intersection over Union.

    Args:
        pred_mask: Predicted mask [B, H, W]
        target_mask: Target mask [B, H, W]
        threshold: Threshold for binarization

    Returns:
        IoU value
    """
    pred_binary = (pred_mask > threshold).float()
    target_binary = target_mask.float()

    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    iou = intersection / (union + 1e-8)
    return iou


def compute_metrics(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    lpips_fn=None,
) -> Dict[str, float]:
    """
    Compute all metrics.

    Args:
        pred: Dictionary with 'rgb', 'alpha' predictions
        target: Dictionary with 'rgb', 'mask' targets
        lpips_fn: Optional LPIPS model

    Returns:
        Dictionary of metric values
    """
    metrics = {}

    # RGB metrics
    if "rgb" in pred and "rgb" in target:
        metrics["psnr"] = compute_psnr(pred["rgb"], target["rgb"]).item()
        metrics["ssim"] = compute_ssim(pred["rgb"], target["rgb"]).item()
        metrics["lpips"] = compute_lpips(pred["rgb"], target["rgb"], lpips_fn).item()

    # Mask metrics
    if "alpha" in pred and "mask" in target:
        metrics["iou"] = compute_iou(pred["alpha"], target["mask"]).item()

    return metrics
