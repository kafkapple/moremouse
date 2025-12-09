"""
Gaussian Mouse Avatar (AGAM)

UV-based Gaussian avatar for laboratory mouse.
Each Gaussian is controlled by UV coordinates on the mesh surface,
enabling pose-dependent deformation via Linear Blend Skinning.

Reference: MoReMouse paper Section 3.1
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mouse_body import MouseBodyModel


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (wxyz convention)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (wxyz convention)."""
    # Shepperd's method
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] is largest
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] is largest
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    # Normalize
    q = F.normalize(q, dim=-1)

    return q.reshape(*batch_shape, 4)


class GaussianAvatar(nn.Module):
    """
    Gaussian Mouse Avatar (AGAM) for synthetic data generation.

    Each Gaussian point encodes:
    - Position offset from mesh surface
    - RGB color
    - Opacity
    - Scale (3D anisotropic)
    - Rotation (quaternion)

    All parameters are UV-mapped for consistent deformation.

    Args:
        body_model: MouseBodyModel instance
        num_gaussians_per_vertex: Number of Gaussians per mesh vertex
        init_scale: Initial scale for Gaussians
        opacity_init: Initial opacity value
    """

    def __init__(
        self,
        body_model: MouseBodyModel,
        num_gaussians_per_vertex: int = 1,
        init_scale: float = 0.01,
        min_scale: float = 0.001,
        max_scale: float = 0.1,
        opacity_init: float = 0.8,
    ):
        super().__init__()

        self.body_model = body_model
        self.num_gaussians_per_vertex = num_gaussians_per_vertex
        self.num_vertices = body_model.num_vertices
        self.num_gaussians = self.num_vertices * num_gaussians_per_vertex

        self.min_scale = min_scale
        self.max_scale = max_scale

        # Initialize Gaussian parameters
        # Position offsets (delta mu)
        self.register_parameter(
            "position_offsets",
            nn.Parameter(torch.zeros(self.num_gaussians, 3))
        )

        # RGB colors (before sigmoid)
        self.register_parameter(
            "colors_raw",
            nn.Parameter(torch.zeros(self.num_gaussians, 3))
        )

        # Opacity (before sigmoid)
        opacity_raw = torch.ones(self.num_gaussians) * self._inverse_sigmoid(opacity_init)
        self.register_parameter(
            "opacity_raw",
            nn.Parameter(opacity_raw)
        )

        # Scales (log scale for positivity)
        log_scale = math.log(init_scale)
        self.register_parameter(
            "log_scales",
            nn.Parameter(torch.ones(self.num_gaussians, 3) * log_scale)
        )

        # Rotations (quaternion wxyz)
        quaternions = torch.zeros(self.num_gaussians, 4)
        quaternions[:, 0] = 1.0  # Identity rotation
        self.register_parameter(
            "quaternions",
            nn.Parameter(quaternions)
        )

        # Vertex-to-Gaussian mapping
        self.register_buffer(
            "vertex_indices",
            torch.arange(self.num_vertices).repeat_interleave(num_gaussians_per_vertex)
        )

    @staticmethod
    def _inverse_sigmoid(x: float) -> float:
        """Inverse of sigmoid function."""
        return math.log(x / (1 - x))

    def get_colors(self) -> torch.Tensor:
        """Get RGB colors [N, 3] in [0, 1]."""
        return torch.sigmoid(self.colors_raw)

    def get_opacity(self) -> torch.Tensor:
        """Get opacity [N] in [0, 1]."""
        return torch.sigmoid(self.opacity_raw)

    def get_scales(self) -> torch.Tensor:
        """Get scales [N, 3] in [min_scale, max_scale]."""
        scales = torch.exp(self.log_scales)
        return torch.clamp(scales, self.min_scale, self.max_scale)

    def get_rotations(self) -> torch.Tensor:
        """Get normalized quaternions [N, 4]."""
        return F.normalize(self.quaternions, dim=-1)

    def forward(
        self,
        pose: torch.Tensor,
        bone_lengths: torch.Tensor = None,
        center_bone_length: torch.Tensor = None,
        trans: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute posed Gaussian parameters.

        Args:
            pose: [B, J*3] joint angles
            bone_lengths: [B, 28] bone length parameters
            center_bone_length: [B, 1] center bone scale
            trans: [B, 3] global translation
            scale: [B, 1] global scale

        Returns:
            Dictionary with:
            - means: [B, N, 3] Gaussian positions
            - colors: [B, N, 3] RGB colors
            - opacity: [B, N] opacity values
            - scales: [B, N, 3] anisotropic scales
            - rotations: [B, N, 4] quaternion rotations
        """
        B = pose.shape[0]
        device = pose.device

        # Get posed mesh vertices
        V, J = self.body_model(pose, bone_lengths, center_bone_length, trans, scale)

        # Base positions: vertices corresponding to each Gaussian
        base_positions = V[:, self.vertex_indices]  # [B, N, 3]

        # Add position offsets
        means = base_positions + self.position_offsets.unsqueeze(0)

        # Get other parameters (same for all batch items)
        colors = self.get_colors().unsqueeze(0).expand(B, -1, -1)
        opacity = self.get_opacity().unsqueeze(0).expand(B, -1)
        scales = self.get_scales().unsqueeze(0).expand(B, -1, -1)
        rotations = self.get_rotations().unsqueeze(0).expand(B, -1, -1)

        # For proper deformation, we should also transform rotations
        # based on local coordinate frame of mesh surface
        # This is a simplified version; full implementation would use
        # vertex normals and tangent frames

        return {
            "means": means,
            "colors": colors,
            "opacity": opacity,
            "scales": scales,
            "rotations": rotations,
        }

    def render(
        self,
        gaussian_params: Dict[str, torch.Tensor],
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Render Gaussians to image using gsplat.

        Args:
            gaussian_params: Output from forward()
            viewmat: [B, 4, 4] camera view matrices
            K: [B, 3, 3] camera intrinsics
            width: Image width
            height: Image height

        Returns:
            rgb: [B, H, W, 3] rendered images
            alpha: [B, H, W] alpha masks
        """
        try:
            from gsplat.rendering import rasterization
        except ImportError:
            raise ImportError("gsplat is required for rendering. Install with: pip install gsplat")

        B = gaussian_params["means"].shape[0]
        device = gaussian_params["means"].device

        rendered_images = []
        rendered_alphas = []

        for b in range(B):
            means = gaussian_params["means"][b]
            quats = gaussian_params["rotations"][b]
            scales = gaussian_params["scales"][b]
            colors = gaussian_params["colors"][b]
            opacities = gaussian_params["opacity"][b]

            # Render using gsplat
            # backgrounds must be [C] or [H, W, C] for gsplat
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat[b:b+1],
                Ks=K[b:b+1],
                width=width,
                height=height,
                backgrounds=torch.ones(3, device=device),  # [C] format
            )

            rendered_images.append(render_colors[0])  # [H, W, 3]
            rendered_alphas.append(render_alphas[0])  # [H, W]

        rgb = torch.stack(rendered_images, dim=0)  # [B, H, W, 3]
        alpha = torch.stack(rendered_alphas, dim=0)  # [B, H, W]

        return rgb, alpha


class GaussianAvatarTrainer:
    """
    Trainer for Gaussian Mouse Avatar.

    Uses multi-view images to optimize Gaussian parameters.
    Loss: L1 + SSIM + LPIPS

    Reference: MoReMouse paper - 800 frames, 400k iterations
    """

    def __init__(
        self,
        avatar: GaussianAvatar,
        lr: float = 1e-3,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.1,
        device: torch.device = None,
    ):
        self.avatar = avatar
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.avatar.to(self.device)

        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips

        # Optimizer
        self.optimizer = torch.optim.Adam(avatar.parameters(), lr=lr)

        # LPIPS loss
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
        except ImportError:
            print("Warning: lpips not installed. LPIPS loss will be skipped.")
            self.lpips_fn = None

    def compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
    ) -> torch.Tensor:
        """Compute SSIM loss."""
        # Simple SSIM implementation
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Convert to BCHW
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target

        ssim = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

        return 1 - ssim.mean()

    def train_step(
        self,
        pose: torch.Tensor,
        target_images: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            pose: [B, J*3] pose parameters
            target_images: [B, H, W, 3] target images
            viewmats: [B, 4, 4] view matrices
            Ks: [B, 3, 3] intrinsics
            width: Image width
            height: Image height

        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()

        # Forward pass
        gaussian_params = self.avatar(pose.to(self.device))
        rgb, alpha = self.avatar.render(
            gaussian_params,
            viewmats.to(self.device),
            Ks.to(self.device),
            width,
            height,
        )

        target = target_images.to(self.device)

        # L1 loss
        l1_loss = F.l1_loss(rgb, target)

        # SSIM loss
        ssim_loss = self.compute_ssim(rgb, target)

        # LPIPS loss
        if self.lpips_fn is not None:
            # LPIPS expects BCHW in [-1, 1]
            pred_lpips = rgb.permute(0, 3, 1, 2) * 2 - 1
            target_lpips = target.permute(0, 3, 1, 2) * 2 - 1
            lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=self.device)

        # Total loss
        total_loss = l1_loss + self.lambda_ssim * ssim_loss + self.lambda_lpips * lpips_loss

        # Backward
        total_loss.backward()
        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "lpips": lpips_loss.item(),
        }
