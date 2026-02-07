"""
Gaussian Mouse Avatar (AGAM)

UV-based Gaussian avatar for laboratory mouse.
Each Gaussian is controlled by UV coordinates on the mesh surface,
enabling pose-dependent deformation via Linear Blend Skinning.

Key features (based on MoReMouse paper Section 3.1):
- UV-parameterized Gaussian positions on mesh surface
- Local tangent frame for rotation optimization
- LBS deformation: μ'_Ψl = D_Ψl(μ0 + Δμ_Ψl)
- Loss: L1 + SSIM + LPIPS + TV (total variation)

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

    UV-based implementation following MoReMouse paper:
    - Gaussians are anchored to UV coordinates on mesh surface
    - Position offsets are defined in local tangent frame
    - Rotations are optimized in local coordinate system
    - LBS deformation: μ'_Ψl = D_Ψl(μ0 + Δμ_Ψl)

    Each Gaussian point encodes:
    - Position offset from mesh surface (in local tangent frame)
    - RGB color
    - Opacity
    - Scale (3D anisotropic)
    - Rotation (quaternion in local frame)

    Args:
        body_model: MouseBodyModel instance with UV coordinates
        num_gaussians_per_vertex: Number of Gaussians per mesh vertex
        init_scale: Initial scale for Gaussians
        min_scale: Minimum allowed scale
        max_scale: Maximum allowed scale
        opacity_init: Initial opacity value
        use_local_frame: Whether to use local tangent frame for offsets/rotations
    """

    def __init__(
        self,
        body_model: MouseBodyModel,
        num_gaussians_per_vertex: int = 1,
        init_scale: float = 0.005,  # Reduced from 0.01 for finer detail
        min_scale: float = 0.0005,  # Reduced from 0.001
        max_scale: float = 0.05,    # Reduced from 0.1
        opacity_init: float = 0.9,   # Increased from 0.8
        use_local_frame: bool = True,
    ):
        super().__init__()

        self.body_model = body_model
        self.num_gaussians_per_vertex = num_gaussians_per_vertex
        self.use_local_frame = use_local_frame

        # Use UV-based or vertex-based Gaussians
        if body_model.has_uv:
            # UV-based: one Gaussian per UV coordinate
            self.num_anchor_points = body_model.num_uv_coords
            self.use_uv = True
            # UV to vertex mapping for position lookup
            self.register_buffer("anchor_to_vert", body_model.uv_to_vert)
        else:
            # Fallback to vertex-based
            self.num_anchor_points = body_model.num_vertices
            self.use_uv = False
            self.register_buffer("anchor_to_vert", torch.arange(body_model.num_vertices))

        self.num_vertices = body_model.num_vertices
        self.num_gaussians = self.num_anchor_points * num_gaussians_per_vertex

        self.min_scale = min_scale
        self.max_scale = max_scale

        # Initialize Gaussian parameters
        # Position offsets (delta mu) - in LOCAL tangent frame
        # Small random initialization for better coverage
        position_offsets = torch.randn(self.num_gaussians, 3) * 0.001
        self.register_parameter(
            "position_offsets",
            nn.Parameter(position_offsets)
        )

        # RGB colors (before sigmoid) - initialize to dark color (mouse ~0.15-0.2)
        # sigmoid(-1.7) ≈ 0.15, which matches typical mouse color
        colors_raw = torch.randn(self.num_gaussians, 3) * 0.1 - 1.7
        self.register_parameter(
            "colors_raw",
            nn.Parameter(colors_raw)
        )

        # Opacity (before sigmoid) - higher initial opacity
        opacity_raw = torch.ones(self.num_gaussians) * self._inverse_sigmoid(opacity_init)
        self.register_parameter(
            "opacity_raw",
            nn.Parameter(opacity_raw)
        )

        # Scales (log scale for positivity) - per-axis variation
        log_scale = math.log(init_scale)
        # Add small variation to scales
        log_scales = torch.ones(self.num_gaussians, 3) * log_scale
        log_scales += torch.randn_like(log_scales) * 0.1
        self.register_parameter(
            "log_scales",
            nn.Parameter(log_scales)
        )

        # Rotations (quaternion wxyz) - in LOCAL tangent frame
        quaternions = torch.zeros(self.num_gaussians, 4)
        quaternions[:, 0] = 1.0  # Identity rotation in local frame
        # Add small perturbation for diversity
        quaternions[:, 1:] = torch.randn(self.num_gaussians, 3) * 0.01
        self.register_parameter(
            "quaternions",
            nn.Parameter(quaternions)
        )

        # Anchor point indices (UV coords or vertices)
        self.register_buffer(
            "anchor_indices",
            torch.arange(self.num_anchor_points).repeat_interleave(num_gaussians_per_vertex)
        )

        # For backward compatibility
        self.register_buffer(
            "vertex_indices",
            self.anchor_to_vert[self.anchor_indices]
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
        Compute posed Gaussian parameters using local tangent frame.

        Following MoReMouse paper:
        μ'_Ψl = D_Ψl(μ0 + Δμ_Ψl)

        Position offsets (Δμ) are defined in local tangent frame and
        transformed to world space using the tangent frame (T, B, N).

        Args:
            pose: [B, J*3] joint angles
            bone_lengths: [B, 28] bone length parameters
            center_bone_length: [B, 1] center bone scale
            trans: [B, 3] global translation
            scale: [B, 1] global scale

        Returns:
            Dictionary with:
            - means: [B, N, 3] Gaussian positions in world space
            - colors: [B, N, 3] RGB colors
            - opacity: [B, N] opacity values
            - scales: [B, N, 3] anisotropic scales
            - rotations: [B, N, 4] quaternion rotations in world space
        """
        B = pose.shape[0]
        device = pose.device

        # Get posed mesh vertices via LBS
        V, J = self.body_model(pose, bone_lengths, center_bone_length, trans, scale)

        # Base positions: vertices corresponding to each Gaussian anchor
        base_positions = V[:, self.vertex_indices]  # [B, N, 3]

        if self.use_local_frame:
            # Compute local tangent frames at each vertex
            normals, tangents, bitangents = self.body_model.compute_tangent_frames(V)

            # Get frames for each Gaussian
            N_gaussians = normals[:, self.vertex_indices]   # [B, N, 3]
            T_gaussians = tangents[:, self.vertex_indices]  # [B, N, 3]
            B_gaussians = bitangents[:, self.vertex_indices]  # [B, N, 3]

            # Transform position offsets from local to world frame
            # offset_world = offset.x * T + offset.y * B + offset.z * N
            offsets = self.position_offsets.unsqueeze(0)  # [1, N, 3]
            offset_world = (
                offsets[..., 0:1] * T_gaussians +
                offsets[..., 1:2] * B_gaussians +
                offsets[..., 2:3] * N_gaussians
            )

            means = base_positions + offset_world

            # Transform rotations from local to world frame
            # Build rotation matrix from tangent frame [T, B, N]
            # This matrix transforms from local to world coordinates
            frame_matrix = torch.stack([T_gaussians, B_gaussians, N_gaussians], dim=-1)  # [B, N, 3, 3]

            # Convert frame to quaternion
            frame_quats = rotation_matrix_to_quaternion(frame_matrix)  # [B, N, 4]

            # Local rotation quaternion
            local_quats = self.get_rotations().unsqueeze(0).expand(B, -1, -1)  # [B, N, 4]

            # Combine: world_rotation = frame_rotation * local_rotation
            rotations = quaternion_multiply(frame_quats, local_quats)
        else:
            # Simple version: add offsets directly in world space
            means = base_positions + self.position_offsets.unsqueeze(0)
            rotations = self.get_rotations().unsqueeze(0).expand(B, -1, -1)

        # Get other parameters (same for all batch items)
        colors = self.get_colors().unsqueeze(0).expand(B, -1, -1)
        opacity = self.get_opacity().unsqueeze(0).expand(B, -1)
        scales = self.get_scales().unsqueeze(0).expand(B, -1, -1)

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
            # Get Gaussian parameters (keep gradients for backprop)
            means = gaussian_params["means"][b]
            quats = gaussian_params["rotations"][b]
            scales = gaussian_params["scales"][b]
            colors = gaussian_params["colors"][b]
            opacities = gaussian_params["opacity"][b]

            # Render using gsplat with gradient support
            # Note: gsplat's rasterization supports autograd
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
                backgrounds=torch.ones(1, 3, device=device),  # (C, channels) for gsplat 1.0
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
    Loss: L1 + SSIM + LPIPS + TV (total variation)

    Following MoReMouse paper:
    - Loss = L1 + λ_SSIM * (1 - SSIM) + λ_LPIPS * LPIPS + λ_TV * TV
    - λ_SSIM = 0.2, λ_LPIPS = 0.1 (from paper)
    - TV loss for spatial smoothness of Gaussian parameters
    - Surface distance constraint to keep Gaussians on mesh

    Reference: MoReMouse paper - 800 frames, 400k iterations
    """

    def __init__(
        self,
        avatar: GaussianAvatar,
        lr: float = 1e-3,
        lambda_ssim: float = 0.2,
        lambda_lpips: float = 0.1,
        lambda_tv: float = 0.01,
        lambda_surface: float = 0.1,
        device: torch.device = None,
    ):
        self.avatar = avatar
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.avatar.to(self.device)

        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_tv = lambda_tv
        self.lambda_surface = lambda_surface

        # Optimizer with per-parameter learning rates
        param_groups = [
            {'params': [avatar.position_offsets], 'lr': lr * 0.1},  # Slower for positions
            {'params': [avatar.colors_raw], 'lr': lr},
            {'params': [avatar.opacity_raw], 'lr': lr * 0.5},
            {'params': [avatar.log_scales], 'lr': lr * 0.5},
            {'params': [avatar.quaternions], 'lr': lr * 0.1},  # Slower for rotations
        ]
        self.optimizer = torch.optim.Adam(param_groups)

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
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute SSIM loss with optional mask.

        Args:
            pred: [B, H, W, 3] predicted images
            target: [B, H, W, 3] target images
            window_size: SSIM window size
            mask: [B, H, W, 1] optional mask (1=foreground, 0=background)
        """
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

        # Apply mask if provided
        if mask is not None:
            # mask: [B, H, W, 1] -> [B, 1, H, W] for broadcasting with [B, C, H, W]
            mask_bchw = mask.permute(0, 3, 1, 2)
            ssim_loss = 1 - ssim  # Convert to loss
            masked_loss = (ssim_loss * mask_bchw).sum() / (mask_bchw.sum() + 1e-6)
            return masked_loss
        else:
            return 1 - ssim.mean()

    def compute_tv_loss(self) -> torch.Tensor:
        """
        Compute Total Variation loss for Gaussian parameters.

        Encourages spatial smoothness in the parameter space.
        For UV-based Gaussians, this penalizes differences between
        neighboring Gaussians in the UV space.
        """
        # TV loss on position offsets
        offsets = self.avatar.position_offsets  # [N, 3]

        # Since Gaussians are ordered by UV/vertex index, neighbors in array
        # are approximately neighbors on the surface
        offset_diff = offsets[1:] - offsets[:-1]
        tv_offset = (offset_diff ** 2).sum()

        # TV loss on colors
        colors = self.avatar.colors_raw  # [N, 3]
        color_diff = colors[1:] - colors[:-1]
        tv_color = (color_diff ** 2).sum()

        # TV loss on scales
        scales = self.avatar.log_scales  # [N, 3]
        scale_diff = scales[1:] - scales[:-1]
        tv_scale = (scale_diff ** 2).sum()

        # Normalize by number of parameters
        N = self.avatar.num_gaussians
        tv_loss = (tv_offset + tv_color * 0.1 + tv_scale * 0.1) / N

        return tv_loss

    def compute_surface_loss(self) -> torch.Tensor:
        """
        Compute surface distance constraint loss.

        Penalizes large position offsets to keep Gaussians
        close to the mesh surface.
        """
        offsets = self.avatar.position_offsets  # [N, 3]

        # L2 norm of offsets - penalize Gaussians that drift too far
        offset_magnitude = (offsets ** 2).sum(dim=-1)

        # Soft constraint: allow small offsets, penalize large ones
        # Using Huber-like loss: linear penalty above threshold
        threshold = 0.005  # 5mm threshold
        surface_loss = torch.where(
            offset_magnitude < threshold ** 2,
            offset_magnitude,  # Quadratic below threshold
            2 * threshold * torch.sqrt(offset_magnitude) - threshold ** 2  # Linear above
        ).mean()

        return surface_loss

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """
        Multiply two quaternions (Hamilton product).

        Args:
            q1, q2: [N, 4] quaternions in (w, x, y, z) format

        Returns:
            [N, 4] product quaternion q1 * q2
        """
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    def train_step(
        self,
        pose: torch.Tensor,
        target_images: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: torch.Tensor,
        width: int,
        height: int,
        trans: torch.Tensor = None,
        scale: torch.Tensor = None,
        world_scale: float = 1.0,
        world_trans: torch.Tensor = None,
        yaw_angle: torch.Tensor = None,
        masks: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Single training step with full loss function.

        Loss = L1 + λ_SSIM * (1-SSIM) + λ_LPIPS * LPIPS + λ_TV * TV + λ_surface * Surface

        Args:
            pose: [B, J*3] pose parameters
            target_images: [B, H, W, 3] target images
            viewmats: [B, 4, 4] view matrices
            Ks: [B, 3, 3] intrinsics
            width: Image width
            height: Image height
            trans: [B, 3] global translation in model space (not used currently)
            scale: [B, 1] global scale (from MAMMAL)
            world_scale: float, scale factor to convert model coords to world coords (e.g., 100 for mm)
            world_trans: [B, 3] global translation in world coords (from center_rotation.npz)
            yaw_angle: [B] yaw rotation angle in radians (Z-axis rotation from center_rotation.npz)
            masks: [B, H, W] segmentation masks (1=foreground, 0=background)

        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()

        # Forward pass (no trans here - we apply world_trans after scaling)
        gaussian_params = self.avatar(
            pose.to(self.device),
            trans=None,  # Don't apply trans in model space
            scale=scale.to(self.device) if scale is not None else None,
        )

        # Apply world_scale to Gaussian means (convert from model coords to world coords)
        if world_scale != 1.0:
            gaussian_params["means"] = gaussian_params["means"] * world_scale
            gaussian_params["scales"] = gaussian_params["scales"] * world_scale

        # Apply base rotation: Body model is Y-up, but camera world is Z-up
        # Rotate -90 degrees around X axis: Y -> Z, Z -> -Y
        # R_x(-90) = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        means = gaussian_params["means"]  # [B, N, 3]
        x, y, z = means[..., 0], means[..., 1], means[..., 2]
        # After rotation: x' = x, y' = z, z' = -y
        gaussian_params["means"] = torch.stack([x, z, -y], dim=-1)

        # Also rotate quaternions for base rotation
        rotations = gaussian_params["rotations"]  # [B, N, 4] (w, x, y, z)
        # Quaternion for -90 deg X rotation: (cos(-45), sin(-45), 0, 0) = (0.7071, -0.7071, 0, 0)
        import math
        base_quat = torch.tensor([math.cos(-math.pi/4), math.sin(-math.pi/4), 0, 0],
                                  dtype=rotations.dtype, device=rotations.device)
        base_quat = base_quat.view(1, 1, 4).expand(rotations.shape[0], rotations.shape[1], 4)
        gaussian_params["rotations"] = self._quaternion_multiply(base_quat, rotations)

        # Apply yaw rotation (Z-axis rotation from center_rotation.npz)
        # Order: scale -> base_rotation -> yaw_rotation -> translate
        if yaw_angle is not None:
            # Build Z-axis rotation matrix from yaw angle [B]
            yaw = yaw_angle.to(self.device)
            B = yaw.shape[0]
            cos_a = torch.cos(yaw)  # [B]
            sin_a = torch.sin(yaw)  # [B]
            zeros = torch.zeros_like(cos_a)
            ones = torch.ones_like(cos_a)

            # R_z = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
            R = torch.stack([
                torch.stack([cos_a, -sin_a, zeros], dim=-1),
                torch.stack([sin_a, cos_a, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ], dim=-2)  # [B, 3, 3]

            # Rotate gaussian means: [B, N, 3] @ [B, 3, 3].T -> [B, N, 3]
            means = gaussian_params["means"]  # [B, N, 3]
            rotated_means = torch.einsum('bni,bji->bnj', means, R)  # R^T @ means
            gaussian_params["means"] = rotated_means

            # Rotate gaussian quaternions: apply R_z to each quaternion
            # For Z-axis rotation, quaternion is [cos(θ/2), 0, 0, sin(θ/2)]
            rotations = gaussian_params["rotations"]  # [B, N, 4]
            N = rotations.shape[1]

            half_angle = yaw / 2.0  # [B]
            R_quat_wxyz = torch.stack([
                torch.cos(half_angle),
                torch.zeros_like(half_angle),
                torch.zeros_like(half_angle),
                torch.sin(half_angle),
            ], dim=-1)  # [B, 4] in (w, x, y, z) format

            for b in range(B):
                q_old = rotations[b]  # [N, 4]
                q_new = self._quaternion_multiply(
                    R_quat_wxyz[b:b+1].expand(N, -1), q_old
                )
                gaussian_params["rotations"][b] = q_new

        # Apply world translation AFTER scaling and rotation
        if world_trans is not None:
            gaussian_params["means"] = gaussian_params["means"] + world_trans.to(self.device).unsqueeze(1)
        rgb, alpha = self.avatar.render(
            gaussian_params,
            viewmats.to(self.device),
            Ks.to(self.device),
            width,
            height,
        )

        target = target_images.to(self.device)

        # Prepare mask for loss computation
        # masks: [B, H, W] -> [B, H, W, 1] for broadcasting with [B, H, W, 3]
        if masks is not None:
            mask = masks.to(self.device).unsqueeze(-1)  # [B, H, W, 1]
            mask_sum = mask.sum() + 1e-6  # Avoid division by zero
        else:
            mask = None

        # L1 loss (primary reconstruction) - masked if available
        if mask is not None:
            l1_loss = (torch.abs(rgb - target) * mask).sum() / mask_sum
        else:
            l1_loss = F.l1_loss(rgb, target)

        # SSIM loss (structural similarity) - masked if available
        ssim_loss = self.compute_ssim(rgb, target, mask=mask)

        # LPIPS loss (perceptual)
        if self.lpips_fn is not None:
            # LPIPS expects BCHW in [-1, 1]
            pred_lpips = rgb.permute(0, 3, 1, 2) * 2 - 1
            target_lpips = target.permute(0, 3, 1, 2) * 2 - 1
            lpips_loss = self.lpips_fn(pred_lpips, target_lpips).mean()
        else:
            lpips_loss = torch.tensor(0.0, device=self.device)

        # TV loss (spatial smoothness)
        tv_loss = self.compute_tv_loss()

        # Surface constraint loss (keep Gaussians on mesh)
        surface_loss = self.compute_surface_loss()

        # Total loss (following paper: L1 + λ_SSIM * SSIM + λ_LPIPS * LPIPS)
        total_loss = (
            l1_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_lpips * lpips_loss +
            self.lambda_tv * tv_loss +
            self.lambda_surface * surface_loss
        )

        # Backward
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.avatar.parameters(), max_norm=1.0)

        self.optimizer.step()

        return {
            "total": total_loss.item(),
            "l1": l1_loss.item(),
            "ssim": ssim_loss.item(),
            "lpips": lpips_loss.item(),
            "tv": tv_loss.item(),
            "surface": surface_loss.item(),
        }

    def train(
        self,
        dataloader,
        num_iterations: int = 400000,
        checkpoint_dir: str = "checkpoints/avatar",
        save_freq: int = 10000,
        log_freq: int = 100,
        vis_freq: int = 1000,
        vis_dir: str = "outputs/avatar_vis",
        resume_from: str = None,
        world_scale: float = None,
        canonical_mode: bool = False,
    ):
        """
        Full training loop for Gaussian Avatar.

        Args:
            dataloader: MAMMAL multi-view dataloader
            num_iterations: Total iterations (paper: 400K)
            checkpoint_dir: Directory for checkpoints
            save_freq: Save checkpoint every N iterations
            log_freq: Log metrics every N iterations
            vis_freq: Save visualization every N iterations
            vis_dir: Directory for visualizations
            resume_from: Path to checkpoint to resume from (optional)
            world_scale: Scale factor to convert model coords to camera coords.
                         If None, auto-compute from camera positions.
            canonical_mode: If True, use MoReMouse canonical space (Ψ_g = 0)
                           - Mesh scaled by 1/180 to fit in unit sphere
                           - No global transform (translation, rotation)
                           - Synthetic cameras on sphere radius 2.22
        """
        from pathlib import Path
        from tqdm import tqdm
        import cv2

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = Path(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Resume from checkpoint if specified
        start_iteration = 0
        if resume_from is not None:
            if Path(resume_from).exists():
                start_iteration = self.load_checkpoint(resume_from)
                print(f"Resuming from iteration {start_iteration}")
            else:
                print(f"Warning: Checkpoint {resume_from} not found, starting from scratch")
        else:
            # Auto-detect latest checkpoint
            # Priority: avatar_iter_*.pt (by iteration number) > avatar_final.pt
            existing_checkpoints = sorted(checkpoint_dir.glob("avatar_iter_*.pt"))
            final_checkpoint = checkpoint_dir / "avatar_final.pt"

            if existing_checkpoints:
                latest = existing_checkpoints[-1]
                print(f"Found existing checkpoint: {latest}")
                start_iteration = self.load_checkpoint(str(latest))
                print(f"Auto-resuming from iteration {start_iteration}")
            elif final_checkpoint.exists():
                print(f"Found final checkpoint: {final_checkpoint}")
                start_iteration = self.load_checkpoint(str(final_checkpoint))
                print(f"Auto-resuming from iteration {start_iteration}")

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_iterations, eta_min=1e-6
        )
        # Step scheduler to correct position if resuming
        for _ in range(start_iteration):
            scheduler.step()

        # Store canonical mode flag
        self._canonical_mode = canonical_mode

        # Set world_scale based on mode
        if canonical_mode:
            # Canonical mode: mesh scaled by 1/180 to fit in unit sphere
            # No global transform applied (Ψ_g = 0)
            CANONICAL_MESH_SCALE = 1.0 / 180.0
            world_scale = CANONICAL_MESH_SCALE
            print(f"[Canonical Mode] Using mesh scale: {world_scale:.6f} (1/180)")
            print(f"[Canonical Mode] Ψ_g = 0 (no global transform)")
        elif world_scale is None:
            # Calibrated via scripts/calibrate_grid_compare.py (2025-12-13)
            # Best result: scale=160 with neg_yaw gives 119.6px error
            # Procrustes analysis suggested scale ~160, which was verified by grid search
            world_scale = 160.0  # Calibrated with Procrustes + grid search
            print(f"Using calibrated world_scale: {world_scale:.2f}")

        self._world_scale = world_scale  # Store for visualization

        # Training loop
        data_iter = iter(dataloader)
        remaining_iterations = num_iterations - start_iteration
        pbar = tqdm(range(remaining_iterations), desc=f"Training Avatar (from {start_iteration})")

        running_loss = {"total": 0, "l1": 0, "ssim": 0, "lpips": 0, "tv": 0, "surface": 0}
        log_count = 0

        for i in pbar:
            iteration = start_iteration + i
            # Get batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Multi-view training: randomly select one view
            images = batch["images"]  # [B, num_cams, H, W, 3]
            viewmats = batch["viewmats"]  # [B, num_cams, 4, 4]
            Ks = batch["Ks"]  # [B, num_cams, 3, 3]
            pose = batch["pose"]  # [B, J*3] or None
            global_transform = batch.get("global_transform")  # Dict with center, angle
            mammal_global = batch.get("mammal_global")  # Dict with R, T, s
            keypoints2d = batch.get("keypoints2d")  # [B, num_cams, 22, 3] or None
            real_camera = batch.get("real_camera")  # Real MAMMAL camera for Option A
            crop_info = batch.get("crop_info")  # Crop transform info for Option A

            B, num_cams = images.shape[:2]
            H, W = images.shape[2:4]

            # Random camera selection for this iteration
            cam_idx = torch.randint(0, num_cams, (B,))

            # Select images and cameras
            target_images = images[torch.arange(B), cam_idx]  # [B, H, W, 3]
            viewmat = viewmats[torch.arange(B), cam_idx]  # [B, 4, 4]
            K = Ks[torch.arange(B), cam_idx]  # [B, 3, 3]

            # Select masks for selected camera (if available)
            target_masks = None
            if "masks" in batch and batch["masks"] is not None:
                masks_all = batch["masks"]  # [B, num_cams, H, W]
                target_masks = masks_all[torch.arange(B), cam_idx]  # [B, H, W]

            # Select GT keypoints for selected camera
            gt_keypoints2d = None
            if keypoints2d is not None:
                gt_keypoints2d = keypoints2d[torch.arange(B), cam_idx]  # [B, 22, 3]

            # Handle missing pose (use random)
            if pose is None or pose[0] is None:
                pose = torch.randn(B, self.avatar.body_model.num_joints * 3) * 0.1

            # Extract world_trans from global_transform (center_rotation.npz)
            # In canonical mode: Ψ_g = 0, so no global transform is applied
            world_trans = None
            yaw_angle = None

            if not canonical_mode:
                # Standard mode: use global transform from center_rotation.npz
                gt_valid = global_transform.get("valid", torch.tensor(False)) if global_transform is not None else torch.tensor(False)
                if isinstance(gt_valid, torch.Tensor):
                    gt_valid = gt_valid.any().item() if gt_valid.dim() > 0 else gt_valid.item()

                if gt_valid and global_transform.get("center") is not None:
                    # Use center from center_rotation.npz as world translation
                    # center_rotation.npz is in floor-aligned coordinates (meters)
                    # Need to: 1) convert m -> cm, 2) add platform offset
                    #
                    # Platform offset (calibrated for markerless_mouse_1_nerf):
                    # - The center_rotation origin is offset from camera world origin
                    # - Calibrated via scripts/calibrate_transforms.py (2025-12-13)
                    # - X offset: +140 cm (horizontal platform position - recalibrated with base rotation)
                    # - Y offset: +0.1 cm (negligible)
                    # - Z offset: +43.9 cm (platform height in world coords)
                    PLATFORM_OFFSET = torch.tensor([140.0, 0.1, 43.9], dtype=torch.float32)

                    center = global_transform["center"].clone()
                    center = center * 100.0  # meters -> cm (uniform conversion)
                    center = center + PLATFORM_OFFSET.to(center.device)
                    world_trans = center.unsqueeze(0) if center.dim() == 1 else center

                # Extract yaw angle from global_transform (center_rotation.npz)
                # This is the mouse's yaw rotation around Z-axis
                # NOTE: Negate yaw angle - calibration shows MAMMAL's angle convention is opposite
                # (calibrated via scripts/calibrate_grid_compare.py: neg_yaw gives 119.6px vs 137.9px)
                if gt_valid and global_transform.get("angle") is not None:
                    angle = global_transform["angle"]
                    if isinstance(angle, torch.Tensor):
                        yaw_angle = -angle.unsqueeze(0) if angle.dim() == 0 else -angle  # [1], negated
                    else:
                        yaw_angle = torch.tensor([-angle], dtype=torch.float32)  # negated

            # Training step with world_scale, world_trans, yaw_angle, and masks
            losses = self.train_step(
                pose, target_images, viewmat, K, W, H,
                trans=None, scale=None, world_scale=self._world_scale,
                world_trans=world_trans, yaw_angle=yaw_angle,
                masks=target_masks
            )
            scheduler.step()

            # Accumulate losses
            for k, v in losses.items():
                running_loss[k] += v
            log_count += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total']:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # Logging
            if (iteration + 1) % log_freq == 0:
                avg_losses = {k: v / log_count for k, v in running_loss.items()}
                print(f"\nIter {iteration + 1}: " +
                      ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))
                running_loss = {k: 0 for k in running_loss}
                log_count = 0

            # Visualization
            if (iteration + 1) % vis_freq == 0:
                # Debug: log pose stats before visualization
                pose_nonzero = (pose[0].abs() > 0.01).sum().item()
                print(f"\n[DEBUG] Vis at iter {iteration+1}: pose nonzero={pose_nonzero}/420")
                self._save_visualization(
                    pose[:1], target_images[:1], viewmat[:1], K[:1], W, H,
                    vis_dir / f"iter_{iteration + 1:06d}.png",
                    world_trans=world_trans[:1] if world_trans is not None else None,
                    yaw_angle=yaw_angle[:1] if yaw_angle is not None else None,
                    mammal_global=mammal_global,
                    gt_keypoints2d=gt_keypoints2d[0] if gt_keypoints2d is not None else None,
                    real_camera=real_camera,  # Option A: MAMMAL camera for real projection
                    crop_info=crop_info,  # Option A: Crop transform
                )

            # Checkpoint
            if (iteration + 1) % save_freq == 0:
                self.save_checkpoint(
                    checkpoint_dir / f"avatar_iter_{iteration + 1:06d}.pt",
                    iteration + 1,
                )

        # Final checkpoint
        self.save_checkpoint(checkpoint_dir / "avatar_final.pt", num_iterations)
        print(f"Training complete! Final checkpoint saved to {checkpoint_dir}")

    def _save_visualization(
        self,
        pose: torch.Tensor,
        target_image: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        width: int,
        height: int,
        save_path,
        world_trans: torch.Tensor = None,
        yaw_angle: torch.Tensor = None,
        mammal_global: dict = None,
        gt_keypoints2d: torch.Tensor = None,  # [22, 3] GT 2D keypoints (x, y, conf)
        real_camera: dict = None,  # Real MAMMAL camera params for Option A projection
        crop_info: dict = None,  # Crop transform info for Option A projection
    ):
        """Save visualization comparing prediction and target with keypoint overlay.

        Visualization layout:
        - Left (Rendered): Model keypoints (multi-color, projected from 3D)
        - Right (GT): GT 2D keypoints (green, from detector)

        Option A (MoReMouse original):
        If real_camera and crop_info are provided, model keypoints in Panel 3
        are projected using real MAMMAL camera and crop transform, showing
        actual pose estimation error without Procrustes alignment.
        """
        import cv2
        from pathlib import Path

        self.avatar.eval()
        world_scale = getattr(self, '_world_scale', 1.0)
        canonical_mode = getattr(self, '_canonical_mode', False)

        with torch.no_grad():
            gaussian_params = self.avatar(
                pose.to(self.device),
                trans=None,  # Don't apply trans in model space
                scale=None,
            )

            # Get joint positions from body model (after forward pass)
            joints_3d = self.avatar.body_model._J_posed  # [B, J, 3]

            # In canonical mode, skip all transforms (mesh is already in canonical space)
            if canonical_mode:
                # Canonical mode: mesh already scaled by 1/180, camera is synthetic
                # Just apply base rotation for Y-up to Z-up conversion
                means = gaussian_params["means"]  # [B, N, 3]
                x, y, z = means[..., 0], means[..., 1], means[..., 2]
                gaussian_params["means"] = torch.stack([x, z, -y], dim=-1)

                # Also rotate joints
                jx, jy, jz = joints_3d[..., 0], joints_3d[..., 1], joints_3d[..., 2]
                joints_3d = torch.stack([jx, jz, -jy], dim=-1)

                # Rotate quaternions for base rotation
                quats = gaussian_params["rotations"]  # [B, N, 4]
                import math
                base_quat = torch.tensor([math.cos(-math.pi/4), math.sin(-math.pi/4), 0, 0],
                                          dtype=quats.dtype, device=quats.device)
                base_quat = base_quat.view(1, 1, 4).expand(quats.shape[0], quats.shape[1], 4)
                gaussian_params["rotations"] = self._quaternion_multiply(base_quat, quats)
            else:
                # Standard mode: apply world_scale and transforms
                # Apply world_scale to joints and gaussians
                if world_scale != 1.0:
                    gaussian_params["means"] = gaussian_params["means"] * world_scale
                    gaussian_params["scales"] = gaussian_params["scales"] * world_scale
                    joints_3d = joints_3d * world_scale

                # Apply base rotation: Body model is Y-up, but camera world is Z-up
                # Rotate -90 degrees around X axis: Y -> Z, Z -> -Y
                means = gaussian_params["means"]  # [B, N, 3]
                x, y, z = means[..., 0], means[..., 1], means[..., 2]
                gaussian_params["means"] = torch.stack([x, z, -y], dim=-1)

                # Also rotate joints
                jx, jy, jz = joints_3d[..., 0], joints_3d[..., 1], joints_3d[..., 2]
                joints_3d = torch.stack([jx, jz, -jy], dim=-1)

                # Rotate quaternions for base rotation
                quats = gaussian_params["rotations"]  # [B, N, 4]
                import math
                base_quat = torch.tensor([math.cos(-math.pi/4), math.sin(-math.pi/4), 0, 0],
                                          dtype=quats.dtype, device=quats.device)
                base_quat = base_quat.view(1, 1, 4).expand(quats.shape[0], quats.shape[1], 4)
                gaussian_params["rotations"] = self._quaternion_multiply(base_quat, quats)

            # Apply yaw rotation and translation only in standard mode
            if not canonical_mode:
                # Apply yaw rotation (Z-axis rotation from center_rotation.npz)
                # Order: scale -> base_rotation -> yaw_rotation -> translate
                if yaw_angle is not None:
                    # Build Z-axis rotation matrix from yaw angle [B]
                    yaw = yaw_angle.to(self.device)
                    B = yaw.shape[0]
                    cos_a = torch.cos(yaw)  # [B]
                    sin_a = torch.sin(yaw)  # [B]
                    zeros = torch.zeros_like(cos_a)
                    ones = torch.ones_like(cos_a)

                    # R_z = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
                    R = torch.stack([
                        torch.stack([cos_a, -sin_a, zeros], dim=-1),
                        torch.stack([sin_a, cos_a, zeros], dim=-1),
                        torch.stack([zeros, zeros, ones], dim=-1),
                    ], dim=-2)  # [B, 3, 3]

                    # Rotate gaussian means: [B, N, 3] @ [B, 3, 3].T -> [B, N, 3]
                    means = gaussian_params["means"]  # [B, N, 3]
                    rotated_means = torch.einsum('bni,bji->bnj', means, R)
                    gaussian_params["means"] = rotated_means

                    # Rotate joints
                    rotated_joints = torch.einsum('bji,bki->bjk', joints_3d, R)
                    joints_3d = rotated_joints

                    # Rotate gaussian quaternions: Z-axis rotation quaternion
                    rotations = gaussian_params["rotations"]  # [B, N, 4]
                    N = rotations.shape[1]

                    half_angle = yaw / 2.0  # [B]
                    R_quat_wxyz = torch.stack([
                        torch.cos(half_angle),
                        torch.zeros_like(half_angle),
                        torch.zeros_like(half_angle),
                        torch.sin(half_angle),
                    ], dim=-1)  # [B, 4] in (w, x, y, z) format

                    for b in range(B):
                        q_old = rotations[b]  # [N, 4]
                        q_new = self._quaternion_multiply(
                            R_quat_wxyz[b:b+1].expand(N, -1), q_old
                        )
                        gaussian_params["rotations"][b] = q_new

                # Apply world translation AFTER rotation and scaling
                if world_trans is not None:
                    world_trans_device = world_trans.to(self.device)
                    gaussian_params["means"] = gaussian_params["means"] + world_trans_device.unsqueeze(1)
                    joints_3d = joints_3d + world_trans_device.unsqueeze(1)

            rgb, alpha = self.avatar.render(
                gaussian_params,
                viewmat.to(self.device),
                K.to(self.device),
                width, height,
            )

            # Project joints to 2D (model keypoints)
            model_joints_2d = self._project_points(joints_3d[0], viewmat[0].to(self.device), K[0].to(self.device))

        pred_img = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
        target_img = (target_image[0].cpu().numpy() * 255).astype(np.uint8)

        # Create debug panel
        debug_img = self._create_debug_panel(
            pose[0], model_joints_2d, world_scale, width, height
        )

        # Draw MODEL keypoints on predicted image (multi-color, 140 joints)
        pred_with_kp = self._draw_keypoints(pred_img.copy(), model_joints_2d, width, height)

        # Draw GT keypoints on target image (circles, 22 joints from detector)
        if gt_keypoints2d is not None:
            gt_kp = gt_keypoints2d.cpu().numpy() if isinstance(gt_keypoints2d, torch.Tensor) else gt_keypoints2d
            target_with_kp = self._draw_gt_keypoints(target_img.copy(), gt_kp, width, height)

            # Create alignment comparison panel: GT (circles) + Model projected (X markers)
            alignment_img = self._draw_gt_keypoints(target_img.copy(), gt_kp, width, height)

            # Option A (MoReMouse original): Use real camera projection + crop transform
            # This shows actual pose estimation error without Procrustes hiding it
            if real_camera is not None and crop_info is not None:
                # Get original 3D joints (before base rotation and transforms)
                # We need to get joints in world coordinates for real camera projection
                joints_3d_world = self.avatar.body_model._J_posed[0].cpu().numpy()  # [140, 3]

                # Project using real camera and apply crop transform
                model_joints_2d_cropped = self._project_joints_3d_to_cropped(
                    joints_3d_world, real_camera, crop_info
                )
                alignment_img = self._draw_model_keypoints_x(
                    alignment_img, model_joints_2d_cropped, gt_kp, width, height
                )
            else:
                # Fallback: use synthetic camera projection (less accurate)
                alignment_img = self._draw_model_keypoints_x(
                    alignment_img, model_joints_2d, gt_kp, width, height
                )
        else:
            # Fallback: draw model keypoints on target (alignment check)
            target_with_kp = self._draw_keypoints(target_img.copy(), model_joints_2d, width, height)
            alignment_img = target_with_kp.copy()

        # Concatenate: Prediction | Target+GT_kp | Alignment(GT+Model) | Debug
        vis = np.concatenate([pred_with_kp, target_with_kp, alignment_img, debug_img], axis=1)
        success = cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        if success:
            print(f"Saved visualization to {save_path}")
        else:
            print(f"[ERROR] Failed to save visualization to {save_path}")

        self.avatar.train()

    def _project_points(
        self,
        points_3d: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
    ) -> np.ndarray:
        """Project 3D points to 2D image coordinates."""
        # points_3d: [N, 3], viewmat: [4, 4], K: [3, 3]
        N = points_3d.shape[0]
        device = points_3d.device

        # Transform to camera space
        R = viewmat[:3, :3]  # [3, 3]
        t = viewmat[:3, 3]   # [3]
        points_cam = points_3d @ R.T + t  # [N, 3]

        # Project to image plane
        points_2d = points_cam @ K.T  # [N, 3]
        points_2d = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-8)  # [N, 2]

        return points_2d.cpu().numpy()

    def _draw_keypoints(
        self,
        img: np.ndarray,
        joints_2d: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Draw keypoints on image with color coding."""
        import cv2

        # Key joint indices for visualization (subset of 140 joints)
        # Head, spine, limbs
        key_joints = {
            0: (255, 0, 0),      # Root - Red
            1: (255, 128, 0),    # Spine 1 - Orange
            5: (255, 255, 0),    # Spine 5 - Yellow
            10: (0, 255, 0),     # Head - Green
            20: (0, 255, 255),   # Front left - Cyan
            40: (0, 128, 255),   # Front right - Light blue
            60: (0, 0, 255),     # Back left - Blue
            80: (128, 0, 255),   # Back right - Purple
            100: (255, 0, 255),  # Tail start - Magenta
            130: (255, 128, 128),  # Tail end - Pink
        }

        for joint_idx, color in key_joints.items():
            if joint_idx < len(joints_2d):
                x, y = int(joints_2d[joint_idx, 0]), int(joints_2d[joint_idx, 1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(img, (x, y), 5, color, -1)
                    cv2.circle(img, (x, y), 6, (0, 0, 0), 1)  # Black outline

        # Draw skeleton lines between some joints
        skeleton = [(0, 1), (1, 5), (5, 10), (0, 100), (100, 130)]
        for i, j in skeleton:
            if i < len(joints_2d) and j < len(joints_2d):
                x1, y1 = int(joints_2d[i, 0]), int(joints_2d[i, 1])
                x2, y2 = int(joints_2d[j, 0]), int(joints_2d[j, 1])
                if (0 <= x1 < width and 0 <= y1 < height and
                    0 <= x2 < width and 0 <= y2 < height):
                    cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

        return img

    def _draw_gt_keypoints(
        self,
        img: np.ndarray,
        keypoints: np.ndarray,  # (22, 3) - x, y, confidence
        width: int,
        height: int,
        conf_threshold: float = 0.3,
    ) -> np.ndarray:
        """Draw GT 2D keypoints on image using MAMMAL color scheme.

        MAMMAL 22 keypoints:
        0: left_ear_tip, 1: right_ear_tip, 2: nose
        3: neck, 4: body_middle, 5: tail_root, 6: tail_middle, 7: tail_end
        8: left_paw, 9: left_paw_end, 10: left_elbow, 11: left_shoulder
        12: right_paw, 13: right_paw_end, 14: right_elbow, 15: right_shoulder
        16: left_foot, 17: left_knee, 18: left_hip
        19: right_foot, 20: right_knee, 21: right_hip
        """
        import cv2

        # MAMMAL color scheme (RGB)
        COLORS = [
            (92, 94, 170),    # 0: purple (ears+nose)
            (187, 97, 166),   # 1: pink (left front leg)
            (109, 192, 91),   # 2: green (right front leg)
            (221, 94, 86),    # 3: red (spine/body)
            (210, 220, 88),   # 4: yellow (left hind leg)
            (98, 201, 211),   # 5: blue (right hind leg)
        ]

        # Joint color index (which color for each of 22 keypoints)
        JOINT_COLOR_INDEX = [
            0, 0, 0,          # 0-2: ears + nose (purple)
            3, 3, 3, 3, 3,    # 3-7: neck, body, tail (red)
            1, 1, 1, 1,       # 8-11: left front leg (pink)
            2, 2, 2, 2,       # 12-15: right front leg (green)
            4, 4, 4,          # 16-18: left hind leg (yellow)
            5, 5, 5           # 19-21: right hind leg (blue)
        ]

        # Keypoint names for legend
        KEYPOINT_NAMES = [
            "L_ear", "R_ear", "nose",
            "neck", "body", "tail_root", "tail_mid", "tail_end",
            "L_paw", "L_paw_end", "L_elbow", "L_shoulder",
            "R_paw", "R_paw_end", "R_elbow", "R_shoulder",
            "L_foot", "L_knee", "L_hip",
            "R_foot", "R_knee", "R_hip"
        ]

        # MAMMAL skeleton bones (21 connections)
        BONES = [
            [0, 2], [1, 2],                      # ears to nose
            [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],  # spine (nose->tail)
            [8, 9], [9, 10], [10, 11], [11, 3],  # left front leg
            [12, 13], [13, 14], [14, 15], [15, 3],  # right front leg
            [16, 17], [17, 18], [18, 5],         # left hind leg
            [19, 20], [20, 21], [21, 5]          # right hind leg
        ]

        # Bone color index
        BONE_COLOR_INDEX = [
            0, 0,             # ears
            3, 3, 3, 3, 3,    # spine
            1, 1, 1, 1,       # left front leg
            2, 2, 2, 2,       # right front leg
            4, 4, 4,          # left hind leg
            5, 5, 5           # right hind leg
        ]

        # Draw skeleton bones first (behind keypoints)
        for bone_idx, (i, j) in enumerate(BONES):
            if i < len(keypoints) and j < len(keypoints):
                if keypoints[i, 2] < conf_threshold or keypoints[j, 2] < conf_threshold:
                    continue
                x1, y1 = int(keypoints[i, 0]), int(keypoints[i, 1])
                x2, y2 = int(keypoints[j, 0]), int(keypoints[j, 1])
                if (0 <= x1 < width and 0 <= y1 < height and
                    0 <= x2 < width and 0 <= y2 < height):
                    color_idx = BONE_COLOR_INDEX[bone_idx]
                    color = COLORS[color_idx]
                    cv2.line(img, (x1, y1), (x2, y2), color, 3)

        # Draw keypoints
        for i in range(len(keypoints)):
            x, y, conf = keypoints[i]
            if conf < conf_threshold:
                continue

            x, y = int(x), int(y)
            if 0 <= x < width and 0 <= y < height:
                color_idx = JOINT_COLOR_INDEX[i] if i < len(JOINT_COLOR_INDEX) else 3
                color = COLORS[color_idx]
                cv2.circle(img, (x, y), 6, color, -1)
                cv2.circle(img, (x, y), 7, (0, 0, 0), 1)  # Black outline

        return img

    def _compute_2d_procrustes(
        self,
        source_pts: np.ndarray,  # [N, 2] points to transform
        target_pts: np.ndarray,  # [N, 2] target points
        weights: np.ndarray = None,  # [N] optional confidence weights
    ) -> tuple:
        """Compute 2D similarity transform (scale + rotation + translation).

        Returns transform parameters such that:
            transformed = scale * R @ source + translation

        Returns:
            scale, rotation_matrix, translation, transformed_source
        """
        if weights is None:
            weights = np.ones(len(source_pts))

        # Filter by valid weights
        valid = weights > 0.3
        if valid.sum() < 3:
            # Not enough points, return identity
            return 1.0, np.eye(2), np.zeros(2), source_pts

        src = source_pts[valid]
        tgt = target_pts[valid]
        w = weights[valid]

        # Weighted centroids
        w_sum = w.sum()
        src_centroid = (src * w[:, None]).sum(axis=0) / w_sum
        tgt_centroid = (tgt * w[:, None]).sum(axis=0) / w_sum

        # Center points
        src_centered = src - src_centroid
        tgt_centered = tgt - tgt_centroid

        # Weighted covariance
        H = (src_centered * w[:, None]).T @ tgt_centered

        # SVD for rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Scale (ratio of dispersions)
        src_var = ((src_centered ** 2) * w[:, None]).sum()
        tgt_var = ((tgt_centered ** 2) * w[:, None]).sum()
        scale = np.sqrt(tgt_var / (src_var + 1e-8))

        # Translation
        translation = tgt_centroid - scale * R @ src_centroid

        # Transform all source points
        transformed = scale * (source_pts @ R.T) + translation

        return scale, R, translation, transformed

    def _project_joints_3d_to_cropped(
        self,
        joints_3d: np.ndarray,  # [140, 3] 3D joints in world space
        real_camera: Dict,  # K, viewmat from MAMMAL calibration
        crop_info: Dict,  # x1, y1, scale from cropping
    ) -> np.ndarray:
        """Project 3D joints to cropped image coordinates (Option A - MoReMouse method).

        This is the correct way to compare Model keypoints with GT keypoints:
        1. Get model 3D joints in world coordinates
        2. Project using REAL MAMMAL camera (K, viewmat)
        3. Apply crop transform to get cropped image coordinates

        Unlike Procrustes alignment, this shows the actual pose estimation error.
        """
        # Get camera parameters
        K = real_camera['K'].cpu().numpy() if torch.is_tensor(real_camera['K']) else real_camera['K']
        viewmat = real_camera['viewmat'].cpu().numpy() if torch.is_tensor(real_camera['viewmat']) else real_camera['viewmat']

        # Extract R and T from viewmat (world-to-camera)
        R = viewmat[:3, :3]  # [3, 3]
        T = viewmat[:3, 3]   # [3]

        # Transform 3D joints to camera space
        # P_cam = R @ P_world + T
        joints_cam = (R @ joints_3d.T).T + T  # [140, 3]

        # Project to image coordinates
        # p = K @ P_cam (then divide by z)
        joints_proj = (K @ joints_cam.T).T  # [140, 3]

        # Perspective divide
        z = joints_proj[:, 2:3]
        z = np.maximum(z, 1e-6)  # Avoid division by zero
        joints_2d_orig = joints_proj[:, :2] / z  # [140, 2] in original image coords

        # Apply crop transform: (p_orig - crop_origin) * scale
        x1 = crop_info['x1'].cpu().item() if torch.is_tensor(crop_info['x1']) else crop_info['x1']
        y1 = crop_info['y1'].cpu().item() if torch.is_tensor(crop_info['y1']) else crop_info['y1']
        scale = crop_info['scale'].cpu().item() if torch.is_tensor(crop_info['scale']) else crop_info['scale']

        joints_2d_cropped = np.zeros_like(joints_2d_orig)
        joints_2d_cropped[:, 0] = (joints_2d_orig[:, 0] - x1) * scale
        joints_2d_cropped[:, 1] = (joints_2d_orig[:, 1] - y1) * scale

        return joints_2d_cropped

    def _draw_model_keypoints_x(
        self,
        img: np.ndarray,
        joints_2d: np.ndarray,  # [140, 2] model joints in cropped image coords
        gt_keypoints: np.ndarray,  # [22, 3] GT keypoints with confidence
        width: int,
        height: int,
    ) -> np.ndarray:
        """Draw model keypoints as X markers.

        Option A (MoReMouse original): joints_2d should already be projected
        via real camera and crop transform by _project_joints_3d_to_cropped().

        Model joints (140) to MAMMAL keypoints (22) mapping from keypoint22_mapper.json.
        """
        import cv2

        # Key joints to visualize using actual model joint indices from keypoint22_mapper.json
        # Colors match MAMMAL color scheme
        key_joints = {
            64: (221, 94, 86),    # Joint 64 -> neck - Red (spine)
            48: (221, 94, 86),    # Joint 48 -> tail_root - Red (spine)
            54: (221, 94, 86),    # Joint 54 -> tail_middle - Red (spine)
            61: (221, 94, 86),    # Joint 61 -> tail_end - Red (spine)
            70: (187, 97, 166),   # Joint 70 -> left_shoulder - Pink (left front)
            73: (187, 97, 166),   # Joint 73 -> left_elbow - Pink (left front)
            79: (187, 97, 166),   # Joint 79 -> left_paw - Pink (left front)
            95: (109, 192, 91),   # Joint 95 -> right_shoulder - Green (right front)
            98: (109, 192, 91),   # Joint 98 -> right_elbow - Green (right front)
            104: (109, 192, 91),  # Joint 104 -> right_paw - Green (right front)
            4: (210, 220, 88),    # Joint 4 -> left_hip - Yellow (left hind)
            5: (210, 220, 88),    # Joint 5 -> left_knee - Yellow (left hind)
            15: (210, 220, 88),   # Joint 15 -> left_foot - Yellow (left hind)
            27: (98, 201, 211),   # Joint 27 -> right_hip - Cyan (right hind)
            28: (98, 201, 211),   # Joint 28 -> right_knee - Cyan (right hind)
            38: (98, 201, 211),   # Joint 38 -> right_foot - Cyan (right hind)
        }

        marker_size = 6

        for joint_idx, color in key_joints.items():
            if joint_idx < len(joints_2d):
                x, y = int(joints_2d[joint_idx, 0]), int(joints_2d[joint_idx, 1])
                if 0 <= x < width and 0 <= y < height:
                    # Draw X marker
                    cv2.line(img, (x - marker_size, y - marker_size),
                             (x + marker_size, y + marker_size), color, 2)
                    cv2.line(img, (x - marker_size, y + marker_size),
                             (x + marker_size, y - marker_size), color, 2)
                    # Black outline for visibility
                    cv2.line(img, (x - marker_size - 1, y - marker_size - 1),
                             (x + marker_size + 1, y + marker_size + 1), (0, 0, 0), 1)
                    cv2.line(img, (x - marker_size - 1, y + marker_size + 1),
                             (x + marker_size + 1, y - marker_size - 1), (0, 0, 0), 1)

        return img

    def _create_debug_panel(
        self,
        pose: torch.Tensor,
        joints_2d: np.ndarray,
        world_scale: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Create debug info panel with MAMMAL legend."""
        import cv2

        # Create white panel
        panel = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Pose statistics
        pose_np = pose.cpu().numpy()
        pose_mean = np.abs(pose_np).mean()
        pose_max = np.abs(pose_np).max()
        pose_nonzero = (np.abs(pose_np) > 0.01).sum()

        # Joint projection statistics
        joints_in_frame = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < width) &
                          (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < height)).sum()
        joints_center = joints_2d.mean(axis=0)

        # Draw text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 0, 0)
        y_offset = 30
        line_height = 22

        texts = [
            f"=== Debug Info ===",
            f"world_scale: {world_scale:.2f}",
            f"",
            f"=== Pose Stats ===",
            f"pose mean: {pose_mean:.4f}",
            f"pose max: {pose_max:.4f}",
            f"pose nonzero: {pose_nonzero}/420",
            f"",
            f"=== Projection ===",
            f"joints in frame: {joints_in_frame}/140",
            f"joints center: ({joints_center[0]:.0f}, {joints_center[1]:.0f})",
            f"expected: ({width//2}, {height//2})",
        ]

        for i, text in enumerate(texts):
            cv2.putText(panel, text, (10, y_offset + i * line_height),
                       font, font_scale, color, 1, cv2.LINE_AA)

        # MAMMAL Color Legend with actual color indicators
        legend_y = y_offset + len(texts) * line_height + 20

        # MAMMAL colors (RGB for display)
        legend_items = [
            ("=== GT Keypoints (MAMMAL 22) ===", None),
            ("Ears + Nose", (92, 94, 170)),      # purple
            ("Spine/Body/Tail", (221, 94, 86)),   # red
            ("Left Front Leg", (187, 97, 166)),   # pink
            ("Right Front Leg", (109, 192, 91)),  # green
            ("Left Hind Leg", (210, 220, 88)),    # yellow
            ("Right Hind Leg", (98, 201, 211)),   # blue
        ]

        for i, (text, legend_color) in enumerate(legend_items):
            y_pos = legend_y + i * line_height
            if legend_color is not None:
                # Draw color circle
                cv2.circle(panel, (20, y_pos - 5), 8, legend_color, -1)
                cv2.circle(panel, (20, y_pos - 5), 9, (0, 0, 0), 1)
                cv2.putText(panel, text, (35, y_pos),
                           font, font_scale, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, text, (10, y_pos),
                           font, font_scale, color, 1, cv2.LINE_AA)

        return panel

    def save_checkpoint(self, path, iteration: int, keep_last: int = 3):
        """Save avatar checkpoint and keep only the last N checkpoints.

        Args:
            path: Path to save checkpoint
            iteration: Current iteration number
            keep_last: Number of recent checkpoints to keep (default: 3)
        """
        from pathlib import Path
        checkpoint_dir = Path(path).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "iteration": iteration,
            "avatar_state_dict": self.avatar.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": {
                "num_vertices": self.avatar.num_vertices,
                "num_gaussians": self.avatar.num_gaussians,
                "num_anchor_points": self.avatar.num_anchor_points,
                "num_gaussians_per_vertex": self.avatar.num_gaussians_per_vertex,
                "use_uv": self.avatar.use_uv,
                "use_local_frame": self.avatar.use_local_frame,
                "lambda_ssim": self.lambda_ssim,
                "lambda_lpips": self.lambda_lpips,
                "lambda_tv": self.lambda_tv,
                "lambda_surface": self.lambda_surface,
            }
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Clean up old checkpoints (keep only last N iteration checkpoints)
        # Exclude 'avatar_final.pt' from deletion
        iter_checkpoints = sorted(
            checkpoint_dir.glob("avatar_iter_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])  # Sort by iteration number
        )

        if len(iter_checkpoints) > keep_last:
            for old_ckpt in iter_checkpoints[:-keep_last]:
                old_ckpt.unlink()
                print(f"Removed old checkpoint: {old_ckpt.name}")

    def load_checkpoint(self, path):
        """Load avatar checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.avatar.load_state_dict(checkpoint["avatar_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from {path} (iteration {checkpoint['iteration']})")
        return checkpoint["iteration"]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        body_model,
        device: torch.device = None,
    ):
        """Create trainer from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        avatar = GaussianAvatar(
            body_model=body_model,
            num_gaussians_per_vertex=config.get("num_gaussians_per_vertex", 1),
            use_local_frame=config.get("use_local_frame", True),
        )
        avatar.load_state_dict(checkpoint["avatar_state_dict"])

        trainer = cls(
            avatar=avatar,
            lambda_ssim=config.get("lambda_ssim", 0.2),
            lambda_lpips=config.get("lambda_lpips", 0.1),
            lambda_tv=config.get("lambda_tv", 0.01),
            lambda_surface=config.get("lambda_surface", 0.1),
            device=device,
        )

        return trainer, checkpoint["iteration"]
