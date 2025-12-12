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

        # RGB colors (before sigmoid) - initialize with small variation
        # This gives initial colors around 0.5 with slight variation
        colors_raw = torch.randn(self.num_gaussians, 3) * 0.1
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
            trans: [B, 3] global translation (from MAMMAL)
            scale: [B, 1] global scale (from MAMMAL)
            world_scale: float, scale factor to convert model coords to world coords (e.g., 100 for mm)

        Returns:
            Dictionary of loss values
        """
        self.optimizer.zero_grad()

        # Forward pass with global transform
        gaussian_params = self.avatar(
            pose.to(self.device),
            trans=trans.to(self.device) if trans is not None else None,
            scale=scale.to(self.device) if scale is not None else None,
        )

        # Apply world_scale to Gaussian means (convert from model coords to camera coords)
        if world_scale != 1.0:
            gaussian_params["means"] = gaussian_params["means"] * world_scale
            gaussian_params["scales"] = gaussian_params["scales"] * world_scale
        rgb, alpha = self.avatar.render(
            gaussian_params,
            viewmats.to(self.device),
            Ks.to(self.device),
            width,
            height,
        )

        target = target_images.to(self.device)

        # L1 loss (primary reconstruction)
        l1_loss = F.l1_loss(rgb, target)

        # SSIM loss (structural similarity)
        ssim_loss = self.compute_ssim(rgb, target)

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
            existing_checkpoints = sorted(checkpoint_dir.glob("avatar_iter_*.pt"))
            if existing_checkpoints:
                latest = existing_checkpoints[-1]
                print(f"Found existing checkpoint: {latest}")
                start_iteration = self.load_checkpoint(str(latest))
                print(f"Auto-resuming from iteration {start_iteration}")

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_iterations, eta_min=1e-6
        )
        # Step scheduler to correct position if resuming
        for _ in range(start_iteration):
            scheduler.step()

        # Auto-compute world_scale from camera positions if not provided
        if world_scale is None:
            # Get camera positions from first batch
            first_batch = next(iter(dataloader))
            viewmats = first_batch["viewmats"]  # [B, num_cams, 4, 4]
            # Camera position = -R^T @ t, where R is viewmat[:3,:3], t is viewmat[:3,3]
            R = viewmats[0, 0, :3, :3]  # First camera rotation
            t = viewmats[0, 0, :3, 3]   # First camera translation
            cam_pos = -R.T @ t
            cam_dist = cam_pos.norm().item()

            # Body model size is ~1 unit (T-pose spans ~1 meter)
            # If camera is at ~200mm, scale = 200 (model meters -> camera mm)
            # If camera is at ~2m, scale = 1 (both in meters)
            if cam_dist > 10:  # Camera distance > 10 units = likely millimeters
                world_scale = cam_dist / 3.0  # Mouse at ~1/3 of camera distance
            else:
                world_scale = 1.0  # Already in matching units

            print(f"Auto-computed world_scale: {world_scale:.2f} (camera distance: {cam_dist:.2f})")

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

            B, num_cams = images.shape[:2]
            H, W = images.shape[2:4]

            # Random camera selection for this iteration
            cam_idx = torch.randint(0, num_cams, (B,))

            # Select images and cameras
            target_images = images[torch.arange(B), cam_idx]  # [B, H, W, 3]
            viewmat = viewmats[torch.arange(B), cam_idx]  # [B, 4, 4]
            K = Ks[torch.arange(B), cam_idx]  # [B, 3, 3]

            # Handle missing pose (use random)
            if pose is None or pose[0] is None:
                pose = torch.randn(B, self.avatar.body_model.num_joints * 3) * 0.1

            # Extract trans and scale from global_transform or mammal_global
            trans = None
            scale = None
            # Check if global_transform is valid (has 'valid' flag = True)
            gt_valid = global_transform.get("valid", torch.tensor(False)) if global_transform is not None else torch.tensor(False)
            if isinstance(gt_valid, torch.Tensor):
                gt_valid = gt_valid.any().item() if gt_valid.dim() > 0 else gt_valid.item()

            if gt_valid and global_transform.get("center") is not None:
                # Use center from center_rotation.npz
                center = global_transform["center"]
                trans = center.unsqueeze(0) if center.dim() == 1 else center
            # Note: mammal_global T is in pixel coords, not directly usable for 3D translation

            # Training step with world_scale
            losses = self.train_step(
                pose, target_images, viewmat, K, W, H,
                trans=trans, scale=scale, world_scale=self._world_scale
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
                self._save_visualization(
                    pose[:1], target_images[:1], viewmat[:1], K[:1], W, H,
                    vis_dir / f"iter_{iteration + 1:06d}.png",
                    trans=trans[:1] if trans is not None else None,
                    scale=scale[:1] if scale is not None else None,
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
        trans: torch.Tensor = None,
        scale: torch.Tensor = None,
    ):
        """Save visualization comparing prediction and target."""
        import cv2
        from pathlib import Path

        self.avatar.eval()
        world_scale = getattr(self, '_world_scale', 1.0)

        with torch.no_grad():
            gaussian_params = self.avatar(
                pose.to(self.device),
                trans=trans.to(self.device) if trans is not None else None,
                scale=scale.to(self.device) if scale is not None else None,
            )

            # Apply world_scale
            if world_scale != 1.0:
                gaussian_params["means"] = gaussian_params["means"] * world_scale
                gaussian_params["scales"] = gaussian_params["scales"] * world_scale

            rgb, alpha = self.avatar.render(
                gaussian_params,
                viewmat.to(self.device),
                K.to(self.device),
                width, height,
            )

        pred_img = (rgb[0].cpu().numpy() * 255).astype(np.uint8)
        target_img = (target_image[0].cpu().numpy() * 255).astype(np.uint8)

        # Concatenate side by side: Prediction | Target
        vis = np.concatenate([pred_img, target_img], axis=1)
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        self.avatar.train()

    def save_checkpoint(self, path, iteration: int):
        """Save avatar checkpoint."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

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
