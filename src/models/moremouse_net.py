"""
MoReMouse Network

Main reconstruction network for monocular mouse reconstruction.
Architecture:
1. DINOv2 encoder for image feature extraction
2. Transformer decoder to generate triplane representation
3. NeRF/DMTet renderer for novel view synthesis

Reference: MoReMouse paper (arXiv:2507.04258v2)
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triplane import TriplaneGenerator, TriplaneDecoder, MultiHeadMLP


class DINOv2Encoder(nn.Module):
    """
    DINOv2 image encoder.

    Uses pretrained DINOv2-B/14 model to extract image features.
    Input: 378x378 images
    Output: 768-dim patch features

    Args:
        model_name: DINOv2 model variant
        freeze: Whether to freeze encoder weights
        input_size: Expected input image size
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        input_size: int = 378,
    ):
        super().__init__()

        self.model_name = model_name
        self.input_size = input_size
        self.freeze = freeze

        # Load DINOv2 model
        try:
            self.encoder = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=True,
            )
        except Exception as e:
            # Fallback: use transformers library
            print(f"torch.hub failed: {e}")
            print("Trying transformers library...")
            self._load_from_transformers()

        self.feature_dim = self.encoder.embed_dim

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _load_from_transformers(self):
        """Load DINOv2 from transformers library."""
        try:
            from transformers import Dinov2Model

            model_map = {
                "dinov2_vits14": "facebook/dinov2-small",
                "dinov2_vitb14": "facebook/dinov2-base",
                "dinov2_vitl14": "facebook/dinov2-large",
                "dinov2_vitg14": "facebook/dinov2-giant",
            }

            model_id = model_map.get(self.model_name, "facebook/dinov2-base")
            self.encoder = Dinov2Model.from_pretrained(model_id)
            self._use_transformers = True
        except ImportError:
            raise ImportError(
                "Neither torch.hub nor transformers could load DINOv2. "
                "Install transformers: pip install transformers"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features.

        Args:
            images: [B, 3, H, W] RGB images in [0, 1]

        Returns:
            [B, N, D] patch features where N = (H/14) * (W/14)
        """
        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        # Resize if needed
        if images.shape[-1] != self.input_size:
            images = F.interpolate(
                images,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )

        if hasattr(self, '_use_transformers') and self._use_transformers:
            outputs = self.encoder(images, return_dict=True)
            features = outputs.last_hidden_state[:, 1:]  # Remove CLS token
        else:
            # torch.hub DINOv2
            features = self.encoder.forward_features(images)
            if isinstance(features, dict):
                features = features['x_norm_patchtokens']

        return features


class MoReMouse(nn.Module):
    """
    MoReMouse: Monocular Reconstruction of Laboratory Mouse.

    Main network combining:
    - DINOv2 encoder
    - Triplane generator
    - Multi-head MLP decoder
    - NeRF/DMTet rendering

    Args:
        encoder_config: DINOv2 encoder configuration
        triplane_config: Triplane generator configuration
        decoder_config: MLP decoder configuration
        render_mode: "nerf" or "dmtet"
    """

    def __init__(
        self,
        encoder_config: Dict = None,
        triplane_config: Dict = None,
        decoder_config: Dict = None,
        render_mode: str = "nerf",
    ):
        super().__init__()

        # Default configs
        encoder_config = encoder_config or {}
        triplane_config = triplane_config or {}
        decoder_config = decoder_config or {}

        # Image encoder
        self.encoder = DINOv2Encoder(**encoder_config)

        # Triplane generator
        triplane_config.setdefault("image_feature_dim", self.encoder.feature_dim)
        self.triplane_generator = TriplaneGenerator(**triplane_config)

        # MLP decoder
        triplane_channels = triplane_config.get("triplane_channels", 512)
        decoder_config.setdefault("input_dim", triplane_channels * 3)
        self.decoder = MultiHeadMLP(**decoder_config)

        # Triplane decoder (for sampling)
        self.triplane_decoder = TriplaneDecoder(
            triplane_channels=triplane_channels,
        )

        self.render_mode = render_mode

        # NeRF parameters
        self.num_samples = 128
        self.near = 0.1
        self.far = 4.0
        self.radius = 0.87

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode input images to triplane representation.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            [B, 3, C, H, W] triplane features
        """
        # Extract image features
        features = self.encoder(images)  # [B, N, D]

        # Generate triplane
        triplane = self.triplane_generator(features)  # [B, 3, C, H, W]

        return triplane

    def query_points(
        self,
        triplane: torch.Tensor,
        points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Query properties at 3D points.

        Args:
            triplane: [B, 3, C, H, W] triplane features
            points: [B, N, 3] 3D query points

        Returns:
            Dictionary with rgb, density, embedding
        """
        return self.triplane_decoder(triplane, points)

    def render_rays(
        self,
        triplane: torch.Tensor,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float = None,
        far: float = None,
        num_samples: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render rays using volumetric rendering (NeRF).

        Args:
            triplane: [B, 3, C, H, W] triplane features
            rays_o: [B, N, 3] ray origins
            rays_d: [B, N, 3] ray directions
            near: Near plane
            far: Far plane
            num_samples: Samples per ray

        Returns:
            Dictionary with rgb, depth, alpha, embedding
        """
        near = near or self.near
        far = far or self.far
        num_samples = num_samples or self.num_samples

        B, N, _ = rays_o.shape
        device = rays_o.device

        # Sample points along rays
        t_vals = torch.linspace(0, 1, num_samples, device=device)
        z_vals = near + (far - near) * t_vals  # [S]
        z_vals = z_vals.unsqueeze(0).unsqueeze(0).expand(B, N, -1)  # [B, N, S]

        # Add noise for training
        if self.training:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

        # Compute 3D points
        pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)
        # [B, N, S, 3]

        # Normalize to [-1, 1] for triplane sampling
        pts_normalized = pts / self.radius

        # Query triplane
        pts_flat = pts_normalized.reshape(B, -1, 3)  # [B, N*S, 3]
        outputs = self.triplane_decoder(triplane, pts_flat)

        # Reshape outputs
        rgb = outputs["rgb"].reshape(B, N, num_samples, 3)
        density = outputs["density"].reshape(B, N, num_samples, 1)
        embedding = outputs["embedding"].reshape(B, N, num_samples, 3)

        # Apply density activation (trunc_exp as in paper)
        density = self._trunc_exp(density)

        # Volumetric rendering
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([
            dists,
            torch.full((B, N, 1), 1e10, device=device)
        ], dim=-1)

        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)  # [B, N, S]

        # Transmittance
        T = torch.cumprod(
            torch.cat([
                torch.ones((B, N, 1), device=device),
                1.0 - alpha + 1e-10
            ], dim=-1),
            dim=-1
        )[:, :, :-1]

        weights = alpha * T  # [B, N, S]

        # Composite
        rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=-2)  # [B, N, 3]
        depth = (weights * z_vals).sum(dim=-1)  # [B, N]
        alpha_final = weights.sum(dim=-1)  # [B, N]
        embedding_final = (weights.unsqueeze(-1) * embedding).sum(dim=-2)  # [B, N, 3]

        return {
            "rgb": rgb_final,
            "depth": depth,
            "alpha": alpha_final,
            "embedding": embedding_final,
            "weights": weights,
        }

    @staticmethod
    def _trunc_exp(x: torch.Tensor) -> torch.Tensor:
        """Truncated exponential activation for density."""
        return torch.exp(torch.clamp(x, max=15))

    def render_image(
        self,
        triplane: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Render full image from triplane.

        Args:
            triplane: [B, 3, C, H, W] triplane features
            viewmat: [B, 4, 4] camera view matrix
            K: [B, 3, 3] camera intrinsics
            height: Image height
            width: Image width

        Returns:
            Dictionary with rendered rgb, depth, alpha, embedding images
        """
        B = triplane.shape[0]
        device = triplane.device

        # Generate rays
        rays_o, rays_d = self._generate_rays(viewmat, K, height, width)

        # Flatten rays
        rays_o = rays_o.reshape(B, -1, 3)
        rays_d = rays_d.reshape(B, -1, 3)

        # Render (in chunks to save memory)
        chunk_size = 4096
        num_rays = rays_o.shape[1]

        outputs = {
            "rgb": [],
            "depth": [],
            "alpha": [],
            "embedding": [],
        }

        for i in range(0, num_rays, chunk_size):
            chunk_o = rays_o[:, i:i+chunk_size]
            chunk_d = rays_d[:, i:i+chunk_size]

            chunk_out = self.render_rays(triplane, chunk_o, chunk_d)

            for key in outputs:
                outputs[key].append(chunk_out[key])

        # Concatenate chunks
        for key in outputs:
            outputs[key] = torch.cat(outputs[key], dim=1)

        # Reshape to images
        outputs["rgb"] = outputs["rgb"].reshape(B, height, width, 3)
        outputs["depth"] = outputs["depth"].reshape(B, height, width)
        outputs["alpha"] = outputs["alpha"].reshape(B, height, width)
        outputs["embedding"] = outputs["embedding"].reshape(B, height, width, 3)

        return outputs

    def _generate_rays(
        self,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        height: int,
        width: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate rays for all pixels.

        Args:
            viewmat: [B, 4, 4] world-to-camera transform
            K: [B, 3, 3] intrinsics
            height, width: Image dimensions

        Returns:
            rays_o: [B, H, W, 3] ray origins
            rays_d: [B, H, W, 3] ray directions
        """
        B = viewmat.shape[0]
        device = viewmat.device

        # Pixel coordinates
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing='ij'
        )
        x = x.float()
        y = y.float()

        # Unproject to camera space
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]

        x_cam = (x.unsqueeze(0) - cx.view(B, 1, 1)) / fx.view(B, 1, 1)
        y_cam = (y.unsqueeze(0) - cy.view(B, 1, 1)) / fy.view(B, 1, 1)
        z_cam = torch.ones_like(x_cam)

        dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, H, W, 3]

        # Camera-to-world rotation
        R = viewmat[:, :3, :3]  # [B, 3, 3]
        R_inv = R.transpose(1, 2)

        # Transform directions to world space
        dirs_world = torch.einsum('bij,bhwj->bhwi', R_inv, dirs_cam)
        rays_d = F.normalize(dirs_world, dim=-1)

        # Camera position in world space
        t = viewmat[:, :3, 3]  # [B, 3]
        rays_o = -torch.einsum('bij,bj->bi', R_inv, t)
        rays_o = rays_o.view(B, 1, 1, 3).expand(-1, height, width, -1)

        return rays_o, rays_d

    def forward(
        self,
        images: torch.Tensor,
        viewmats: torch.Tensor = None,
        Ks: torch.Tensor = None,
        render_size: Tuple[int, int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            images: [B, 3, H, W] input images
            viewmats: [B, 4, 4] view matrices for rendering
            Ks: [B, 3, 3] camera intrinsics
            render_size: (height, width) for rendering

        Returns:
            Dictionary with triplane and optionally rendered outputs
        """
        # Encode to triplane
        triplane = self.encode_image(images)

        outputs = {"triplane": triplane}

        # Render if camera parameters provided
        if viewmats is not None and Ks is not None:
            render_size = render_size or (images.shape[2], images.shape[3])
            render_out = self.render_image(
                triplane, viewmats, Ks,
                height=render_size[0],
                width=render_size[1],
            )
            outputs.update(render_out)

        return outputs
