"""
Triplane Representation Module

Implements triplane-based 3D representation used in MoReMouse.
Three orthogonal feature planes (XY, XZ, YZ) encode 3D information.

Reference: MoReMouse paper - 64x64 resolution, 512 channels per plane
Uses memory-efficient Flash Attention with internal lower resolution.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TriplaneGenerator(nn.Module):
    """
    Generates triplane features from image features via transformer decoder.

    Memory optimization: Uses smaller internal resolution (e.g., 16x16) for
    attention operations, then upsamples to target resolution (128x128).
    This reduces attention memory from O((64*64*3)^2) to O((16*16*3)^2).

    Paper spec (Table A3):
    - Triplane: 64x64 internal, 128x128 output
    - Backbone: 512 channels, 12 layers, 16 heads, head_dim=64
    - Upsampler: 512 -> 80 channels, output 3×80×128×128

    Args:
        image_feature_dim: Input image feature dimension (e.g., 768 for DINOv2)
        triplane_resolution: Output resolution of each plane (default: 128)
        internal_resolution: Internal resolution for attention (default: 64)
        triplane_channels: Internal channels (default: 512)
        output_channels: Output channels after upsampler (default: 80)
        num_heads: Number of attention heads (default: 16)
        head_dim: Attention head dimension (default: 64)
        num_layers: Number of transformer layers (default: 12)
        mlp_dim: MLP hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        image_feature_dim: int = 768,
        triplane_resolution: int = 128,   # Paper: 128x128 output
        internal_resolution: int = 64,    # Paper: 64x64 for attention
        triplane_channels: int = 512,     # Internal channels
        output_channels: int = 80,        # Paper: output 80 channels
        num_heads: int = 16,
        head_dim: int = 64,               # Paper: Attention head dimension = 64
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.output_resolution = triplane_resolution
        self.internal_resolution = internal_resolution
        self.channels = triplane_channels
        self.output_channels = output_channels
        self.num_planes = 3

        # Learnable triplane queries at INTERNAL resolution (memory efficient)
        # 64x64x3 = 12,288 tokens (paper spec)
        num_queries = 3 * internal_resolution * internal_resolution
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, triplane_channels) * 0.02
        )

        # Input projection (image features to transformer dim)
        self.input_proj = nn.Linear(image_feature_dim, triplane_channels)

        # Transformer decoder layers with Flash Attention
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=triplane_channels,
                nhead=num_heads,
                head_dim=head_dim,  # Paper: 64
                dim_feedforward=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(triplane_channels)

        # Triplane Upsampler: internal_resolution -> output_resolution
        # Paper: 512 channels -> 80 channels, 64x64 -> 128x128
        self.upsampler = TriplaneUpsampler(
            in_channels=triplane_channels,
            out_channels=output_channels,
            in_resolution=internal_resolution,
            out_resolution=triplane_resolution,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate triplane from image features.

        Args:
            image_features: [B, N, D] image patch features
            image_pos: Optional positional encoding for image features

        Returns:
            [B, 3, C, H, W] triplane features at output_resolution
        """
        B = image_features.shape[0]

        # Project image features
        memory = self.input_proj(image_features)  # [B, N, C]

        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # [B, 3*h*w, C] (h=internal_res)

        # Transformer decoder
        output = queries
        for layer in self.layers:
            output = layer(output, memory)

        output = self.norm(output)

        # Reshape to triplane format at internal resolution
        # [B, 3*h*w, C] -> [B, 3, C, h, w]
        h = w = self.internal_resolution
        triplane = rearrange(
            output,
            'b (n h w) c -> b n c h w',
            n=3, h=h, w=w
        )

        # Upsample to output resolution
        triplane = self.upsampler(triplane)  # [B, 3, C, H, W]

        return triplane


class TriplaneUpsampler(nn.Module):
    """
    Upsamples triplane from internal resolution to output resolution.

    Uses conv-based upsampling for smooth interpolation.
    Reference: MoReMouse paper Table A3 - Triplane Upsampler
    """

    def __init__(
        self,
        in_channels: int = 512,
        out_channels: int = 512,
        in_resolution: int = 16,
        out_resolution: int = 64,
    ):
        super().__init__()

        self.in_resolution = in_resolution
        self.out_resolution = out_resolution

        # Calculate upsampling factor
        scale_factor = out_resolution // in_resolution  # e.g., 64/16 = 4

        # Upsampling layers
        layers = []
        current_res = in_resolution
        current_ch = in_channels

        while current_res < out_resolution:
            # Double resolution each step
            layers.extend([
                nn.ConvTranspose2d(
                    current_ch, current_ch,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.GroupNorm(8, current_ch),
                nn.GELU(),
            ])
            current_res *= 2

        # Final projection
        layers.append(
            nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1)
        )

        self.upsample = nn.Sequential(*layers)

    def forward(self, triplane: torch.Tensor) -> torch.Tensor:
        """
        Args:
            triplane: [B, 3, C, h, w] low-res triplane

        Returns:
            [B, 3, C, H, W] high-res triplane
        """
        B, N, C, h, w = triplane.shape

        # Process each plane
        triplane = rearrange(triplane, 'b n c h w -> (b n) c h w')
        triplane = self.upsample(triplane)
        triplane = rearrange(triplane, '(b n) c h w -> b n c h w', b=B, n=N)

        return triplane


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with Flash Attention.

    Uses F.scaled_dot_product_attention for memory-efficient attention
    (O(n) memory instead of O(n^2) with Flash Attention backend).

    Paper spec: head_dim=64, nhead=16 (separate from d_model=512)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        head_dim: int = 64,  # Paper: Attention head dimension = 64
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = head_dim  # Paper specifies 64, independent of d_model
        self.inner_dim = nhead * head_dim  # 16 * 64 = 1024

        # Self-attention projections (project to inner_dim, not d_model)
        self.q_proj_self = nn.Linear(d_model, self.inner_dim)
        self.k_proj_self = nn.Linear(d_model, self.inner_dim)
        self.v_proj_self = nn.Linear(d_model, self.inner_dim)
        self.out_proj_self = nn.Linear(self.inner_dim, d_model)

        # Cross-attention projections
        self.q_proj_cross = nn.Linear(d_model, self.inner_dim)
        self.k_proj_cross = nn.Linear(d_model, self.inner_dim)
        self.v_proj_cross = nn.Linear(d_model, self.inner_dim)
        self.out_proj_cross = nn.Linear(self.inner_dim, d_model)

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out_proj: nn.Linear,
    ) -> torch.Tensor:
        """
        Memory-efficient attention using scaled_dot_product_attention.

        Args:
            q, k, v: [B, seq_len, inner_dim] (after projection)
            out_proj: Output projection layer

        Returns:
            [B, seq_len, d_model]
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        # Reshape for multi-head attention: [B, heads, seq, head_dim]
        q = q.view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Flash Attention (memory-efficient, uses O(n) memory)
        dropout_p = self.attn_dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )

        # Reshape back: [B, heads, T, head_dim] -> [B, T, inner_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.inner_dim)

        return out_proj(attn_output)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [B, T, D] target (triplane queries)
            memory: [B, S, D] source (image features)

        Returns:
            [B, T, D] updated queries
        """
        # Self-attention with Flash Attention
        q = self.q_proj_self(tgt)
        k = self.k_proj_self(tgt)
        v = self.v_proj_self(tgt)
        tgt2 = self._flash_attention(q, k, v, self.out_proj_self)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention with Flash Attention
        q = self.q_proj_cross(tgt)
        k = self.k_proj_cross(memory)
        v = self.v_proj_cross(memory)
        tgt2 = self._flash_attention(q, k, v, self.out_proj_cross)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt = tgt + self.ffn(tgt)
        tgt = self.norm3(tgt)

        return tgt


class TriplaneDecoder(nn.Module):
    """
    Decodes 3D point features from triplane representation.

    For each 3D point, samples features from three planes and
    aggregates them through an MLP decoder.

    Paper spec (Table A3):
    - Input: 80 channels per plane (from Triplane Upsampler)
    - MultiHeadMLP: 64 neurons, 10 shared hidden layers

    Args:
        triplane_channels: Channels per plane (default: 80)
        hidden_dim: MLP hidden dimension (default: 64)
        num_hidden_layers: Number of hidden layers (default: 10)
        output_rgb: Output RGB channels
        output_density: Output density
        output_embedding: Output geodesic embedding
    """

    def __init__(
        self,
        triplane_channels: int = 80,   # Paper: Triplane Upsampler output 80
        hidden_dim: int = 64,          # Paper: Neurons 64
        num_hidden_layers: int = 10,   # Paper: Shared hidden layers 10
        output_rgb: int = 3,
        output_density: int = 1,
        output_embedding: int = 3,
    ):
        super().__init__()

        self.triplane_channels = triplane_channels

        # Input: concatenated features from 3 planes
        input_dim = triplane_channels * 3

        # Shared MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.shared_mlp = nn.Sequential(*layers)

        # Output heads
        self.rgb_head = nn.Linear(hidden_dim, output_rgb)
        self.density_head = nn.Linear(hidden_dim, output_density)
        self.embedding_head = nn.Linear(hidden_dim, output_embedding)

    def sample_triplane(
        self,
        triplane: torch.Tensor,
        points: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample features from triplane at given 3D points.

        Args:
            triplane: [B, 3, C, H, W] triplane features
            points: [B, N, 3] 3D points in [-1, 1]

        Returns:
            [B, N, 3*C] concatenated features from three planes
        """
        B, N, _ = points.shape

        # Extract coordinates
        x, y, z = points[..., 0], points[..., 1], points[..., 2]

        # Sample from XY plane (use x, y)
        xy_coords = torch.stack([x, y], dim=-1)  # [B, N, 2]
        xy_coords = xy_coords.unsqueeze(1)  # [B, 1, N, 2]
        xy_features = F.grid_sample(
            triplane[:, 0],  # [B, C, H, W]
            xy_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )  # [B, C, 1, N]
        xy_features = xy_features.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # Sample from XZ plane (use x, z)
        xz_coords = torch.stack([x, z], dim=-1).unsqueeze(1)
        xz_features = F.grid_sample(
            triplane[:, 1],
            xz_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze(2).permute(0, 2, 1)

        # Sample from YZ plane (use y, z)
        yz_coords = torch.stack([y, z], dim=-1).unsqueeze(1)
        yz_features = F.grid_sample(
            triplane[:, 2],
            yz_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        ).squeeze(2).permute(0, 2, 1)

        # Concatenate features from all planes
        features = torch.cat([xy_features, xz_features, yz_features], dim=-1)

        return features

    def forward(
        self,
        triplane: torch.Tensor,
        points: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode point properties from triplane.

        Args:
            triplane: [B, 3, C, H, W] triplane features
            points: [B, N, 3] query points in [-1, 1]

        Returns:
            Dictionary with:
            - rgb: [B, N, 3] colors
            - density: [B, N, 1] densities
            - embedding: [B, N, 3] geodesic embeddings
        """
        # Sample triplane features
        features = self.sample_triplane(triplane, points)  # [B, N, 3*C]

        # Shared MLP
        hidden = self.shared_mlp(features)  # [B, N, hidden_dim]

        # Output heads
        rgb = torch.sigmoid(self.rgb_head(hidden))
        density = self.density_head(hidden)
        embedding = self.embedding_head(hidden)

        return {
            "rgb": rgb,
            "density": density,
            "embedding": embedding,
        }


class MultiHeadMLP(nn.Module):
    """
    Multi-head MLP decoder as described in the paper.

    Paper spec (Table A3):
    - Neurons: 64
    - Shared hidden layers: 10
    - Hidden layers for density/feature/deformation heads: 1

    10 shared hidden layers followed by separate heads for
    RGB, density, and embedding outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,          # Paper: Neurons 64
        num_shared_layers: int = 10,   # Paper: Shared hidden layers 10
        output_dims: Dict[str, int] = None,
    ):
        super().__init__()

        if output_dims is None:
            output_dims = {"rgb": 3, "density": 1, "embedding": 3}

        # Shared layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_shared_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        self.shared = nn.Sequential(*layers)

        # Output heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim)
            for name, dim in output_dims.items()
        })

        # Activations
        self.activations = {
            "rgb": torch.sigmoid,
            "density": lambda x: x,  # Raw output, activation in renderer
            "embedding": lambda x: x,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, N, D] input features

        Returns:
            Dictionary of outputs for each head
        """
        hidden = self.shared(x)

        outputs = {}
        for name, head in self.heads.items():
            out = head(hidden)
            if name in self.activations:
                out = self.activations[name](out)
            outputs[name] = out

        return outputs
