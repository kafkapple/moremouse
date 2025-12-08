"""
Triplane Representation Module

Implements triplane-based 3D representation used in MoReMouse.
Three orthogonal feature planes (XY, XZ, YZ) encode 3D information.

Reference: MoReMouse paper - 64x64 resolution, 512 channels per plane
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TriplaneGenerator(nn.Module):
    """
    Generates triplane features from image features via transformer decoder.

    Uses cross-attention between learnable triplane queries and image features.

    Args:
        image_feature_dim: Input image feature dimension (e.g., 768 for DINOv2)
        triplane_resolution: Resolution of each plane (default: 64)
        triplane_channels: Channels per plane (default: 512)
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        mlp_dim: MLP hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        image_feature_dim: int = 768,
        triplane_resolution: int = 64,
        triplane_channels: int = 512,
        num_heads: int = 16,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.resolution = triplane_resolution
        self.channels = triplane_channels
        self.num_planes = 3

        # Learnable triplane queries
        num_queries = 3 * triplane_resolution * triplane_resolution
        self.queries = nn.Parameter(
            torch.randn(1, num_queries, triplane_channels) * 0.02
        )

        # Input projection (image features to transformer dim)
        self.input_proj = nn.Linear(image_feature_dim, triplane_channels)

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=triplane_channels,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(triplane_channels)

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
            [B, 3, C, H, W] triplane features
        """
        B = image_features.shape[0]

        # Project image features
        memory = self.input_proj(image_features)  # [B, N, C]

        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1)  # [B, 3*H*W, C]

        # Transformer decoder
        output = queries
        for layer in self.layers:
            output = layer(output, memory)

        output = self.norm(output)

        # Reshape to triplane format
        # [B, 3*H*W, C] -> [B, 3, H, W, C] -> [B, 3, C, H, W]
        H = W = self.resolution
        triplane = rearrange(
            output,
            'b (n h w) c -> b n c h w',
            n=3, h=H, w=W
        )

        return triplane


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention and cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

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
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, _ = self.cross_attn(tgt, memory, memory)
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

    Args:
        triplane_channels: Channels per plane
        hidden_dim: MLP hidden dimension
        num_hidden_layers: Number of hidden layers
        output_rgb: Output RGB channels
        output_density: Output density
        output_embedding: Output geodesic embedding
    """

    def __init__(
        self,
        triplane_channels: int = 512,
        hidden_dim: int = 256,
        num_hidden_layers: int = 10,
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

    10 shared hidden layers followed by separate heads for
    RGB, density, and embedding outputs.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_shared_layers: int = 10,
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
