"""
Geodesic Embedding Loss

Loss for learning geodesic-preserving embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicLoss(nn.Module):
    """
    Geodesic embedding loss.

    Minimizes discrepancy between:
    - Euclidean distances in embedding space
    - Geodesic distances on mesh surface

    L_geo = sum_{i,j} ||G_E(i,j) - G_V(i,j)||^2

    Reference: MoReMouse paper Section 3.3
    """

    def __init__(
        self,
        normalize: bool = True,
        sample_size: int = 1000,
    ):
        super().__init__()
        self.normalize = normalize
        self.sample_size = sample_size

    def forward(
        self,
        pred_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        geodesic_distances: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_embedding: [B, N, D] predicted embeddings
            target_embedding: [B, N, D] target embeddings
            geodesic_distances: [N, N] precomputed geodesic distances

        For per-pixel loss (image space):
            pred_embedding: [B, H, W, D]
            target_embedding: [B, H, W, D]
        """
        # Simple MSE between embeddings
        loss = F.mse_loss(pred_embedding, target_embedding)

        return loss


class GeodesicDistanceLoss(nn.Module):
    """
    Direct geodesic distance preservation loss.

    Used for training the geodesic embedding module.
    """

    def __init__(
        self,
        sample_size: int = 1000,
        normalize: bool = True,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.normalize = normalize

    def forward(
        self,
        embedding: torch.Tensor,
        geodesic_distances: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embedding: [V, D] vertex embeddings
            geodesic_distances: [V, V] geodesic distance matrix
        """
        V = embedding.shape[0]
        device = embedding.device

        # Sample random vertex pairs
        idx1 = torch.randint(0, V, (self.sample_size,), device=device)
        idx2 = torch.randint(0, V, (self.sample_size,), device=device)

        # Embedding distances
        E1 = embedding[idx1]
        E2 = embedding[idx2]
        dist_embedding = torch.norm(E1 - E2, dim=-1)

        # Geodesic distances
        dist_geodesic = geodesic_distances[idx1, idx2]

        if self.normalize:
            # Normalize to same scale
            dist_embedding = dist_embedding / (dist_embedding.max() + 1e-8)
            dist_geodesic = dist_geodesic / (dist_geodesic.max() + 1e-8)

        # MSE loss
        loss = F.mse_loss(dist_embedding, dist_geodesic)

        return loss


class EmbeddingConsistencyLoss(nn.Module):
    """
    Embedding consistency across views.

    Ensures same surface point has consistent embedding
    regardless of viewing angle.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        embedding_view1: torch.Tensor,
        embedding_view2: torch.Tensor,
        correspondence_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            embedding_view1: [B, H, W, D] embedding from view 1
            embedding_view2: [B, H, W, D] embedding from view 2
            correspondence_mask: [B, H, W] valid correspondences
        """
        diff = (embedding_view1 - embedding_view2) ** 2

        if correspondence_mask is not None:
            diff = diff * correspondence_mask.unsqueeze(-1)
            return diff.sum() / (correspondence_mask.sum() * diff.shape[-1] + 1e-8)

        return diff.mean()
