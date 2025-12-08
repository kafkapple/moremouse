"""
Geodesic Embedding Module

Computes geodesic distances on mesh surface and learns continuous
correspondence embeddings for semantic consistency.

Reference: MoReMouse paper Section 3.3
- Uses heat method for geodesic distance computation
- 3D embedding optimized to preserve geodesic distances
- PCA transformation to HSV color space for visualization
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicEmbedding(nn.Module):
    """
    Geodesic-based correspondence embeddings for mesh surfaces.

    Learns a low-dimensional embedding where Euclidean distances
    approximate geodesic distances on the mesh surface.

    Args:
        num_vertices: Number of mesh vertices
        embedding_dim: Dimension of learned embedding (default: 3 for HSV)
        num_anchors: Number of anchor points for geodesic computation
    """

    def __init__(
        self,
        num_vertices: int,
        embedding_dim: int = 3,
        num_anchors: int = 100,
    ):
        super().__init__()

        self.num_vertices = num_vertices
        self.embedding_dim = embedding_dim
        self.num_anchors = num_anchors

        # Learnable embedding matrix [V, D]
        self.register_parameter(
            "embedding",
            nn.Parameter(torch.randn(num_vertices, embedding_dim) * 0.1)
        )

        # Geodesic distance matrix (computed once, stored as buffer)
        self.register_buffer(
            "geodesic_distances",
            torch.zeros(num_vertices, num_vertices)
        )

        # PCA components for HSV transformation
        self.register_buffer(
            "pca_components",
            torch.eye(embedding_dim)
        )
        self.register_buffer(
            "pca_mean",
            torch.zeros(embedding_dim)
        )

        self._distances_computed = False

    def compute_geodesic_distances(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        use_heat_method: bool = True,
    ) -> torch.Tensor:
        """
        Compute geodesic distances between all vertex pairs.

        Args:
            vertices: [V, 3] mesh vertices
            faces: [F, 3] mesh faces
            use_heat_method: Use heat method (fast) or Dijkstra (exact)

        Returns:
            [V, V] geodesic distance matrix
        """
        V = vertices.shape[0]

        if use_heat_method:
            try:
                import potpourri3d as pp3d

                # Create solver
                solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)

                # Compute distances from each vertex
                distances = np.zeros((V, V), dtype=np.float32)
                for i in range(V):
                    distances[i] = solver.compute_distance(i)

            except ImportError:
                print("Warning: potpourri3d not installed. Using graph-based approximation.")
                distances = self._compute_graph_distances(vertices, faces)
        else:
            distances = self._compute_graph_distances(vertices, faces)

        # Store as buffer
        self.geodesic_distances = torch.from_numpy(distances)
        self._distances_computed = True

        return self.geodesic_distances

    def _compute_graph_distances(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """
        Compute approximate geodesic distances using graph shortest paths.

        This is a fallback when potpourri3d is not available.
        """
        try:
            import networkx as nx
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import dijkstra
        except ImportError:
            raise ImportError("networkx and scipy required for graph-based geodesic computation")

        V = vertices.shape[0]

        # Build adjacency matrix with edge weights
        edges = set()
        for face in faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                if v1 > v2:
                    v1, v2 = v2, v1
                edges.add((v1, v2))

        # Create sparse adjacency matrix
        row, col, data = [], [], []
        for v1, v2 in edges:
            dist = np.linalg.norm(vertices[v1] - vertices[v2])
            row.extend([v1, v2])
            col.extend([v2, v1])
            data.extend([dist, dist])

        adj_matrix = csr_matrix((data, (row, col)), shape=(V, V))

        # Compute shortest paths
        distances = dijkstra(adj_matrix, directed=False)

        return distances.astype(np.float32)

    def compute_pca(self) -> None:
        """
        Compute PCA components for HSV transformation.

        Transforms embedding to HSV-like color space for better visualization.
        """
        E = self.embedding.detach()

        # Center embeddings
        mean = E.mean(dim=0)
        E_centered = E - mean

        # Compute covariance
        cov = torch.mm(E_centered.T, E_centered) / (E.shape[0] - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, idx]

        # Store PCA parameters
        self.pca_components = eigenvectors
        self.pca_mean = mean

    def get_hsv_embeddings(self) -> torch.Tensor:
        """
        Get embeddings transformed to HSV-like color space.

        Returns:
            [V, 3] embeddings as HSV colors (H, S, V)
        """
        E = self.embedding

        # Apply PCA
        E_centered = E - self.pca_mean
        E_pca = torch.mm(E_centered, self.pca_components)

        # Normalize to [0, 1] for each dimension
        E_min = E_pca.min(dim=0)[0]
        E_max = E_pca.max(dim=0)[0]
        E_norm = (E_pca - E_min) / (E_max - E_min + 1e-8)

        # H, S from first two PCA components, V = 1
        hsv = torch.ones_like(E_norm)
        hsv[:, 0] = E_norm[:, 0]  # Hue
        hsv[:, 1] = E_norm[:, 1] if E_norm.shape[1] > 1 else 0.8  # Saturation
        hsv[:, 2] = 1.0  # Value (brightness)

        return hsv

    def hsv_to_rgb(self, hsv: torch.Tensor) -> torch.Tensor:
        """
        Convert HSV to RGB colors.

        Args:
            hsv: [N, 3] HSV colors (H, S, V in [0, 1])

        Returns:
            [N, 3] RGB colors in [0, 1]
        """
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        h6 = h * 6.0
        i = torch.floor(h6).long() % 6
        f = h6 - torch.floor(h6)

        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        rgb = torch.zeros_like(hsv)

        mask0 = (i == 0)
        mask1 = (i == 1)
        mask2 = (i == 2)
        mask3 = (i == 3)
        mask4 = (i == 4)
        mask5 = (i == 5)

        rgb[mask0] = torch.stack([v[mask0], t[mask0], p[mask0]], dim=1)
        rgb[mask1] = torch.stack([q[mask1], v[mask1], p[mask1]], dim=1)
        rgb[mask2] = torch.stack([p[mask2], v[mask2], t[mask2]], dim=1)
        rgb[mask3] = torch.stack([p[mask3], q[mask3], v[mask3]], dim=1)
        rgb[mask4] = torch.stack([t[mask4], p[mask4], v[mask4]], dim=1)
        rgb[mask5] = torch.stack([v[mask5], p[mask5], q[mask5]], dim=1)

        return rgb

    def get_rgb_embeddings(self) -> torch.Tensor:
        """
        Get embeddings as RGB colors.

        Returns:
            [V, 3] RGB colors in [0, 1]
        """
        hsv = self.get_hsv_embeddings()
        return self.hsv_to_rgb(hsv)

    def forward(self, vertex_indices: torch.Tensor = None) -> torch.Tensor:
        """
        Get embeddings for specified vertices.

        Args:
            vertex_indices: Optional indices to select (default: all)

        Returns:
            [N, D] embeddings
        """
        if vertex_indices is None:
            return self.embedding
        return self.embedding[vertex_indices]

    def compute_loss(
        self,
        sample_size: int = 1000,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Compute geodesic embedding loss.

        Minimizes: sum_{i,j} ||G_E(i,j) - G_V(i,j)||^2

        Where:
        - G_E: Euclidean distance in embedding space
        - G_V: Geodesic distance on mesh surface

        Args:
            sample_size: Number of vertex pairs to sample
            normalize: Normalize distances before comparison

        Returns:
            Scalar loss value
        """
        if not self._distances_computed:
            raise RuntimeError("Call compute_geodesic_distances() first")

        V = self.num_vertices
        device = self.embedding.device

        # Sample random vertex pairs
        idx1 = torch.randint(0, V, (sample_size,), device=device)
        idx2 = torch.randint(0, V, (sample_size,), device=device)

        # Embedding distances
        E1 = self.embedding[idx1]
        E2 = self.embedding[idx2]
        dist_embedding = torch.norm(E1 - E2, dim=1)

        # Geodesic distances
        dist_geodesic = self.geodesic_distances[idx1, idx2].to(device)

        if normalize:
            # Normalize to same scale
            dist_embedding = dist_embedding / (dist_embedding.max() + 1e-8)
            dist_geodesic = dist_geodesic / (dist_geodesic.max() + 1e-8)

        # MSE loss
        loss = F.mse_loss(dist_embedding, dist_geodesic)

        return loss


def create_geodesic_embedding(
    vertices: np.ndarray,
    faces: np.ndarray,
    embedding_dim: int = 3,
    num_iterations: int = 1000,
    lr: float = 0.01,
    device: torch.device = None,
) -> GeodesicEmbedding:
    """
    Create and optimize geodesic embedding.

    Args:
        vertices: [V, 3] mesh vertices
        faces: [F, 3] mesh faces
        embedding_dim: Embedding dimension
        num_iterations: Optimization iterations
        lr: Learning rate
        device: Torch device

    Returns:
        Optimized GeodesicEmbedding module
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    V = vertices.shape[0]

    # Create embedding module
    embedding = GeodesicEmbedding(V, embedding_dim).to(device)

    # Compute geodesic distances
    print("Computing geodesic distances...")
    embedding.compute_geodesic_distances(vertices, faces)

    # Optimize embedding
    print("Optimizing embedding...")
    optimizer = torch.optim.Adam(embedding.parameters(), lr=lr)

    for i in range(num_iterations):
        optimizer.zero_grad()
        loss = embedding.compute_loss()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}")

    # Compute PCA for HSV transformation
    embedding.compute_pca()

    return embedding
