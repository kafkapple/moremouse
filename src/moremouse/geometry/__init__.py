"""Geometry utilities."""

from moremouse.geometry.geodesic import compute_embedding_distances, validate_embedding
from moremouse.geometry.geodesic_surface import (
    farthest_point_anchors,
    geodesic_anchor_distances,
    geodesic_rgb_embedding,
)

__all__ = [
    "compute_embedding_distances",
    "farthest_point_anchors",
    "geodesic_anchor_distances",
    "geodesic_rgb_embedding",
    "validate_embedding",
]
