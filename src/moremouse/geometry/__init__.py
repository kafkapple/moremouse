"""Geometry utilities."""

from moremouse.geometry.geodesic import compute_embedding_distances, validate_embedding
from moremouse.geometry.dmtet import cube_tetrahedra_grid, marching_tetrahedra
from moremouse.geometry.geodesic_surface import (
    farthest_point_anchors,
    geodesic_anchor_distances,
    geodesic_rgb_embedding,
)

__all__ = [
    "compute_embedding_distances",
    "cube_tetrahedra_grid",
    "farthest_point_anchors",
    "geodesic_anchor_distances",
    "geodesic_rgb_embedding",
    "marching_tetrahedra",
    "validate_embedding",
]
