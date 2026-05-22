import numpy as np
import pytest

from moremouse.geometry import farthest_point_anchors, geodesic_anchor_distances, geodesic_rgb_embedding


def test_geodesic_anchor_distances_on_square_mesh() -> None:
    """Compute shortest paths on a two-triangle square mesh."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    distances = geodesic_anchor_distances(vertices, faces, np.array([0], dtype=np.int32))

    assert distances.shape == (4, 1)
    assert distances[0, 0] == 0
    assert distances[2, 0] == pytest.approx(np.sqrt(2))


def test_geodesic_rgb_embedding_is_normalized() -> None:
    """Map anchor distances to three bounded correspondence channels."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    anchors = farthest_point_anchors(vertices, 3)
    distances = np.abs(vertices[:, :1] - vertices[anchors, 0][None, :])
    colors = geodesic_rgb_embedding(distances.astype(np.float32))

    assert colors.shape == (4, 3)
    assert float(colors.min()) >= 0.0
    assert float(colors.max()) <= 1.0
