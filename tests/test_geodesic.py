import numpy as np
import pytest

from moremouse.geometry import compute_embedding_distances, validate_embedding


def test_compute_embedding_distances_shape_and_values() -> None:
    """Compute pairwise distances for a simple Euclidean embedding."""
    embedding = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float64)

    distances = compute_embedding_distances(embedding)

    assert distances.shape == (3, 3)
    assert distances[0, 1] == pytest.approx(5.0)
    assert distances[1, 2] == pytest.approx(5.0)


def test_validate_embedding_rejects_nan() -> None:
    """Reject non-finite embedding values before distance computation."""
    embedding = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="NaN"):
        validate_embedding(embedding)
