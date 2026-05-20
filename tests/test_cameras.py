import numpy as np
import pytest

from moremouse.rendering import sample_sphere_cameras

NUM_VIEWS = 64
CAMERA_RADIUS = 2.22
SEED = int("260520")


def test_sample_sphere_cameras_radius() -> None:
    """Sample camera centers on the configured sphere radius."""
    centers = sample_sphere_cameras(num_views=NUM_VIEWS, radius=CAMERA_RADIUS, seed=SEED)

    assert centers.shape == (NUM_VIEWS, 3)
    assert np.linalg.norm(centers, axis=1).mean() == pytest.approx(CAMERA_RADIUS)


def test_sample_sphere_cameras_rejects_invalid_count() -> None:
    """Reject invalid camera counts."""
    with pytest.raises(ValueError, match="num_views"):
        sample_sphere_cameras(num_views=0, radius=CAMERA_RADIUS, seed=1)
