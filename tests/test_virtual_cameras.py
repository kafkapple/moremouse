import numpy as np
import pytest

from moremouse.rendering.virtual_cameras import spherical_virtual_cameras


def test_spherical_virtual_cameras_contract() -> None:
    """Generate dense orbit cameras with OpenCV-style fields."""
    cameras = spherical_virtual_cameras(count=4, radius=2.22, image_size=(64, 64), fov_degrees=29.86)

    assert len(cameras) == 4
    assert cameras[0]["K"].shape == (3, 3)
    assert cameras[0]["mapx"].shape == (64, 64)
    assert np.linalg.det(cameras[0]["R"]) == pytest.approx(1.0)
