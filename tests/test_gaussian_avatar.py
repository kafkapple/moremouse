import numpy as np

from moremouse.rendering.gaussian_avatar import build_surface_gaussians, render_gaussian_avatar


def test_build_and_render_surface_gaussians() -> None:
    """Build triangle Gaussians and render them through a simple camera."""
    vertices = np.array([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    camera = {
        "R": np.eye(3),
        "T": np.zeros(3),
        "K": np.array([[40, 0, 32], [0, 40, 32], [0, 0, 1]], dtype=np.float64),
        "mapx": np.zeros((64, 64), dtype=np.float32),
    }

    avatar = build_surface_gaussians(vertices, faces)
    image = render_gaussian_avatar(avatar, camera, (64, 64))

    assert avatar.centers.shape == (1, 3)
    assert image.size == (64, 64)
    assert np.asarray(image).sum() > 0
