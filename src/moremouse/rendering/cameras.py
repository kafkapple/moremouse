"""Camera sampling utilities."""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def sample_sphere_cameras(num_views: int, radius: float, seed: int) -> FloatArray:
    """Sample camera centers uniformly on a sphere.

    Parameters
    ----------
    num_views:
        Number of camera centers to sample.
    radius:
        Sphere radius.
    seed:
        Random seed.

    Returns
    -------
    FloatArray
        Camera centers with shape ``(num_views, 3)``.
    """
    if num_views <= 0:
        raise ValueError("num_views must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(num_views, 3))
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("Invalid sampled camera centers")
    return centers / norms * radius

