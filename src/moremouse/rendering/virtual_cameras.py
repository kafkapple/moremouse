"""Dense virtual camera generation for synthetic MoReMouse views."""

import numpy as np


def spherical_virtual_cameras(count: int, radius: float, image_size: tuple[int, int],
                              fov_degrees: float, elevation_degrees: float = 15.0) -> list[dict]:
    """Create orbit cameras that look at the origin using the MAMMAL -Y-up convention."""
    if count < 1:
        raise ValueError("camera count must be positive")
    width, height = image_size
    focal = 0.5 * width / np.tan(np.deg2rad(fov_degrees) * 0.5)
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    cameras = []
    elevation = np.deg2rad(elevation_degrees)
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    for index in range(count):
        azimuth = 2.0 * np.pi * index / count
        position = radius * np.array(
            [np.cos(elevation) * np.cos(azimuth), -np.sin(elevation), np.cos(elevation) * np.sin(azimuth)],
            dtype=np.float64,
        )
        rotation = look_at_rotation(position, np.zeros(3, dtype=np.float64), up)
        translation = -rotation @ position
        cameras.append({"R": rotation, "T": translation, "K": intrinsic,
                        "mapx": np.zeros((height, width), dtype=np.float32)})
    return cameras


def look_at_rotation(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Return world-to-camera rotation for an OpenCV-style look-at camera."""
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=0)
