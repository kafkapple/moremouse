"""Camera projection and 2D silhouette helpers."""

import numpy as np
from PIL import Image, ImageDraw


def project_vertices(vertices: np.ndarray, camera: dict) -> tuple[np.ndarray, float]:
    """Project world vertices using MAMMAL camera R/T/K."""
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["T"], dtype=np.float64)
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    camera_points = (rotation @ vertices.astype(np.float64).T + translation[:, None]).T
    positive_depth = camera_points[:, 2] > np.finfo(np.float32).eps
    projected = np.full((vertices.shape[0], 2), np.nan, dtype=np.float64)
    homogeneous = (intrinsic @ camera_points[positive_depth].T).T
    projected[positive_depth] = homogeneous[:, :2] / homogeneous[:, 2:3]
    width = int(camera["mapx"].shape[1])
    height = int(camera["mapx"].shape[0])
    inside = (
        positive_depth
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    return projected, float(inside.mean())


def rasterize_projected_silhouette(
    uv: np.ndarray,
    faces: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Rasterize projected mesh triangles into a binary 2D silhouette."""
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for face in faces:
        vertices = [int(index) for index in face[:3]]
        if not np.isfinite(uv[vertices]).all():
            continue
        points = [tuple(uv[index]) for index in vertices]
        draw.polygon(points, fill=int(np.iinfo(np.uint8).max))
    return np.asarray(canvas) > 0


def binary_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    if union == 0:
        raise ValueError("Degenerate binary IoU union")
    return float(intersection / union)
