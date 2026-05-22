"""Surface Gaussian avatar utilities for dense-view supervision."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from moremouse.geometry.projection import project_vertices


@dataclass(frozen=True)
class GaussianAvatar:
    """Mesh-surface Gaussian proxy used as an AGAM-compatible supervision source."""

    centers: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray


def build_surface_gaussians(vertices: np.ndarray,
                            faces: np.ndarray,
                            vertex_colors: np.ndarray | None = None) -> GaussianAvatar:
    """Initialize one anisotropic Gaussian proxy per mesh triangle."""
    triangles = vertices[faces[:, :3]]
    centers = triangles.mean(axis=1).astype(np.float32)
    edge_a = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
    edge_b = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
    edge_c = np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1)
    scales = np.maximum(np.stack([edge_a, edge_b, edge_c], axis=1).mean(axis=1), 1e-4)
    if vertex_colors is None:
        colors = np.full((faces.shape[0], 3), (0.62, 0.52, 0.55), dtype=np.float32)
    else:
        colors = vertex_colors[faces[:, :3]].mean(axis=1).astype(np.float32)
    opacities = np.ones(faces.shape[0], dtype=np.float32)
    return GaussianAvatar(centers=centers, colors=colors, opacities=opacities, scales=scales.astype(np.float32))


def render_gaussian_avatar(avatar: GaussianAvatar, camera: dict, size: tuple[int, int],
                           radius_scale: float = 1.8) -> Image.Image:
    """Render surface Gaussians with a deterministic depth-sorted disk splat."""
    uv, _ = project_vertices(avatar.centers, camera)
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["T"], dtype=np.float64)
    depth = (rotation @ avatar.centers.astype(np.float64).T + translation[:, None]).T[:, 2]
    order = np.argsort(depth)[::-1]
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    for index in order:
        if not np.isfinite(uv[index]).all() or depth[index] <= 0:
            continue
        radius = max(1.0, float(avatar.scales[index] * radius_scale))
        x_coord, y_coord = float(uv[index, 0]), float(uv[index, 1])
        color = tuple(int(value) for value in np.clip(avatar.colors[index] * 255.0, 0, 255))
        alpha = int(np.clip(avatar.opacities[index], 0.0, 1.0) * 210)
        draw.ellipse((x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius),
                     fill=(*color, alpha))
    return image
