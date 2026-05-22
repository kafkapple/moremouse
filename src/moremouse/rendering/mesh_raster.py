"""Simple mesh attribute rasterization utilities."""

import numpy as np
from PIL import Image, ImageDraw


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute normalized face normals for triangle faces."""
    triangles = vertices[faces[:, :3]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, np.finfo(np.float32).eps)
    return ((normals + 1.0) * 0.5).astype(np.float32)


def rasterize_face_colors(uv: np.ndarray, faces: np.ndarray, face_colors: np.ndarray,
                          size: tuple[int, int]) -> Image.Image:
    """Rasterize per-face RGB colors with PIL polygon filling."""
    image = Image.new("RGB", size, (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for face, color in zip(faces[:, :3], face_colors, strict=True):
        vertices = [int(index) for index in face]
        if not np.isfinite(uv[vertices]).all():
            continue
        points = [tuple(uv[index]) for index in vertices]
        rgb = tuple(int(value) for value in np.clip(color * 255.0, 0, 255))
        draw.polygon(points, fill=rgb)
    return image


def vertex_to_face_colors(vertex_colors: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Average vertex colors onto triangle faces."""
    return vertex_colors[faces[:, :3]].mean(axis=1).astype(np.float32)
