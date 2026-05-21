"""Orthographic mesh preview rendering."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from moremouse.geometry.obj import ObjMesh


def render_mesh_triplet(mesh: ObjMesh, output_path: Path, size: int = 360) -> Path:
    """Render XY, XZ, and YZ orthographic wire previews."""
    if size <= 0:
        raise ValueError("size must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    views = [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]
    canvas = Image.new("RGB", (size * 3, size), (18, 18, 18))
    for index, (label, axes) in enumerate(views):
        image = _render_projection(mesh, axes, size, label)
        canvas.paste(image, (index * size, 0))
    canvas.save(output_path)
    return output_path


def _render_projection(mesh: ObjMesh, axes: tuple[int, int], size: int, label: str) -> Image.Image:
    image = Image.new("RGB", (size, size), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    points = _project(mesh.vertices[:, axes], size)
    face_limit = min(mesh.faces.shape[0], 3500)
    for face in mesh.faces[:face_limit]:
        valid = [int(index) for index in face if index >= 0]
        for start, end in zip(valid, valid[1:] + valid[:1]):
            draw.line((tuple(points[start]), tuple(points[end])), fill=(70, 150, 220), width=1)
    draw.text((8, 8), label, fill=(255, 255, 255))
    return image


def _project(points: np.ndarray, size: int) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)
    normalized = (points - minimum) / span
    padded = normalized * (size - 32) + 16
    padded[:, 1] = size - padded[:, 1]
    return padded.astype(np.int32)
