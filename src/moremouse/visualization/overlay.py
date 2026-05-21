"""Image overlays for markerless mouse diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from jaxtyping import Float


def load_rgb(path: Path) -> Image.Image:
    """Load an image as RGB."""
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return Image.open(path).convert("RGB")


def overlay_mask(rgb: Image.Image, mask: Image.Image, alpha: float = 0.35) -> Image.Image:
    """Overlay a binary or grayscale segmentation mask on an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base = rgb.convert("RGB")
    mask_l = mask.convert("L").resize(base.size)
    color = Image.new("RGB", base.size, (30, 180, 255))
    blended = Image.blend(base, color, alpha)
    return Image.composite(blended, base, mask_l.point(lambda value: 255 if value > 0 else 0))


def draw_keypoints(
    image: Image.Image,
    keypoints: Float[np.ndarray, "keypoints channels"],
    radius: int = 4,
) -> Image.Image:
    """Draw visible 2D keypoints on an image."""
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("keypoints must have shape [K, >=2]")
    if not np.isfinite(keypoints[:, :2]).all():
        raise ValueError("keypoints contain non-finite coordinates")
    output = image.copy()
    draw = ImageDraw.Draw(output)
    colors = [(255, 70, 70), (255, 210, 60), (60, 220, 130), (255, 120, 220)]
    for index, point in enumerate(keypoints):
        if point.shape[0] >= 3 and point[2] <= 0:
            continue
        x_coord = float(point[0])
        y_coord = float(point[1])
        color = colors[index % len(colors)]
        draw.ellipse(
            (x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius),
            fill=color,
            outline=(0, 0, 0),
        )
    return output
