"""Image overlays for markerless mouse diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


KEYPOINT_NAMES = (
    "L_ear",
    "R_ear",
    "nose",
    "neck",
    "body_middle",
    "tail_root",
    "tail_middle",
    "tail_end",
    "L_paw",
    "L_paw_end",
    "L_elbow",
    "L_shoulder",
    "R_paw",
    "R_paw_end",
    "R_elbow",
    "R_shoulder",
    "L_foot",
    "L_knee",
    "L_hip",
    "R_foot",
    "R_knee",
    "R_hip",
)


KEYPOINT_COLORS = (
    (230, 57, 70),
    (255, 159, 28),
    (255, 209, 102),
    (42, 157, 143),
    (46, 196, 182),
    (17, 138, 178),
    (67, 97, 238),
    (114, 9, 183),
    (255, 0, 110),
    (131, 56, 236),
    (58, 134, 255),
    (6, 214, 160),
    (255, 99, 72),
    (255, 183, 3),
    (33, 158, 188),
    (2, 48, 71),
    (173, 181, 189),
    (108, 117, 125),
    (73, 80, 87),
    (144, 190, 109),
    (249, 132, 74),
    (87, 117, 144),
)


def load_rgb(path: Path) -> Image.Image:
    """Load an image as RGB."""
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    return Image.open(path).convert("RGB")


def overlay_mask(
    rgb: Image.Image,
    mask: Image.Image,
    alpha: float = 0.35,
    threshold: int = 0,
) -> Image.Image:
    """Overlay a binary or grayscale segmentation mask on an RGB image."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    max_value = int(np.iinfo(np.uint8).max)
    if not 0 <= threshold <= max_value:
        raise ValueError("threshold must be in [0, 255]")
    base = rgb.convert("RGB")
    mask_l = mask.convert("L").resize(base.size)
    color = Image.new("RGB", base.size, (30, 180, 255))
    blended = Image.blend(base, color, alpha)
    binary = mask_l.point(lambda value: max_value if value > threshold else 0)
    return Image.composite(blended, base, binary)


def draw_keypoints(
    image: Image.Image,
    keypoints: np.ndarray,
    radius: int = 4,
) -> Image.Image:
    """Draw visible 2D keypoints on an image."""
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        raise ValueError("keypoints must have shape [K, >=2]")
    if not np.isfinite(keypoints[:, :2]).all():
        raise ValueError("keypoints contain non-finite coordinates")
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for index, point in enumerate(keypoints):
        if point.shape[0] >= 3 and point[2] <= 0:
            continue
        x_coord = float(point[0])
        y_coord = float(point[1])
        color = KEYPOINT_COLORS[index % len(KEYPOINT_COLORS)]
        draw.ellipse(
            (x_coord - radius, y_coord - radius, x_coord + radius, y_coord + radius),
            fill=color,
            outline=(0, 0, 0),
        )
    return output
