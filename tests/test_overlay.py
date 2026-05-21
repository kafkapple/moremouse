"""Tests for visualization overlays."""

import numpy as np
import pytest
from PIL import Image

from moremouse.visualization.overlay import draw_keypoints, overlay_mask


def test_overlay_mask_preserves_size() -> None:
    """Verify mask overlays keep the input image dimensions."""
    rgb = Image.new("RGB", (8, 8), (10, 20, 30))
    mask = Image.new("L", (8, 8), 255)

    output = overlay_mask(rgb, mask)

    assert output.size == rgb.size


def test_draw_keypoints_rejects_non_finite() -> None:
    """Verify keypoint overlays fail fast for invalid coordinates."""
    image = Image.new("RGB", (8, 8), (0, 0, 0))
    keypoints = np.array([[1.0, np.nan, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        draw_keypoints(image, keypoints)
