"""
MoReMouse Utilities
"""

from .logging import setup_logging, get_logger
from .metrics import compute_metrics
from .visualization import (
    compute_normal_map,
    normal_to_rgb,
    create_rotation_cameras,
    render_novel_views,
    create_rotation_video,
    create_comparison_grid,
    visualize_depth,
    create_inference_visualization,
    VideoRenderer,
)

__all__ = [
    "setup_logging",
    "get_logger",
    "compute_metrics",
    # Visualization
    "compute_normal_map",
    "normal_to_rgb",
    "create_rotation_cameras",
    "render_novel_views",
    "create_rotation_video",
    "create_comparison_grid",
    "visualize_depth",
    "create_inference_visualization",
    "VideoRenderer",
]
