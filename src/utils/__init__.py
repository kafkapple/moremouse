"""
MoReMouse Utilities
"""

from .logging import setup_logging, get_logger
from .metrics import compute_metrics
from .transforms import (
    PLATFORM_OFFSET,
    WORLD_SCALE_DEFAULT,
    CANONICAL_MESH_SCALE,
    CANONICAL_CAMERA_RADIUS,
    CANONICAL_FOV_DEG,
    CANONICAL_IMAGE_SIZE,
    METERS_TO_CM,
    quaternion_multiply,
    yup_to_zup_means,
    yup_to_zup_quaternions,
    apply_coordinate_transform,
    center_rotation_to_world_translation,
    build_z_rotation_matrix,
    apply_yaw_rotation,
    project_points,
)
from .geometry import (
    KEYPOINT22_JOINT_MAP,
    BONES,
    KEYPOINT22_NAMES,
    extract_keypoints22,
    draw_skeleton,
)
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
