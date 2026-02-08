"""
Avatar Visualization Utilities

Visualization helper functions extracted from GaussianAvatarTrainer.
Pure rendering/drawing functions with no model dependencies.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def project_points_to_2d(
    points_3d: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> np.ndarray:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: [N, 3] 3D points
        viewmat: [4, 4] world-to-camera transform
        K: [3, 3] camera intrinsics

    Returns:
        [N, 2] 2D image coordinates (numpy)
    """
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    points_cam = points_3d @ R.T + t
    points_2d = points_cam @ K.T
    points_2d = points_2d[:, :2] / (points_2d[:, 2:3] + 1e-8)
    return points_2d.cpu().numpy()


def draw_model_keypoints(
    img: np.ndarray,
    joints_2d: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw 140-joint keypoints on image with color coding.

    Args:
        img: [H, W, 3] RGB image (modified in-place)
        joints_2d: [140, 2] projected joint coordinates
        width: Image width
        height: Image height

    Returns:
        Image with keypoints drawn
    """
    import cv2

    key_joints = {
        0: (255, 0, 0),       # Root - Red
        1: (255, 128, 0),     # Spine 1 - Orange
        5: (255, 255, 0),     # Spine 5 - Yellow
        10: (0, 255, 0),      # Head - Green
        20: (0, 255, 255),    # Front left - Cyan
        40: (0, 128, 255),    # Front right - Light blue
        60: (0, 0, 255),      # Back left - Blue
        80: (128, 0, 255),    # Back right - Purple
        100: (255, 0, 255),   # Tail start - Magenta
        130: (255, 128, 128), # Tail end - Pink
    }

    for joint_idx, color in key_joints.items():
        if joint_idx < len(joints_2d):
            x, y = int(joints_2d[joint_idx, 0]), int(joints_2d[joint_idx, 1])
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(img, (x, y), 5, color, -1)
                cv2.circle(img, (x, y), 6, (0, 0, 0), 1)

    skeleton = [(0, 1), (1, 5), (5, 10), (0, 100), (100, 130)]
    for i, j in skeleton:
        if i < len(joints_2d) and j < len(joints_2d):
            x1, y1 = int(joints_2d[i, 0]), int(joints_2d[i, 1])
            x2, y2 = int(joints_2d[j, 0]), int(joints_2d[j, 1])
            if (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height):
                cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

    return img


# MAMMAL 22-keypoint color scheme
_MAMMAL_COLORS = [
    (92, 94, 170),    # 0: purple (ears+nose)
    (187, 97, 166),   # 1: pink (left front leg)
    (109, 192, 91),   # 2: green (right front leg)
    (221, 94, 86),    # 3: red (spine/body)
    (210, 220, 88),   # 4: yellow (left hind leg)
    (98, 201, 211),   # 5: blue (right hind leg)
]

_JOINT_COLOR_INDEX = [
    0, 0, 0,          # 0-2: ears + nose (purple)
    3, 3, 3, 3, 3,    # 3-7: neck, body, tail (red)
    1, 1, 1, 1,       # 8-11: left front leg (pink)
    2, 2, 2, 2,       # 12-15: right front leg (green)
    4, 4, 4,          # 16-18: left hind leg (yellow)
    5, 5, 5,          # 19-21: right hind leg (blue)
]

# MAMMAL skeleton bones (21 connections)
_MAMMAL_BONES = [
    [0, 2], [1, 2],
    [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [11, 3],
    [12, 13], [13, 14], [14, 15], [15, 3],
    [16, 17], [17, 18], [18, 5],
    [19, 20], [20, 21], [21, 5],
]

_BONE_COLOR_INDEX = [
    0, 0,
    3, 3, 3, 3, 3,
    1, 1, 1, 1,
    2, 2, 2, 2,
    4, 4, 4,
    5, 5, 5,
]


def draw_gt_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray,
    width: int,
    height: int,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Draw GT 2D keypoints on image using MAMMAL color scheme.

    Args:
        img: [H, W, 3] RGB image (modified in-place)
        keypoints: [22, 3] keypoints (x, y, confidence)
        width: Image width
        height: Image height
        conf_threshold: Minimum confidence to draw

    Returns:
        Image with GT keypoints drawn
    """
    import cv2

    # Draw skeleton bones first
    for bone_idx, (i, j) in enumerate(_MAMMAL_BONES):
        if i < len(keypoints) and j < len(keypoints):
            if keypoints[i, 2] < conf_threshold or keypoints[j, 2] < conf_threshold:
                continue
            x1, y1 = int(keypoints[i, 0]), int(keypoints[i, 1])
            x2, y2 = int(keypoints[j, 0]), int(keypoints[j, 1])
            if (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height):
                color = _MAMMAL_COLORS[_BONE_COLOR_INDEX[bone_idx]]
                cv2.line(img, (x1, y1), (x2, y2), color, 3)

    # Draw keypoints
    for i in range(len(keypoints)):
        x, y, conf = keypoints[i]
        if conf < conf_threshold:
            continue
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            color_idx = _JOINT_COLOR_INDEX[i] if i < len(_JOINT_COLOR_INDEX) else 3
            cv2.circle(img, (x, y), 6, _MAMMAL_COLORS[color_idx], -1)
            cv2.circle(img, (x, y), 7, (0, 0, 0), 1)

    return img


def compute_2d_procrustes(
    source_pts: np.ndarray,
    target_pts: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D similarity transform (scale + rotation + translation).

    Args:
        source_pts: [N, 2] points to transform
        target_pts: [N, 2] target points
        weights: [N] optional confidence weights

    Returns:
        (scale, rotation_matrix, translation, transformed_source)
    """
    if weights is None:
        weights = np.ones(len(source_pts))

    valid = weights > 0.3
    if valid.sum() < 3:
        return 1.0, np.eye(2), np.zeros(2), source_pts

    src = source_pts[valid]
    tgt = target_pts[valid]
    w = weights[valid]

    w_sum = w.sum()
    src_centroid = (src * w[:, None]).sum(axis=0) / w_sum
    tgt_centroid = (tgt * w[:, None]).sum(axis=0) / w_sum

    src_centered = src - src_centroid
    tgt_centered = tgt - tgt_centroid

    H = (src_centered * w[:, None]).T @ tgt_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    src_var = ((src_centered ** 2) * w[:, None]).sum()
    tgt_var = ((tgt_centered ** 2) * w[:, None]).sum()
    scale = np.sqrt(tgt_var / (src_var + 1e-8))

    translation = tgt_centroid - scale * R @ src_centroid
    transformed = scale * (source_pts @ R.T) + translation

    return scale, R, translation, transformed


def project_joints_3d_to_cropped(
    joints_3d: np.ndarray,
    real_camera: Dict,
    crop_info: Dict,
) -> np.ndarray:
    """Project 3D joints to cropped image coordinates (MoReMouse method).

    Args:
        joints_3d: [140, 3] 3D joints in world space
        real_camera: Dict with 'K' and 'viewmat'
        crop_info: Dict with 'x1', 'y1', 'scale'

    Returns:
        [140, 2] projected coordinates in cropped image space
    """
    K = real_camera['K'].cpu().numpy() if torch.is_tensor(real_camera['K']) else real_camera['K']
    viewmat = real_camera['viewmat'].cpu().numpy() if torch.is_tensor(real_camera['viewmat']) else real_camera['viewmat']

    R = viewmat[:3, :3]
    T = viewmat[:3, 3]

    joints_cam = (R @ joints_3d.T).T + T
    joints_proj = (K @ joints_cam.T).T

    z = np.maximum(joints_proj[:, 2:3], 1e-6)
    joints_2d_orig = joints_proj[:, :2] / z

    x1 = crop_info['x1'].cpu().item() if torch.is_tensor(crop_info['x1']) else crop_info['x1']
    y1 = crop_info['y1'].cpu().item() if torch.is_tensor(crop_info['y1']) else crop_info['y1']
    scale = crop_info['scale'].cpu().item() if torch.is_tensor(crop_info['scale']) else crop_info['scale']

    joints_2d_cropped = np.zeros_like(joints_2d_orig)
    joints_2d_cropped[:, 0] = (joints_2d_orig[:, 0] - x1) * scale
    joints_2d_cropped[:, 1] = (joints_2d_orig[:, 1] - y1) * scale

    return joints_2d_cropped


def draw_model_keypoints_x(
    img: np.ndarray,
    joints_2d: np.ndarray,
    gt_keypoints: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw model keypoints as X markers with MAMMAL color scheme.

    Args:
        img: [H, W, 3] RGB image (modified in-place)
        joints_2d: [140, 2] model joints in cropped image coords
        gt_keypoints: [22, 3] GT keypoints with confidence
        width: Image width
        height: Image height

    Returns:
        Image with X markers drawn
    """
    import cv2

    key_joints = {
        64: (221, 94, 86),    # neck - Red (spine)
        48: (221, 94, 86),    # tail_root - Red (spine)
        54: (221, 94, 86),    # tail_middle - Red (spine)
        61: (221, 94, 86),    # tail_end - Red (spine)
        70: (187, 97, 166),   # left_shoulder - Pink (left front)
        73: (187, 97, 166),   # left_elbow - Pink (left front)
        79: (187, 97, 166),   # left_paw - Pink (left front)
        95: (109, 192, 91),   # right_shoulder - Green (right front)
        98: (109, 192, 91),   # right_elbow - Green (right front)
        104: (109, 192, 91),  # right_paw - Green (right front)
        4: (210, 220, 88),    # left_hip - Yellow (left hind)
        5: (210, 220, 88),    # left_knee - Yellow (left hind)
        15: (210, 220, 88),   # left_foot - Yellow (left hind)
        27: (98, 201, 211),   # right_hip - Cyan (right hind)
        28: (98, 201, 211),   # right_knee - Cyan (right hind)
        38: (98, 201, 211),   # right_foot - Cyan (right hind)
    }

    marker_size = 6
    for joint_idx, color in key_joints.items():
        if joint_idx < len(joints_2d):
            x, y = int(joints_2d[joint_idx, 0]), int(joints_2d[joint_idx, 1])
            if 0 <= x < width and 0 <= y < height:
                cv2.line(img, (x - marker_size, y - marker_size),
                         (x + marker_size, y + marker_size), color, 2)
                cv2.line(img, (x - marker_size, y + marker_size),
                         (x + marker_size, y - marker_size), color, 2)
                cv2.line(img, (x - marker_size - 1, y - marker_size - 1),
                         (x + marker_size + 1, y + marker_size + 1), (0, 0, 0), 1)
                cv2.line(img, (x - marker_size - 1, y + marker_size + 1),
                         (x + marker_size + 1, y - marker_size - 1), (0, 0, 0), 1)

    return img


def create_debug_panel(
    pose: torch.Tensor,
    joints_2d: np.ndarray,
    world_scale: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Create debug info panel with MAMMAL legend.

    Args:
        pose: [J*3] pose parameters
        joints_2d: [140, 2] projected joint coordinates
        world_scale: Current world scale value
        width: Panel width
        height: Panel height

    Returns:
        [H, W, 3] debug panel image
    """
    import cv2

    panel = np.ones((height, width, 3), dtype=np.uint8) * 255

    pose_np = pose.cpu().numpy()
    pose_mean = np.abs(pose_np).mean()
    pose_max = np.abs(pose_np).max()
    pose_nonzero = (np.abs(pose_np) > 0.01).sum()

    joints_in_frame = ((joints_2d[:, 0] >= 0) & (joints_2d[:, 0] < width) &
                       (joints_2d[:, 1] >= 0) & (joints_2d[:, 1] < height)).sum()
    joints_center = joints_2d.mean(axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 0, 0)
    y_offset = 30
    line_height = 22

    texts = [
        "=== Debug Info ===",
        f"world_scale: {world_scale:.2f}",
        "",
        "=== Pose Stats ===",
        f"pose mean: {pose_mean:.4f}",
        f"pose max: {pose_max:.4f}",
        f"pose nonzero: {pose_nonzero}/420",
        "",
        "=== Projection ===",
        f"joints in frame: {joints_in_frame}/140",
        f"joints center: ({joints_center[0]:.0f}, {joints_center[1]:.0f})",
        f"expected: ({width // 2}, {height // 2})",
    ]

    for i, text in enumerate(texts):
        cv2.putText(panel, text, (10, y_offset + i * line_height),
                    font, font_scale, color, 1, cv2.LINE_AA)

    legend_y = y_offset + len(texts) * line_height + 20
    legend_items = [
        ("=== GT Keypoints (MAMMAL 22) ===", None),
        ("Ears + Nose", (92, 94, 170)),
        ("Spine/Body/Tail", (221, 94, 86)),
        ("Left Front Leg", (187, 97, 166)),
        ("Right Front Leg", (109, 192, 91)),
        ("Left Hind Leg", (210, 220, 88)),
        ("Right Hind Leg", (98, 201, 211)),
    ]

    for i, (text, legend_color) in enumerate(legend_items):
        y_pos = legend_y + i * line_height
        if legend_color is not None:
            cv2.circle(panel, (20, y_pos - 5), 8, legend_color, -1)
            cv2.circle(panel, (20, y_pos - 5), 9, (0, 0, 0), 1)
            cv2.putText(panel, text, (35, y_pos),
                        font, font_scale, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(panel, text, (10, y_pos),
                        font, font_scale, color, 1, cv2.LINE_AA)

    return panel
