"""
Coordinate System Transforms & Constants

Centralizes all coordinate system constants and transforms used across
MoReMouse components. Prevents scattered hardcoded values.

Coordinate systems:
- MAMMAL body model: Y-up (internal joint/vertex space)
- World (camera system): Z-up (after base rotation)
- Canonical (MoReMouse paper): mesh scaled by 1/180, camera at r=2.22
"""

import math
from typing import Optional

import torch


# =============================================================================
# Coordinate Constants
# =============================================================================

# Platform offset (calibrated for markerless_mouse_1_nerf setup)
# Origin of center_rotation.npz is offset from camera world origin.
# Calibrated via scripts/calibrate_transforms.py (2025-12-13)
PLATFORM_OFFSET = torch.tensor([140.0, 0.1, 43.9], dtype=torch.float32)

# World scale (MAMMAL local coords -> world coords in mm)
# Calibrated via scripts/calibrate_grid_compare.py (2025-12-13)
# Best result: scale=160 with neg_yaw gives 119.6px error
WORLD_SCALE_DEFAULT = 160.0

# Base rotation: -90 deg around X axis (Y-up -> Z-up)
# quaternion = [cos(theta/2), sin(theta/2) * axis]
# For -90 deg around X: [cos(-45deg), sin(-45deg), 0, 0]
BASE_ROTATION_QUAT = torch.tensor(
    [math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0, 0],
    dtype=torch.float32,
)

# Canonical space constants (MoReMouse paper Section 3.2)
CANONICAL_MESH_SCALE = 1.0 / 180.0  # Scale to fit in unit sphere
CANONICAL_CAMERA_RADIUS = 2.22      # Camera distance from origin
CANONICAL_FOV_DEG = 29.86           # Field of view in degrees
CANONICAL_IMAGE_SIZE = 800           # Paper uses 800x800

# Unit conversion
METERS_TO_CM = 100.0


# =============================================================================
# Transform Functions
# =============================================================================

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (wxyz convention).

    Args:
        q1: [..., 4] first quaternion
        q2: [..., 4] second quaternion

    Returns:
        [..., 4] product quaternion q1 * q2
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def yup_to_zup_means(means: torch.Tensor) -> torch.Tensor:
    """Transform 3D positions from Y-up to Z-up: [x,y,z] -> [x,z,-y].

    Args:
        means: [..., 3] positions in Y-up coordinate system

    Returns:
        [..., 3] positions in Z-up coordinate system
    """
    x, y, z = means[..., 0], means[..., 1], means[..., 2]
    return torch.stack([x, z, -y], dim=-1)


def yup_to_zup_quaternions(quats: torch.Tensor) -> torch.Tensor:
    """Rotate quaternions from Y-up to Z-up frame.

    Applies -90 deg X-axis rotation to quaternion orientations.

    Args:
        quats: [..., 4] quaternions in (w,x,y,z) format, Y-up frame

    Returns:
        [..., 4] quaternions in Z-up frame
    """
    base = BASE_ROTATION_QUAT.to(dtype=quats.dtype, device=quats.device)
    # Expand base_quat to match quats shape
    expand_shape = [1] * (quats.dim() - 1) + [4]
    base = base.view(*expand_shape).expand_as(quats)
    return quaternion_multiply(base, quats)


def apply_coordinate_transform(
    gaussian_params: dict,
    device: torch.device,
    world_scale: float = WORLD_SCALE_DEFAULT,
) -> dict:
    """Apply world_scale and Y-up to Z-up coordinate transform.

    Body model uses Y-up coordinate system, but camera uses Z-up.
    This applies:
    1. world_scale to means and scales (model coords -> world coords in mm)
    2. -90 degree rotation around X axis: Y -> Z, Z -> -Y

    Args:
        gaussian_params: Dict with 'means', 'scales', 'rotations' tensors
        device: Target torch device
        world_scale: Scale factor (default: calibrated 160.0)

    Returns:
        Transformed gaussian_params dict (modified in-place)
    """
    if world_scale != 1.0:
        gaussian_params["means"] = gaussian_params["means"] * world_scale
        gaussian_params["scales"] = gaussian_params["scales"] * world_scale

    gaussian_params["means"] = yup_to_zup_means(gaussian_params["means"])
    gaussian_params["rotations"] = yup_to_zup_quaternions(
        gaussian_params["rotations"]
    )
    return gaussian_params


def center_rotation_to_world_translation(
    center: torch.Tensor,
    platform_offset: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert center_rotation.npz center (meters) to world translation (cm).

    Args:
        center: [3] or [B, 3] center from center_rotation.npz (in meters)
        platform_offset: Platform offset tensor (default: PLATFORM_OFFSET)

    Returns:
        [3] or [B, 3] world translation in cm
    """
    if platform_offset is None:
        platform_offset = PLATFORM_OFFSET

    result = center.clone() * METERS_TO_CM
    result = result + platform_offset.to(result.device)
    return result


def project_points(
    points_3d: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: [N, 3] 3D points
        viewmat: [4, 4] world-to-camera transform
        K: [3, 3] camera intrinsics

    Returns:
        [N, 2] 2D image coordinates
    """
    N = points_3d.shape[0]
    ones = torch.ones(N, 1, device=points_3d.device, dtype=points_3d.dtype)
    points_homo = torch.cat([points_3d, ones], dim=-1)  # [N, 4]
    points_cam = (viewmat @ points_homo.T).T[:, :3]     # [N, 3]
    points_2d = (K @ points_cam.T).T                     # [N, 3]
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]    # [N, 2]
    return points_2d
