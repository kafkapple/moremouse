#!/usr/bin/env python
"""
Test MoReMouse canonical space approach.

MoReMouse paper approach:
1. Scale mesh by 1/180 to fit in unit sphere
2. Place cameras on sphere of radius 2.22
3. Use Ψ_g = 0 (no global transform)
4. FoV = 29.86°, Resolution = 800x800

This script:
1. Generates canonical space cameras (random on sphere)
2. Crops real images around mouse centroid
3. Renders mesh in canonical space
4. Compares with cropped real images
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import math

from src.models.mouse_body import load_mouse_model
from src.data import create_mammal_dataloader
from src.utils.geometry import KEYPOINT22_JOINT_MAP, BONES, extract_keypoints22, draw_skeleton
from src.utils.transforms import (
    CANONICAL_MESH_SCALE as MESH_SCALE,
    CANONICAL_CAMERA_RADIUS as CAMERA_RADIUS,
    CANONICAL_FOV_DEG as FOV_DEG,
    CANONICAL_IMAGE_SIZE as IMAGE_SIZE,
)


def fov_to_focal_length(fov_deg: float, image_size: int) -> float:
    """Convert FoV to focal length.

    FoV = 2 * atan(image_size / (2 * focal_length))
    focal_length = image_size / (2 * tan(FoV / 2))
    """
    fov_rad = math.radians(fov_deg)
    focal_length = image_size / (2 * math.tan(fov_rad / 2))
    return focal_length


def generate_canonical_camera(azimuth: float, elevation: float, radius: float = CAMERA_RADIUS) -> np.ndarray:
    """
    Generate camera view matrix for canonical space.

    Camera is placed on sphere looking at origin.

    Args:
        azimuth: Angle around Y-axis (0-360 degrees)
        elevation: Angle from XZ plane (-90 to 90 degrees)
        radius: Distance from origin

    Returns:
        viewmat: [4, 4] view matrix (world-to-camera)
    """
    # Convert to radians
    az = math.radians(azimuth)
    el = math.radians(elevation)

    # Camera position on sphere
    x = radius * math.cos(el) * math.sin(az)
    y = radius * math.sin(el)
    z = radius * math.cos(el) * math.cos(az)

    cam_pos = np.array([x, y, z])

    # Look at origin
    target = np.array([0.0, 0.0, 0.0])

    # Up vector (Y-up for canonical space)
    up = np.array([0.0, 1.0, 0.0])

    # Build camera coordinate system
    forward = target - cam_pos
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    cam_up = np.cross(right, forward)

    # View matrix: [R | t] where t = -R @ cam_pos
    R = np.stack([right, cam_up, -forward], axis=0)  # [3, 3]
    t = -R @ cam_pos  # [3]

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    return viewmat


def generate_canonical_intrinsics(fov_deg: float = FOV_DEG, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Generate camera intrinsics for canonical space."""
    focal = fov_to_focal_length(fov_deg, image_size)
    cx = image_size / 2.0
    cy = image_size / 2.0

    K = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def sample_cameras_on_sphere(n_cameras: int, radius: float = CAMERA_RADIUS) -> list:
    """
    Sample camera positions uniformly on sphere.

    Uses Fibonacci lattice for approximately uniform distribution.

    Args:
        n_cameras: Number of cameras to generate
        radius: Sphere radius

    Returns:
        List of (azimuth, elevation) pairs in degrees
    """
    cameras = []
    golden_ratio = (1 + math.sqrt(5)) / 2

    for i in range(n_cameras):
        # Fibonacci lattice
        theta = 2 * math.pi * i / golden_ratio  # Azimuth
        phi = math.acos(1 - 2 * (i + 0.5) / n_cameras)  # Polar angle

        # Convert to azimuth/elevation
        azimuth = math.degrees(theta) % 360
        elevation = 90 - math.degrees(phi)  # Convert from polar to elevation

        cameras.append((azimuth, elevation))

    return cameras


def crop_image_around_keypoints(
    image: np.ndarray,
    keypoints_2d: np.ndarray,
    K: np.ndarray,
    output_size: int = IMAGE_SIZE,
    padding: float = 1.5,
) -> tuple:
    """
    Crop image around mouse centroid and adjust intrinsics.

    Args:
        image: [H, W, 3] original image
        keypoints_2d: [N, 2] or [N, 3] 2D keypoints (x, y, [conf])
        K: [3, 3] original intrinsics
        output_size: Target crop size
        padding: Padding factor around bounding box

    Returns:
        cropped_image: [output_size, output_size, 3]
        new_K: [3, 3] adjusted intrinsics
        crop_info: dict with crop parameters
    """
    H, W = image.shape[:2]

    # Get valid keypoints
    if keypoints_2d.shape[-1] == 3:
        valid = keypoints_2d[:, 2] > 0.5
        kps = keypoints_2d[valid, :2]
    else:
        kps = keypoints_2d

    if len(kps) < 3:
        # Fallback: use image center
        cx, cy = W / 2, H / 2
        size = min(W, H)
    else:
        # Compute bounding box
        x_min, y_min = kps.min(axis=0)
        x_max, y_max = kps.max(axis=0)

        # Centroid
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        # Size with padding
        w = (x_max - x_min) * padding
        h = (y_max - y_min) * padding
        size = max(w, h)

    # Ensure square crop
    half_size = size / 2

    # Crop bounds (may extend beyond image)
    x1 = int(cx - half_size)
    y1 = int(cy - half_size)
    x2 = int(cx + half_size)
    y2 = int(cy + half_size)

    # Pad image if crop extends beyond bounds
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - W)
    pad_bottom = max(0, y2 - H)

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    # Crop
    cropped = image[y1:y2, x1:x2]

    # Resize to output size
    cropped = cv2.resize(cropped, (output_size, output_size))

    # Adjust intrinsics
    # After crop: x' = (x - x1) * scale, y' = (y - y1) * scale
    scale = output_size / size
    new_K = K.copy()
    new_K[0, 0] *= scale  # fx
    new_K[1, 1] *= scale  # fy
    new_K[0, 2] = (K[0, 2] - x1 + pad_left) * scale  # cx
    new_K[1, 2] = (K[1, 2] - y1 + pad_top) * scale  # cy

    crop_info = {
        'x1': x1 - pad_left,
        'y1': y1 - pad_top,
        'size': size,
        'scale': scale,
        'centroid': (cx, cy),
    }

    return cropped, new_K, crop_info


def project_mesh_canonical(
    vertices: torch.Tensor,
    viewmat: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """
    Project mesh vertices to 2D using canonical camera.

    Args:
        vertices: [V, 3] mesh vertices (already in canonical space)
        viewmat: [4, 4] view matrix
        K: [3, 3] intrinsics

    Returns:
        points_2d: [V, 2] projected 2D points
    """
    V = vertices.shape[0]

    # Convert to numpy
    if isinstance(vertices, torch.Tensor):
        verts = vertices.cpu().numpy()
    else:
        verts = vertices

    # Transform to camera space
    verts_homo = np.concatenate([verts, np.ones((V, 1))], axis=1)  # [V, 4]
    verts_cam = (viewmat @ verts_homo.T).T[:, :3]  # [V, 3]

    # Project
    verts_2d = (K @ verts_cam.T).T  # [V, 3]
    verts_2d = verts_2d[:, :2] / (verts_2d[:, 2:3] + 1e-8)

    return verts_2d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouse-model', type=str,
                       default=os.environ.get('MOUSE_MODEL_DIR'),
                       help='Path to mouse model (env: MOUSE_MODEL_DIR)')
    parser.add_argument('--data-dir', type=str,
                       default=os.environ.get('NERF_DATA_DIR'),
                       help='Path to NeRF capture data (env: NERF_DATA_DIR)')
    parser.add_argument('--pose-dir', type=str,
                       default=os.environ.get('MAMMAL_POSE_DIR'),
                       help='MAMMAL pose results directory (env: MAMMAL_POSE_DIR)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs/canonical_test')
    parser.add_argument('--n-cameras', type=int, default=6, help='Number of canonical cameras')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MoReMouse Canonical Space Test")
    print("=" * 60)
    print(f"\nPaper constants:")
    print(f"  Mesh scale: 1/{int(1/MESH_SCALE)} = {MESH_SCALE:.6f}")
    print(f"  Camera radius: {CAMERA_RADIUS}")
    print(f"  FoV: {FOV_DEG}°")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")

    # Generate canonical cameras
    print(f"\n[1] Generating {args.n_cameras} canonical cameras on sphere...")
    camera_positions = sample_cameras_on_sphere(args.n_cameras)
    K_canonical = generate_canonical_intrinsics()
    focal = K_canonical[0, 0]
    print(f"  Focal length: {focal:.2f} px")

    for i, (az, el) in enumerate(camera_positions):
        print(f"  Camera {i}: azimuth={az:.1f}°, elevation={el:.1f}°")

    # Load body model
    print("\n[2] Loading body model...")
    body_model = load_mouse_model(args.mouse_model, device=device)

    # Load data
    print("\n[3] Loading real data...")
    dataloader = create_mammal_dataloader(
        args.data_dir,
        batch_size=1,
        num_workers=0,
        num_frames=100,
        image_size=IMAGE_SIZE,
        pose_dir=args.pose_dir,
        require_pose=True,
    )

    # Get a valid frame
    batch = None
    for b in dataloader:
        if b['pose'] is not None and (b['pose'].abs() > 0.01).any():
            batch = b
            break

    if batch is None:
        print("ERROR: No valid frame found!")
        return

    frame_idx = batch.get('frame_indices', batch.get('idx', [0]))[0]
    print(f"  Found valid frame: {frame_idx}")

    # Extract data
    images = batch['images'][0]  # [C, H, W, 3]
    gt_keypoints = batch.get('keypoints2d')  # [1, C, 22, 3]
    pose = batch['pose']

    n_cams = images.shape[0]
    print(f"  Number of views: {n_cams}")

    # Get mesh in canonical space
    print("\n[4] Computing canonical mesh...")
    with torch.no_grad():
        # Forward with pose only (Ψ_g = 0)
        result = body_model(pose.to(device))
        # Handle tuple or tensor return
        if isinstance(result, tuple):
            vertices = result[0]  # [1, V, 3]
        else:
            vertices = result
        joints_3d = body_model._J_posed.clone()  # [1, J, 3]

        # Scale to fit in unit sphere (1/180)
        vertices_canonical = vertices * MESH_SCALE
        joints_canonical = joints_3d * MESH_SCALE

        # Extract 22 keypoints
        kps_canonical = extract_keypoints22(joints_canonical, device)  # [1, 22, 3]

    # Compute mesh bounds
    v_min = vertices_canonical[0].min(dim=0)[0].cpu().numpy()
    v_max = vertices_canonical[0].max(dim=0)[0].cpu().numpy()
    v_center = (v_min + v_max) / 2
    v_extent = np.linalg.norm(v_max - v_min)

    print(f"  Canonical mesh bounds: {v_min} to {v_max}")
    print(f"  Center: {v_center}, Extent: {v_extent:.4f}")
    print(f"  Fits in unit sphere: {v_extent/2 < 1.0}")

    # Render canonical views
    print("\n[5] Rendering canonical views...")
    canonical_views = []

    for cam_idx, (az, el) in enumerate(camera_positions):
        viewmat = generate_canonical_camera(az, el)

        # Project keypoints
        kps_3d = kps_canonical[0].cpu().numpy()  # [22, 3]
        kps_2d = project_mesh_canonical(kps_3d, viewmat, K_canonical)  # [22, 2]

        # Create blank canvas
        canvas = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 50  # Dark gray

        # Draw skeleton (red = predicted)
        draw_skeleton(canvas, kps_2d, (0, 0, 255))

        # Add label
        cv2.putText(canvas, f"Canonical cam{cam_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"az={az:.0f} el={el:.0f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        canonical_views.append(canvas)

    # Process real images with cropping
    print("\n[6] Processing real images with mouse-centered crop...")
    real_views = []

    for cam_idx in range(min(n_cams, args.n_cameras)):
        img = images[cam_idx].cpu().numpy()
        img = (img * 255).astype(np.uint8)

        # Get GT keypoints for this camera
        if gt_keypoints is not None:
            gt_kp = gt_keypoints[0, cam_idx].cpu().numpy()

            # Crop around mouse
            K_orig = batch['Ks'][0, cam_idx].cpu().numpy()
            cropped, K_new, crop_info = crop_image_around_keypoints(
                img, gt_kp, K_orig, output_size=IMAGE_SIZE, padding=1.8
            )

            # Draw GT keypoints (green) on cropped image
            # Transform keypoints to crop coordinates
            gt_kp_cropped = gt_kp.copy()
            gt_kp_cropped[:, 0] = (gt_kp[:, 0] - crop_info['x1']) * crop_info['scale']
            gt_kp_cropped[:, 1] = (gt_kp[:, 1] - crop_info['y1']) * crop_info['scale']

            draw_skeleton(cropped, gt_kp_cropped[:, :2], (0, 255, 0))

            # Add label
            cv2.putText(cropped, f"Real cam{cam_idx} (cropped)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            real_views.append(cropped)
        else:
            # Just resize without crop
            resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            cv2.putText(resized, f"Real cam{cam_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            real_views.append(resized)

    # Create comparison visualization
    print("\n[7] Creating comparison visualization...")

    # Top row: canonical views
    # Bottom row: real cropped views
    n_cols = max(len(canonical_views), len(real_views))

    # Pad to same length
    while len(canonical_views) < n_cols:
        canonical_views.append(np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 50)
    while len(real_views) < n_cols:
        real_views.append(np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 50)

    top_row = np.concatenate(canonical_views[:n_cols], axis=1)
    bottom_row = np.concatenate(real_views[:n_cols], axis=1)

    # Add row labels
    label_panel = np.ones((40, top_row.shape[1], 3), dtype=np.uint8) * 30
    cv2.putText(label_panel, "Canonical Space (mesh scaled 1/180, cameras on sphere r=2.22)",
               (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    label_panel2 = np.ones((40, bottom_row.shape[1], 3), dtype=np.uint8) * 30
    cv2.putText(label_panel2, "Real Images (cropped around mouse centroid)",
               (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    comparison = np.concatenate([label_panel, top_row, label_panel2, bottom_row], axis=0)

    # Save
    output_path = output_dir / 'canonical_space_comparison.png'
    cv2.imwrite(str(output_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"\nSaved comparison to {output_path}")

    # Create info panel
    info_path = output_dir / 'canonical_space_info.txt'
    with open(info_path, 'w') as f:
        f.write("MoReMouse Canonical Space Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write("Paper constants:\n")
        f.write(f"  MESH_SCALE = 1/180 = {MESH_SCALE:.6f}\n")
        f.write(f"  CAMERA_RADIUS = {CAMERA_RADIUS}\n")
        f.write(f"  FOV = {FOV_DEG}°\n")
        f.write(f"  IMAGE_SIZE = {IMAGE_SIZE}x{IMAGE_SIZE}\n")
        f.write(f"  FOCAL_LENGTH = {focal:.2f} px\n")
        f.write(f"\nCanonical mesh stats:\n")
        f.write(f"  Bounds: [{v_min[0]:.4f}, {v_min[1]:.4f}, {v_min[2]:.4f}] to ")
        f.write(f"[{v_max[0]:.4f}, {v_max[1]:.4f}, {v_max[2]:.4f}]\n")
        f.write(f"  Center: [{v_center[0]:.4f}, {v_center[1]:.4f}, {v_center[2]:.4f}]\n")
        f.write(f"  Extent: {v_extent:.4f} (should be < 2.0 for unit sphere)\n")
        f.write(f"\nGenerated cameras:\n")
        for i, (az, el) in enumerate(camera_positions):
            f.write(f"  Camera {i}: azimuth={az:.1f}°, elevation={el:.1f}°\n")

    print(f"Saved info to {info_path}")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
MoReMouse 방식의 핵심:
1. 메시를 1/180으로 축소 → unit sphere 안에 들어감
2. 카메라를 반경 2.22 구 위에 배치 → 항상 원점을 바라봄
3. Ψ_g = 0 → global transform 없음, 메시는 항상 원점에 고정
4. 실제 이미지는 마우스 중심으로 crop → canonical space와 유사한 조건

이 방식은 좌표 정렬 문제를 근본적으로 회피함.
단점: 실제 3D 위치 복원 불가
""")


if __name__ == '__main__':
    main()
