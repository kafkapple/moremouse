#!/usr/bin/env python
"""
Calibration script for world_scale and PLATFORM_OFFSET.

Renders the body model and compares with GT keypoints to find optimal parameters.
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
from tqdm import tqdm

from src.models.mouse_body import load_mouse_model
from src.data import create_mammal_dataloader


def project_points(points_3d, viewmat, K):
    """Project 3D points to 2D using view matrix and intrinsics."""
    # points_3d: [N, 3]
    # viewmat: [4, 4]
    # K: [3, 3]

    # Transform to camera coords
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)  # [N, 4]
    points_cam = (viewmat @ points_homo.T).T[:, :3]  # [N, 3]

    # Project to 2D
    points_2d = (K @ points_cam.T).T  # [N, 3]
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # [N, 2]

    return points_2d


def build_z_rotation(yaw_angle, device='cpu'):
    """Build Z-axis rotation matrix from yaw angle."""
    cos_a = torch.cos(yaw_angle)
    sin_a = torch.sin(yaw_angle)
    zeros = torch.zeros_like(cos_a)
    ones = torch.ones_like(cos_a)

    R = torch.stack([
        torch.stack([cos_a, -sin_a, zeros], dim=-1),
        torch.stack([sin_a, cos_a, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1),
    ], dim=-2)
    return R


def test_parameters(body_model, dataloader, world_scale, platform_offset,
                    negate_yaw=False, device='cpu', num_samples=5):
    """
    Test a set of parameters and compute alignment error.

    Returns:
        mean_error: Mean distance between projected joints and GT keypoints
        vis_images: List of visualization images
    """
    body_model.to(device)
    errors = []
    vis_images = []

    platform_offset_tensor = torch.tensor(platform_offset, dtype=torch.float32, device=device)

    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        # Get data
        images = batch['images'][0]  # [C, H, W, 3]
        viewmats = batch['viewmats'][0]  # [C, 4, 4]
        Ks = batch['Ks'][0]  # [C, 3, 3]
        pose = batch['pose']  # [1, J*3]
        global_transform = batch.get('global_transform', {})
        gt_keypoints = batch.get('keypoints2d')  # [1, C, 22, 3]

        if pose is None or (pose.abs() < 0.01).all():
            continue

        # Get world translation
        world_trans = None
        if global_transform.get('center') is not None:
            center = global_transform['center'].clone().to(device)
            center = center * 100.0  # meters -> cm
            center = center + platform_offset_tensor
            world_trans = center.unsqueeze(0) if center.dim() == 1 else center

        # Get yaw angle
        yaw_angle = None
        if global_transform.get('angle') is not None:
            angle = global_transform['angle']
            if isinstance(angle, torch.Tensor):
                yaw_angle = angle.unsqueeze(0) if angle.dim() == 0 else angle
            else:
                yaw_angle = torch.tensor([angle], dtype=torch.float32)
            if negate_yaw:
                yaw_angle = -yaw_angle
            yaw_angle = yaw_angle.to(device)

        # Forward pass
        with torch.no_grad():
            pose_device = pose.to(device)
            vertices = body_model(pose_device)
            joints_3d = body_model._J_posed.clone()  # [1, J, 3]

            # Apply world_scale
            joints_3d = joints_3d * world_scale

            # Apply base rotation: Body model Y-up -> Camera world Z-up
            # Rotate -90 degrees around X axis: Y -> Z, Z -> -Y
            jx, jy, jz = joints_3d[..., 0], joints_3d[..., 1], joints_3d[..., 2]
            joints_3d = torch.stack([jx, jz, -jy], dim=-1)

            # Apply yaw rotation (after base rotation)
            if yaw_angle is not None:
                R = build_z_rotation(yaw_angle, device)
                joints_3d = torch.einsum('bji,bki->bjk', joints_3d, R)

            # Apply world translation
            if world_trans is not None:
                joints_3d = joints_3d + world_trans.unsqueeze(1)

        # Project to each camera and compare with GT
        for cam_idx in range(min(1, viewmats.shape[0])):  # Only first camera for speed
            viewmat = viewmats[cam_idx].to(device)
            K = Ks[cam_idx].to(device)
            img = images[cam_idx].cpu().numpy()

            # Project joints
            joints_2d = project_points(joints_3d[0], viewmat, K)  # [J, 2]

            # Compare with GT keypoints
            if gt_keypoints is not None:
                gt_kp = gt_keypoints[0, cam_idx]  # [22, 3]
                valid = gt_kp[:, 2] > 0.5
                if valid.sum() > 0:
                    # Use first 22 joints (or as many as we have)
                    n_joints = min(joints_2d.shape[0], 22)
                    pred_2d = joints_2d[:n_joints].cpu().numpy()
                    gt_2d = gt_kp[:n_joints, :2].cpu().numpy()
                    valid_np = valid[:n_joints].cpu().numpy()

                    # Compute error for valid keypoints
                    dists = np.linalg.norm(pred_2d - gt_2d, axis=-1)
                    error = dists[valid_np].mean() if valid_np.sum() > 0 else float('inf')
                    errors.append(error)

            # Create visualization
            vis_img = (img * 255).astype(np.uint8).copy()
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            # Draw projected joints (red)
            for j in range(joints_2d.shape[0]):
                pt = joints_2d[j].cpu().numpy().astype(int)
                if 0 <= pt[0] < vis_img.shape[1] and 0 <= pt[1] < vis_img.shape[0]:
                    cv2.circle(vis_img, tuple(pt), 4, (0, 0, 255), -1)

            # Draw GT keypoints (green)
            if gt_keypoints is not None and gt_keypoints.numel() > 0:
                gt_kp = gt_keypoints[0, cam_idx].cpu().numpy()
                for j in range(gt_kp.shape[0]):
                    if gt_kp[j, 2] > 0.5:
                        pt = gt_kp[j, :2].astype(int)
                        if 0 <= pt[0] < vis_img.shape[1] and 0 <= pt[1] < vis_img.shape[0]:
                            cv2.circle(vis_img, tuple(pt), 4, (0, 255, 0), -1)

            vis_images.append(vis_img)

    mean_error = np.mean(errors) if errors else float('inf')
    return mean_error, vis_images


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
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str, default='outputs/calibration')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load body model
    print("Loading body model...")
    body_model = load_mouse_model(args.mouse_model, device=device)

    # Create dataloader
    print("Loading data...")
    dataloader = create_mammal_dataloader(
        args.data_dir,
        batch_size=1,
        num_workers=0,
        num_frames=100,
        image_size=800,
        pose_dir=args.pose_dir,
        require_pose=True,
    )

    # Current parameters
    current_scale = 47.88
    current_offset = [41.2, 0.1, 43.9]

    print("\n" + "=" * 60)
    print("CALIBRATION TEST")
    print("=" * 60)

    # Test current parameters
    print("\n[1] Testing current parameters...")
    error, vis = test_parameters(body_model, dataloader, current_scale, current_offset,
                                  negate_yaw=False, device=device, num_samples=args.num_samples)
    print(f"  world_scale={current_scale}, offset={current_offset}")
    print(f"  Mean error: {error:.1f} pixels")
    if vis:
        cv2.imwrite(str(output_dir / 'current_params.png'), vis[0])

    # Test with negated yaw
    print("\n[2] Testing with negated yaw...")
    error_neg, vis_neg = test_parameters(body_model, dataloader, current_scale, current_offset,
                                          negate_yaw=True, device=device, num_samples=args.num_samples)
    print(f"  Mean error (negated yaw): {error_neg:.1f} pixels")
    if vis_neg:
        cv2.imwrite(str(output_dir / 'negated_yaw.png'), vis_neg[0])

    # Grid search for scale
    print("\n[3] Grid search for world_scale...")
    # With base rotation applied, try wider range
    scales = [30.0, 40.0, 47.88, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0]
    best_scale = current_scale
    best_scale_error = float('inf')

    for scale in scales:
        error_s, _ = test_parameters(body_model, dataloader, scale, current_offset,
                                      negate_yaw=False, device=device, num_samples=3)
        print(f"  scale={scale}: error={error_s:.1f}")
        if error_s < best_scale_error:
            best_scale_error = error_s
            best_scale = scale

    print(f"\n  Best scale: {best_scale} (error: {best_scale_error:.1f})")

    # Grid search for offset (with best scale)
    print(f"\n[4] Grid search for X offset (with scale={best_scale})...")
    # The projected joints are LEFT of GT -> need to shift RIGHT (+X)
    # Try larger X offsets
    x_offsets = [41.2, 60, 80, 100, 120, 140, 160]
    best_x = 41.2
    best_x_error = float('inf')
    for x in x_offsets:
        offset = [x, 0.1, 43.9]
        error_x, _ = test_parameters(body_model, dataloader, best_scale, offset,
                                      negate_yaw=False, device=device, num_samples=3)
        print(f"  X={x}: error={error_x:.1f}")
        if error_x < best_x_error:
            best_x_error = error_x
            best_x = x

    print(f"\n[5] Grid search for Z offset (with best X={best_x})...")
    z_offsets = [0, 20, 43.9, 60, 80, 100, 120]
    best_z = 43.9
    best_z_error = float('inf')
    for z in z_offsets:
        offset = [best_x, 0.1, z]
        error_z, _ = test_parameters(body_model, dataloader, best_scale, offset,
                                      negate_yaw=False, device=device, num_samples=3)
        print(f"  Z={z}: error={error_z:.1f}")
        if error_z < best_z_error:
            best_z_error = error_z
            best_z = z

    # Also try a few combined offsets
    print(f"\n[6] Testing combined offsets...")
    offsets_to_try = [
        [best_x, 0.1, best_z],
        [best_x + 20, 0.1, best_z + 20],
        [best_x - 20, 0.1, best_z - 20],
    ]
    best_offset = current_offset
    best_offset_error = float('inf')

    for offset in offsets_to_try:
        error_o, _ = test_parameters(body_model, dataloader, best_scale, offset,
                                      negate_yaw=False, device=device, num_samples=3)
        print(f"  offset={offset}: error={error_o:.1f}")
        if error_o < best_offset_error:
            best_offset_error = error_o
            best_offset = offset

    # Final test with best parameters
    print("\n" + "=" * 60)
    print("BEST PARAMETERS:")
    print(f"  world_scale = {best_scale}")
    print(f"  PLATFORM_OFFSET = {best_offset}")
    print(f"  Final error: {best_offset_error:.1f} pixels")
    print("=" * 60)

    # Save final visualization
    error_final, vis_final = test_parameters(body_model, dataloader, best_scale, best_offset,
                                              negate_yaw=False, device=device, num_samples=5)
    if vis_final:
        for i, v in enumerate(vis_final):
            cv2.imwrite(str(output_dir / f'best_params_{i}.png'), v)

    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == '__main__':
    main()
