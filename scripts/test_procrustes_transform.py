#!/usr/bin/env python
"""
Test visualization with Procrustes-computed optimal transformation.
"""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import json

from src.models.mouse_body import load_mouse_model
from src.data import create_mammal_dataloader


# Keypoint 22 mapping from MAMMAL body model
KEYPOINT22_JOINT_MAP = {
    3: [64, 65], 5: [48, 51], 6: [54, 55], 7: [61],
    8: [79], 9: [74], 10: [73], 11: [70],
    12: [104], 13: [99], 14: [98], 15: [95],
    16: [15], 17: [5], 18: [4],
    19: [38], 20: [28], 21: [27],
}

# MAMMAL bone connections
BONES = [
    [0, 2], [1, 2],
    [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
    [8, 9], [9, 10], [10, 11], [11, 3],
    [12, 13], [13, 14], [14, 15], [15, 3],
    [16, 17], [17, 18], [18, 5],
    [19, 20], [20, 21], [21, 5]
]


def extract_keypoints22(joints_3d, device='cpu'):
    B = joints_3d.shape[0]
    keypoints = torch.zeros(B, 22, 3, device=device)
    for kp_idx, joint_ids in KEYPOINT22_JOINT_MAP.items():
        if kp_idx < 22:
            joint_positions = joints_3d[:, joint_ids, :]
            keypoints[:, kp_idx] = joint_positions.mean(dim=1)
    keypoints[:, 0] = joints_3d[:, 64, :]
    keypoints[:, 1] = joints_3d[:, 66, :]
    keypoints[:, 2] = joints_3d[:, 67, :]
    keypoints[:, 4] = joints_3d[:, 0, :]
    return keypoints


def project_points(points_3d, viewmat, K):
    # points_3d: [N, 3]
    N = points_3d.shape[0]
    ones = torch.ones(N, 1, device=points_3d.device, dtype=points_3d.dtype)
    points_homo = torch.cat([points_3d, ones], dim=-1)  # [N, 4]
    points_cam = (viewmat @ points_homo.T).T[:, :3]  # [N, 3]
    points_2d = (K @ points_cam.T).T  # [N, 3]
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]  # [N, 2]
    return points_2d


def draw_skeleton(img, kps_2d, color, bones=BONES):
    """Draw skeleton with bones."""
    for i, j in bones:
        if i < len(kps_2d) and j < len(kps_2d):
            pt1 = tuple(kps_2d[i].astype(int))
            pt2 = tuple(kps_2d[j].astype(int))
            if all(0 <= c < 1500 for c in pt1 + pt2):
                cv2.line(img, pt1, pt2, color, 2)
    for k in range(len(kps_2d)):
        pt = tuple(kps_2d[k].astype(int))
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(img, pt, 5, color, -1)


def test_transform(body_model, batch, transform_params, device, use_procrustes=False):
    """
    Test transformation.

    If use_procrustes=True, apply full rotation matrix from Procrustes.
    Otherwise, use current heuristic (base rotation + yaw).
    """
    images = batch['images'][0]
    viewmats = batch['viewmats'][0]
    Ks = batch['Ks'][0]
    pose = batch['pose']
    global_transform = batch.get('global_transform', {})
    gt_keypoints = batch.get('keypoints2d')

    cam_idx = 0
    viewmat = viewmats[cam_idx].to(device)
    K = Ks[cam_idx].to(device)
    img = images[cam_idx].cpu().numpy()

    # Get body model keypoints
    with torch.no_grad():
        pose_device = pose.to(device)
        vertices = body_model(pose_device)
        joints_3d = body_model._J_posed.clone()
        keypoints_3d = extract_keypoints22(joints_3d, device)  # [1, 22, 3]

    if use_procrustes:
        # Apply Procrustes transformation: scale * R @ X + t
        scale = transform_params['scale']
        R = torch.tensor(transform_params['rotation_matrix'], dtype=torch.float32, device=device)
        t = torch.tensor(transform_params['translation'], dtype=torch.float32, device=device)

        # Transform keypoints
        kps = keypoints_3d[0]  # [22, 3]
        kps_transformed = scale * (kps @ R.T) + t
        keypoints_3d_final = kps_transformed.unsqueeze(0)

    else:
        # Current heuristic transformation
        scale = transform_params.get('scale', 120.0)
        offset = transform_params.get('offset', [140.0, 0.1, 43.9])

        # Get world translation from MAMMAL
        world_trans = None
        if global_transform.get('center') is not None:
            center = global_transform['center'].clone().to(device)
            center = center * 100.0  # meters -> cm
            offset_tensor = torch.tensor(offset, dtype=torch.float32, device=device)
            center = center + offset_tensor
            world_trans = center.unsqueeze(0)

        # Get yaw angle
        yaw_angle = None
        if global_transform.get('angle') is not None:
            angle = global_transform['angle']
            if isinstance(angle, torch.Tensor):
                yaw_angle = angle.unsqueeze(0) if angle.dim() == 0 else angle
            else:
                yaw_angle = torch.tensor([angle], dtype=torch.float32)
            yaw_angle = yaw_angle.to(device)

        # Apply transformations
        kps = keypoints_3d.clone()
        kps = kps * scale

        # Base rotation: Y-up -> Z-up
        jx, jy, jz = kps[..., 0], kps[..., 1], kps[..., 2]
        kps = torch.stack([jx, jz, -jy], dim=-1)

        # Yaw rotation
        if yaw_angle is not None:
            cos_a = torch.cos(yaw_angle)
            sin_a = torch.sin(yaw_angle)
            zeros = torch.zeros_like(cos_a)
            ones = torch.ones_like(cos_a)
            R_yaw = torch.stack([
                torch.stack([cos_a, -sin_a, zeros], dim=-1),
                torch.stack([sin_a, cos_a, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ], dim=-2)
            kps = torch.einsum('bji,bki->bjk', kps, R_yaw)

        # Translation
        if world_trans is not None:
            kps = kps + world_trans.unsqueeze(1)

        keypoints_3d_final = kps

    # Project to 2D
    kps_to_project = keypoints_3d_final[0] if keypoints_3d_final.dim() == 3 else keypoints_3d_final
    kps_to_project = kps_to_project.squeeze()  # Ensure [22, 3]
    kps_2d = project_points(kps_to_project, viewmat, K)
    pred_2d = kps_2d.cpu().numpy()

    # Compute error
    per_kp_errors = {}
    mean_error = float('inf')
    if gt_keypoints is not None:
        gt_kp = gt_keypoints[0, cam_idx].cpu().numpy()
        valid = gt_kp[:, 2] > 0.5
        for k in range(22):
            if valid[k]:
                dist = np.linalg.norm(pred_2d[k] - gt_kp[k, :2])
                per_kp_errors[k] = dist
        if len(per_kp_errors) > 0:
            mean_error = np.mean(list(per_kp_errors.values()))

    # Visualize
    vis_img = (img * 255).astype(np.uint8).copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Draw GT (green)
    if gt_keypoints is not None:
        gt_kp = gt_keypoints[0, cam_idx].cpu().numpy()
        gt_2d = gt_kp[:, :2]
        draw_skeleton(vis_img, gt_2d, (0, 255, 0))

    # Draw predicted (red)
    draw_skeleton(vis_img, pred_2d, (0, 0, 255))

    # Draw error lines (yellow)
    if gt_keypoints is not None:
        gt_kp = gt_keypoints[0, cam_idx].cpu().numpy()
        for k in range(22):
            if gt_kp[k, 2] > 0.5:
                pt1 = tuple(pred_2d[k].astype(int))
                pt2 = tuple(gt_kp[k, :2].astype(int))
                cv2.line(vis_img, pt1, pt2, (0, 255, 255), 1)

    return mean_error, vis_img, per_kp_errors


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
    parser.add_argument('--output-dir', type=str, default='outputs/calibration')
    parser.add_argument('--transform-json', type=str, default='outputs/calibration/optimal_transform.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)

    # Load Procrustes transform
    with open(args.transform_json, 'r') as f:
        procrustes_params = json.load(f)

    print(f"Loaded Procrustes params: scale={procrustes_params['scale']:.2f}")

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

    # Get first valid batch
    batch = None
    for b in dataloader:
        if b['pose'] is not None and (b['pose'].abs() > 0.01).any():
            batch = b
            break

    if batch is None:
        print("No valid frame found")
        return

    print("\n" + "=" * 60)
    print("COMPARING HEURISTIC vs PROCRUSTES")
    print("=" * 60)

    # Test heuristic (best from grid search)
    print("\n[1] Heuristic (scale=120, offset=[140,0.1,43.9]):")
    heuristic_params = {'scale': 120.0, 'offset': [140.0, 0.1, 43.9]}
    err_heur, vis_heur, _ = test_transform(body_model, batch, heuristic_params, device, use_procrustes=False)
    print(f"    Mean error: {err_heur:.1f} px")

    # Test Procrustes
    print("\n[2] Procrustes (optimized):")
    err_proc, vis_proc, per_kp_proc = test_transform(body_model, batch, procrustes_params, device, use_procrustes=True)
    print(f"    Mean error: {err_proc:.1f} px")

    # Create side-by-side comparison
    h, w = vis_heur.shape[:2]
    comparison = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
    comparison[:, :w] = vis_heur
    comparison[:, w+20:] = vis_proc

    # Add labels
    cv2.putText(comparison, f"Heuristic: {err_heur:.1f}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(comparison, f"Procrustes: {err_proc:.1f}px", (w + 30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save
    output_path = output_dir / 'procrustes_comparison.png'
    cv2.imwrite(str(output_path), comparison)
    print(f"\nSaved comparison to {output_path}")

    # Print per-keypoint errors for Procrustes
    print("\nPer-keypoint errors (Procrustes):")
    kp_names = ['Nose', 'L-ear', 'R-ear', 'Spine1', 'Tail', 'Hip', 'Back', 'Tail-start',
                'RF-shoulder', 'RF-elbow', 'RF-wrist', 'RF-paw',
                'LF-shoulder', 'LF-elbow', 'LF-wrist', 'LF-paw',
                'RH-hip', 'RH-knee', 'RH-ankle',
                'LH-hip', 'LH-knee', 'LH-ankle']
    for k, err in sorted(per_kp_proc.items()):
        name = kp_names[k] if k < len(kp_names) else f"kp{k}"
        print(f"  {k:2d} ({name:12s}): {err:.1f} px")


if __name__ == '__main__':
    main()
