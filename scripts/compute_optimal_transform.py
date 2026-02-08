#!/usr/bin/env python
"""
Compute optimal transformation between body model and camera world coordinates.

Uses Procrustes analysis to find the best rigid transformation (scale + rotation + translation)
that aligns body model keypoints to GT keypoints.

Approach:
1. Triangulate GT 2D keypoints to get 3D positions in camera world
2. Get body model 3D keypoints
3. Solve Procrustes problem: find S, R, t that minimizes ||S*R*X + t - Y||^2
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
    # Spine/body
    3: [64, 65],  # Spine mid
    5: [48, 51],  # Hip
    6: [54, 55],  # Back
    7: [61],      # Tail start
    # Right Front Leg
    8: [79],      # Shoulder
    9: [74],      # Elbow
    10: [73],     # Wrist
    11: [70],     # Paw
    # Left Front Leg
    12: [104],    # Shoulder
    13: [99],     # Elbow
    14: [98],     # Wrist
    15: [95],     # Paw
    # Right Hind Leg
    16: [15],     # Hip
    17: [5],      # Knee
    18: [4],      # Ankle
    # Left Hind Leg
    19: [38],     # Hip
    20: [28],     # Knee
    21: [27],     # Ankle
}


def extract_keypoints22(joints_3d, device='cpu'):
    """Extract 22 keypoints from body model joints."""
    B = joints_3d.shape[0]
    keypoints = torch.zeros(B, 22, 3, device=device)

    for kp_idx, joint_ids in KEYPOINT22_JOINT_MAP.items():
        if kp_idx < 22:
            joint_positions = joints_3d[:, joint_ids, :]
            keypoints[:, kp_idx] = joint_positions.mean(dim=1)

    # Vertex-based keypoints (approximations using nearby joints)
    keypoints[:, 0] = joints_3d[:, 64, :]  # Nose
    keypoints[:, 1] = joints_3d[:, 66, :]  # Left ear
    keypoints[:, 2] = joints_3d[:, 67, :]  # Right ear
    keypoints[:, 4] = joints_3d[:, 0, :]   # Tail base

    return keypoints


def triangulate_points(pts_2d_multi, Ps, valid_mask):
    """
    Triangulate 3D point from multiple 2D observations.

    Args:
        pts_2d_multi: [N_cams, 2] 2D points
        Ps: [N_cams, 3, 4] projection matrices
        valid_mask: [N_cams] boolean mask for valid observations

    Returns:
        pt_3d: [3] 3D point
    """
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return None

    # Build DLT matrix
    A = []
    for i in valid_idx:
        x, y = pts_2d_multi[i]
        P = Ps[i]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X[:3] / X[3]

    return X


def procrustes_analysis(X, Y, allow_scale=True):
    """
    Solve Procrustes problem: find S, R, t that minimizes ||S*R*X + t - Y||^2

    Args:
        X: [N, 3] source points (body model)
        Y: [N, 3] target points (GT in camera world)
        allow_scale: whether to compute optimal scale

    Returns:
        scale: scalar
        R: [3, 3] rotation matrix
        t: [3] translation
        error: mean alignment error
    """
    # Center the points
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)

    X_centered = X - mu_X
    Y_centered = Y - mu_Y

    # Compute scale
    if allow_scale:
        scale = np.sqrt((Y_centered ** 2).sum() / (X_centered ** 2).sum())
    else:
        scale = 1.0

    # Compute rotation using SVD
    H = X_centered.T @ Y_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = mu_Y - scale * (R @ mu_X)

    # Compute error
    X_transformed = scale * (X @ R.T) + t
    error = np.linalg.norm(X_transformed - Y, axis=1).mean()

    return scale, R, t, error


def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (XYZ convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) * 180 / np.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouse-model', type=str,
                       default=os.environ.get('MOUSE_MODEL_DIR'),
                       help='Path to mouse model (env: MOUSE_MODEL_DIR)')
    parser.add_argument('--data-dir', type=str,
                       default=os.environ.get('MAMMAL_DATA_DIR'),
                       help='Path to data directory (env: MAMMAL_DATA_DIR)')
    parser.add_argument('--pose-dir', type=str,
                       default=os.environ.get('MAMMAL_POSE_DIR'),
                       help='MAMMAL pose results directory (env: MAMMAL_POSE_DIR)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-frames', type=int, default=10)
    parser.add_argument('--output', type=str, default='outputs/calibration/optimal_transform.json')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load body model
    print("Loading body model...")
    body_model = load_mouse_model(args.mouse_model, device=device)

    # Create dataloader
    print("Loading data...")
    dataloader = create_mammal_dataloader(
        args.data_dir,
        batch_size=1,
        num_workers=0,
        num_frames=1000,
        image_size=800,
        pose_dir=args.pose_dir,
        require_pose=True,
    )

    print("\n" + "=" * 60)
    print("COMPUTING OPTIMAL TRANSFORMATION")
    print("=" * 60)

    all_body_kps = []
    all_gt_kps_3d = []

    frame_count = 0
    for batch_idx, batch in enumerate(dataloader):
        if frame_count >= args.num_frames:
            break

        pose = batch['pose']
        if pose is None or (pose.abs() < 0.01).all():
            continue

        gt_keypoints = batch.get('keypoints2d')  # [1, C, 22, 3]
        viewmats = batch['viewmats'][0]  # [C, 4, 4]
        Ks = batch['Ks'][0]  # [C, 3, 3]
        global_transform = batch.get('global_transform', {})

        if gt_keypoints is None:
            continue

        n_cams = viewmats.shape[0]

        # Build projection matrices P = K @ [R|t]
        Ps = []
        for c in range(n_cams):
            K = Ks[c].cpu().numpy()
            viewmat = viewmats[c].cpu().numpy()
            R = viewmat[:3, :3]
            t = viewmat[:3, 3]
            P = K @ np.hstack([R, t.reshape(3, 1)])
            Ps.append(P)
        Ps = np.array(Ps)

        # Triangulate GT keypoints
        gt_kps_2d = gt_keypoints[0].cpu().numpy()  # [C, 22, 3]
        gt_kps_3d = []
        valid_kps = []

        for k in range(22):
            pts_2d = gt_kps_2d[:, k, :2]  # [C, 2]
            valid = gt_kps_2d[:, k, 2] > 0.5  # [C]

            pt_3d = triangulate_points(pts_2d, Ps, valid)
            if pt_3d is not None:
                gt_kps_3d.append(pt_3d)
                valid_kps.append(k)

        if len(gt_kps_3d) < 10:
            continue

        gt_kps_3d = np.array(gt_kps_3d)

        # Get body model keypoints
        with torch.no_grad():
            pose_device = pose.to(device)
            vertices = body_model(pose_device)
            joints_3d = body_model._J_posed.clone()  # [1, J, 3]

            # Extract 22 keypoints
            body_kps = extract_keypoints22(joints_3d, device)  # [1, 22, 3]
            body_kps = body_kps[0].cpu().numpy()  # [22, 3]

        # Only use valid keypoints
        body_kps_valid = body_kps[valid_kps]

        all_body_kps.append(body_kps_valid)
        all_gt_kps_3d.append(gt_kps_3d)

        frame_count += 1
        print(f"Frame {frame_count}: {len(valid_kps)} valid keypoints triangulated")

    if len(all_body_kps) == 0:
        print("ERROR: No valid frames found!")
        return

    # Concatenate all keypoints
    all_body_kps = np.concatenate(all_body_kps, axis=0)
    all_gt_kps_3d = np.concatenate(all_gt_kps_3d, axis=0)

    print(f"\nTotal keypoint pairs: {len(all_body_kps)}")

    # Run Procrustes analysis
    print("\nRunning Procrustes analysis...")
    scale, R, t, error = procrustes_analysis(all_body_kps, all_gt_kps_3d)

    # Convert rotation to euler angles
    euler = rotation_matrix_to_euler(R)

    print("\n" + "=" * 60)
    print("OPTIMAL TRANSFORMATION FOUND")
    print("=" * 60)
    print(f"\nScale: {scale:.4f}")
    print(f"\nRotation matrix:")
    print(R)
    print(f"\nEuler angles (XYZ, degrees): {euler}")
    print(f"\nTranslation: {t}")
    print(f"\nMean alignment error: {error:.4f} (in camera world units)")

    # Save results
    results = {
        'scale': float(scale),
        'rotation_matrix': R.tolist(),
        'euler_angles_deg': euler.tolist(),
        'translation': t.tolist(),
        'mean_error': float(error),
        'num_keypoint_pairs': len(all_body_kps),
        'num_frames': frame_count,
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Print code snippet to apply transformation
    print("\n" + "=" * 60)
    print("CODE SNIPPET FOR gaussian_avatar.py:")
    print("=" * 60)
    print(f"""
# Optimal transformation from Procrustes analysis
WORLD_SCALE = {scale:.4f}
BASE_ROTATION = np.array({R.tolist()})
PLATFORM_OFFSET = torch.tensor({t.tolist()}, dtype=torch.float32)
""")


if __name__ == '__main__':
    main()
