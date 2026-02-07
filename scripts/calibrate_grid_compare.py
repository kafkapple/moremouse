#!/usr/bin/env python
"""
Grid comparison script for calibration parameters.

Tests multiple parameter combinations and creates a comparison grid image.
Uses MAMMAL-style keypoint visualization with bone connections.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2

from src.models.mouse_body import load_mouse_model
from src.data import create_mammal_dataloader


# MAMMAL 22-keypoint bone connections
BONES = [
    [0,2], [1,2],                        # ears to spine
    [2,3],[3,4],[4,5],[5,6],[6,7],        # spine
    [8,9], [9,10], [10,11], [11,3],       # right front leg
    [12,13], [13,14], [14,15], [15,3],    # left front leg
    [16,17],[17,18],[18,5],               # right hind leg
    [19,20],[20,21],[21,5]                # left hind leg
]

# MAMMAL color scheme (RGB)
COLORS = {
    'purple': [92, 94, 170],
    'pink': [187, 97, 166],
    'green': [109, 192, 91],
    'red': [221, 94, 86],
    'yellow': [210, 220, 88],
    'blue': [98, 201, 211],
}

BONE_COLORS = [
    'purple', 'purple',                    # ears
    'red', 'red', 'red', 'red', 'red',     # spine
    'pink', 'pink', 'pink', 'pink',        # right front leg
    'green', 'green', 'green', 'green',    # left front leg
    'yellow', 'yellow', 'yellow',          # right hind leg
    'blue', 'blue', 'blue'                 # left hind leg
]

# Joint colors matching MAMMAL
JOINT_COLORS = [
    'purple', 'purple', 'purple',          # 0-2: ears, spine top
    'red', 'red', 'red', 'red', 'red',     # 3-7: spine
    'pink', 'pink', 'pink', 'pink',        # 8-11: right front leg
    'green', 'green', 'green', 'green',    # 12-15: left front leg
    'yellow', 'yellow', 'yellow',          # 16-18: right hind leg
    'blue', 'blue', 'blue', 'blue'         # 19-21: left hind leg (extend to 22)
]

# Keypoint 22 mapping from MAMMAL (keypoint index -> joint/vertex ids)
KEYPOINT22_JOINT_MAP = {
    3: [64, 65],   # spine
    5: [48, 51],   #
    6: [54, 55],   #
    7: [61],       #
    8: [79],       # Right Front Leg
    9: [74],       #
    10: [73],      #
    11: [70],      #
    12: [104],     # Left Front Leg
    13: [99],      #
    14: [98],      #
    15: [95],      #
    16: [15],      # Right Hind Leg
    17: [5],       #
    18: [4],       #
    19: [38],      # Left Hind Leg
    20: [28],      #
    21: [27],      #
}


def project_points(points_3d, viewmat, K):
    """Project 3D points to 2D."""
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=-1)
    points_cam = (viewmat @ points_homo.T).T[:, :3]
    points_2d = (K @ points_cam.T).T
    points_2d = points_2d[:, :2] / points_2d[:, 2:3]
    return points_2d


def build_z_rotation(yaw_angle, device='cpu'):
    """Build Z-axis rotation matrix."""
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


def extract_keypoints22(joints_3d, vertices, device='cpu'):
    """Extract 22 keypoints from body model joints."""
    B = joints_3d.shape[0]
    keypoints = torch.zeros(B, 22, 3, device=device)

    for kp_idx, joint_ids in KEYPOINT22_JOINT_MAP.items():
        if kp_idx < 22:
            joint_positions = joints_3d[:, joint_ids, :]
            keypoints[:, kp_idx] = joint_positions.mean(dim=1)

    # Vertex-based keypoints approximations
    keypoints[:, 0] = joints_3d[:, 64, :]  # Nose
    keypoints[:, 1] = joints_3d[:, 66, :]  # Left ear
    keypoints[:, 2] = joints_3d[:, 67, :]  # Right ear
    keypoints[:, 4] = joints_3d[:, 0, :]   # Tail base

    return keypoints


def draw_keypoints_mammal_style(img, keypoints_2d, color_type='pred', draw_bones=True):
    """
    Draw keypoints in MAMMAL style.

    Args:
        img: Image to draw on
        keypoints_2d: [22, 2] array of 2D keypoints
        color_type: 'pred' (red tones) or 'gt' (green tones)
        draw_bones: Whether to draw bone connections
    """
    h, w = img.shape[:2]

    # Draw bones first (behind joints)
    if draw_bones:
        for bone_idx, (i, j) in enumerate(BONES):
            if i >= len(keypoints_2d) or j >= len(keypoints_2d):
                continue
            pt1 = keypoints_2d[i].astype(int)
            pt2 = keypoints_2d[j].astype(int)
            if not (0 <= pt1[0] < w and 0 <= pt1[1] < h):
                continue
            if not (0 <= pt2[0] < w and 0 <= pt2[1] < h):
                continue

            color_name = BONE_COLORS[bone_idx]
            color = COLORS[color_name]
            # BGR format
            if color_type == 'pred':
                cv2.line(img, tuple(pt1), tuple(pt2), color[::-1], 2)
            else:
                cv2.line(img, tuple(pt1), tuple(pt2), color[::-1], 2)

    # Draw joints
    for k in range(min(len(keypoints_2d), 22)):
        pt = keypoints_2d[k].astype(int)
        if not (0 <= pt[0] < w and 0 <= pt[1] < h):
            continue

        color_name = JOINT_COLORS[k] if k < len(JOINT_COLORS) else 'red'
        color = COLORS[color_name]

        if color_type == 'pred':
            # Red circle with colored fill
            cv2.circle(img, tuple(pt), 6, (0, 0, 255), -1)  # Red fill
            cv2.circle(img, tuple(pt), 6, color[::-1], 2)   # Colored border
        else:
            # Green circle with colored fill
            cv2.circle(img, tuple(pt), 6, (0, 255, 0), -1)  # Green fill
            cv2.circle(img, tuple(pt), 6, color[::-1], 2)   # Colored border

    return img


def test_single_config(body_model, batch, world_scale, platform_offset, negate_yaw=False, device='cpu'):
    """Test a single configuration."""
    body_model.to(device)
    platform_offset_tensor = torch.tensor(platform_offset, dtype=torch.float32, device=device)

    images = batch['images'][0]
    viewmats = batch['viewmats'][0]
    Ks = batch['Ks'][0]
    pose = batch['pose']
    global_transform = batch.get('global_transform', {})
    gt_keypoints = batch.get('keypoints2d')

    if pose is None or (pose.abs() < 0.01).all():
        return float('inf'), None, {}

    # World translation
    world_trans = None
    if global_transform.get('center') is not None:
        center = global_transform['center'].clone().to(device)
        center = center * 100.0
        center = center + platform_offset_tensor
        world_trans = center.unsqueeze(0) if center.dim() == 1 else center

    # Yaw angle
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
        joints_3d_all = body_model._J_posed.clone()
        keypoints_3d = extract_keypoints22(joints_3d_all, vertices, device)
        keypoints_3d = keypoints_3d * world_scale

        # Base rotation: Y-up -> Z-up
        kx, ky, kz = keypoints_3d[..., 0], keypoints_3d[..., 1], keypoints_3d[..., 2]
        keypoints_3d = torch.stack([kx, kz, -ky], dim=-1)

        # Yaw rotation
        if yaw_angle is not None:
            R = build_z_rotation(yaw_angle, device)
            keypoints_3d = torch.einsum('bji,bki->bjk', keypoints_3d, R)

        # World translation
        if world_trans is not None:
            keypoints_3d = keypoints_3d + world_trans.unsqueeze(1)

    # Project and compute error
    cam_idx = 0
    viewmat = viewmats[cam_idx].to(device)
    K = Ks[cam_idx].to(device)
    img = images[cam_idx].cpu().numpy()

    keypoints_2d = project_points(keypoints_3d[0], viewmat, K)
    pred_2d = keypoints_2d.cpu().numpy()

    # Compute per-keypoint errors
    per_kp_errors = {}
    mean_error = float('inf')
    if gt_keypoints is not None:
        gt_kp = gt_keypoints[0, cam_idx]
        valid = gt_kp[:, 2] > 0.5
        gt_2d = gt_kp[:, :2].cpu().numpy()
        valid_np = valid.cpu().numpy()

        for k in range(22):
            if valid_np[k]:
                dist = np.linalg.norm(pred_2d[k] - gt_2d[k])
                per_kp_errors[k] = dist

        if len(per_kp_errors) > 0:
            mean_error = np.mean(list(per_kp_errors.values()))

    # Create visualization
    vis_img = (img * 255).astype(np.uint8).copy()
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # Draw GT keypoints (green) with bones
    if gt_keypoints is not None:
        gt_2d_vis = gt_keypoints[0, cam_idx].cpu().numpy()
        gt_2d_valid = gt_2d_vis.copy()
        gt_2d_valid[gt_2d_vis[:, 2] < 0.5, :2] = -1000  # Move invalid points off-screen
        draw_keypoints_mammal_style(vis_img, gt_2d_valid[:, :2], 'gt', draw_bones=True)

    # Draw predicted keypoints (red) with bones
    draw_keypoints_mammal_style(vis_img, pred_2d, 'pred', draw_bones=True)

    # Draw error lines (cyan)
    if gt_keypoints is not None:
        gt_kp = gt_keypoints[0, cam_idx].cpu().numpy()
        for k in range(22):
            if gt_kp[k, 2] > 0.5:
                pred_pt = pred_2d[k].astype(int)
                gt_pt = gt_kp[k, :2].astype(int)
                cv2.line(vis_img, tuple(pred_pt), tuple(gt_pt), (255, 255, 0), 1)

    return mean_error, vis_img, per_kp_errors


def create_comparison_with_stats(images, labels, errors_list, per_kp_errors_list, cols=3):
    """Create grid with stats panel on the right."""
    n = len(images)
    rows = (n + cols - 1) // cols

    img_h, img_w = None, None
    for img in images:
        if img is not None:
            img_h, img_w = img.shape[:2]
            break
    if img_h is None:
        return None

    # Stats panel width
    stats_w = 350
    label_h = 30
    cell_h = img_h + label_h
    cell_w = img_w

    total_w = cols * cell_w + stats_w
    total_h = rows * cell_h

    grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    # Place images
    for i, (img, label) in enumerate(zip(images, labels)):
        row = i // cols
        col = i % cols
        y_start = row * cell_h
        x_start = col * cell_w

        if img is not None:
            grid[y_start:y_start + img_h, x_start:x_start + img_w] = img

        cv2.putText(grid, label, (x_start + 10, y_start + img_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Stats panel
    stats_x = cols * cell_w + 10
    y = 30

    cv2.putText(grid, "=== Results Summary ===", (stats_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    y += 30

    # Sort by error
    sorted_results = sorted(zip(labels, errors_list), key=lambda x: x[1])

    for i, (label, error) in enumerate(sorted_results):
        color = (0, 128, 0) if i == 0 else (0, 0, 0)  # Best in green
        text = f"{i+1}. {label}: {error:.1f}px"
        cv2.putText(grid, text, (stats_x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y += 22

    y += 20
    cv2.putText(grid, "=== Legend ===", (stats_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y += 25
    cv2.putText(grid, "Green: GT keypoints", (stats_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 128, 0), 1)
    y += 18
    cv2.putText(grid, "Red: Predicted keypoints", (stats_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    y += 18
    cv2.putText(grid, "Cyan lines: Error vectors", (stats_x, y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

    return grid


def save_results_json(output_path, configs, errors, per_kp_errors_list):
    """Save quantitative results to JSON."""
    results = {
        'configs': [],
        'best_config': None,
        'best_error': float('inf')
    }

    for i, (config, error, per_kp) in enumerate(zip(configs, errors, per_kp_errors_list)):
        scale, offset, neg_yaw, label = config
        entry = {
            'label': label,
            'world_scale': float(scale),
            'platform_offset': [float(x) for x in offset],
            'negate_yaw': neg_yaw,
            'mean_error_px': float(error),
            'per_keypoint_errors': {str(k): float(v) for k, v in per_kp.items()}
        }
        results['configs'].append(entry)

        if error < results['best_error']:
            results['best_error'] = float(error)
            results['best_config'] = entry

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mouse-model', type=str, default='/home/joon/MAMMAL_mouse/mouse_model')
    parser.add_argument('--data-dir', type=str, default='/home/joon/data/markerless_mouse_1_nerf')
    parser.add_argument('--pose-dir', type=str,
                       default='/home/joon/MAMMAL_mouse/results/monocular/mouse_batch_20251125_132606_mouse_1')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output-dir', type=str, default='outputs/calibration')
    parser.add_argument('--frame-idx', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading body model...")
    body_model = load_mouse_model(args.mouse_model, device=device)

    print("Loading data...")
    dataloader = create_mammal_dataloader(
        args.data_dir, batch_size=1, num_workers=0, num_frames=100,
        image_size=800, pose_dir=args.pose_dir, require_pose=True,
    )

    print(f"Getting frame {args.frame_idx}...")
    batch = None
    for i, b in enumerate(dataloader):
        if i == args.frame_idx:
            batch = b
            break

    if batch is None:
        print("Error: Could not get frame")
        return

    # Test configurations - focus on higher scales based on Procrustes hint (159)
    configs = [
        (120.0, [140.0, 0.1, 43.9], False, "scale=120"),
        (140.0, [140.0, 0.1, 43.9], False, "scale=140"),
        (160.0, [140.0, 0.1, 43.9], False, "scale=160"),
        (180.0, [140.0, 0.1, 43.9], False, "scale=180"),
        (160.0, [94.2, 6.7, 51.1], False, "scale=160 proc_offset"),
        (160.0, [140.0, 0.1, 43.9], True, "scale=160 neg_yaw"),
    ]

    print("\n" + "=" * 60)
    print("CALIBRATION GRID COMPARISON")
    print("=" * 60)

    images = []
    labels = []
    errors = []
    per_kp_errors_list = []

    for scale, offset, neg_yaw, label in configs:
        print(f"\nTesting: {label}")
        error, vis_img, per_kp = test_single_config(
            body_model, batch, scale, offset, neg_yaw, device
        )
        print(f"  Mean error: {error:.1f} px")

        images.append(vis_img)
        labels.append(label)
        errors.append(error)
        per_kp_errors_list.append(per_kp)

    # Create comparison grid with stats
    print("\nCreating comparison grid...")
    grid = create_comparison_with_stats(images, labels, errors, per_kp_errors_list, cols=3)

    if grid is not None:
        grid_path = output_dir / 'grid_comparison.png'
        cv2.imwrite(str(grid_path), grid)
        print(f"Saved grid to {grid_path}")

    # Save JSON results
    json_path = output_dir / 'calibration_results.json'
    save_results_json(json_path, configs, errors, per_kp_errors_list)
    print(f"Saved results to {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY (sorted by error)")
    print("=" * 60)
    sorted_results = sorted(zip(labels, errors), key=lambda x: x[1])
    for label, error in sorted_results:
        print(f"  {label}: {error:.1f} px")

    print(f"\nBest: {sorted_results[0][0]} ({sorted_results[0][1]:.1f} px)")


if __name__ == '__main__':
    main()
