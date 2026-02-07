#!/usr/bin/env python
"""
Gaussian Avatar Rendering Script

Render the trained Gaussian Avatar from different viewpoints.

Pose Data Location:
    MAMMAL fitting results are stored at:
    /home/joon/MAMMAL_mouse/results/fitting/<experiment_name>/params/

    Each frame has a .pkl file with:
    - thetas: [1, 140, 3] - joint rotations (axis-angle)
    - trans: [1, 3] - global translation
    - rotation: [1, 3] - global rotation
    - bone_lengths: [1, 20] - bone lengths
    - scale: [1, 1] - scale factor

Usage:
    # Render with default (rest) pose
    python scripts/render_avatar.py \
        --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
        --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
        --output-dir results/avatar_renders

    # Render with MAMMAL fitting pose (.pkl format)
    python scripts/render_avatar.py \
        --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
        --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
        --pose-file /home/joon/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/params/step_1_frame_000000.pkl \
        --output-dir results/avatar_renders

    # Render animation sequence from multiple poses
    python scripts/render_avatar.py \
        --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
        --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
        --pose-dir /home/joon/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/params/ \
        --output-dir results/avatar_renders \
        --video

    # Render 360 rotation video (single pose)
    python scripts/render_avatar.py \
        --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
        --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
        --num-views 72 \
        --output-dir results/avatar_renders \
        --video
"""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
from tqdm import tqdm


def create_rotation_cameras(
    num_views: int = 36,
    elevation: float = 30.0,
    radius: float = 500.0,
    image_size: int = 512,
    fov_degrees: float = 50.0,
    look_at: np.ndarray = None,
) -> list:
    """Create cameras rotating around a target point.

    The model uses Y-up coordinate system internally, which is transformed to Z-up
    before rendering. After transform with world_scale=160, model is in mm units.
    Model center is approximately at [0, 10, 0] mm after transform.

    Args:
        num_views: Number of views around the object
        elevation: Camera elevation angle in degrees
        radius: Distance from look_at point in mm (500mm = 50cm for good view of ~80mm mouse)
        image_size: Output image size
        fov_degrees: Field of view in degrees (50 degrees to match training)
        look_at: Target point to look at in mm (default: [0, 10, 0] for transformed model center)
    """
    cameras = []

    # After Y-up to Z-up transform with world_scale=160, model center is ~[0, 10, 0] mm
    if look_at is None:
        look_at = np.array([0.0, 10.0, 0.0])  # Model center in mm

    # Intrinsics - focal length ~1.2x image size works well
    fx = fy = image_size * 1.2
    cx = cy = image_size / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        elev_rad = np.radians(elevation)

        # Camera position relative to look_at point
        # Note: After Y-up to Z-up transform, Z is up
        x = radius * np.cos(elev_rad) * np.cos(azimuth)
        y = radius * np.cos(elev_rad) * np.sin(azimuth)
        z = radius * np.sin(elev_rad)

        cam_pos = look_at + np.array([x, y, z])

        # Look at target point
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)
        up = np.array([0, 0, 1])  # Z-up in world coordinates
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:
            up = np.array([0, 1, 0])
            right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # View matrix (world to camera) - OpenGL convention
        R = np.stack([right, -up, forward], axis=0)  # Changed sign of up and forward
        t = -R @ cam_pos

        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        cameras.append({
            "viewmat": viewmat,
            "K": K,
            "azimuth": np.degrees(azimuth),
            "elevation": elevation,
        })

    return cameras


def load_avatar(checkpoint_path: Path, mouse_model_path: Path, device: torch.device):
    """Load trained Gaussian Avatar."""
    from src.models.mouse_body import load_mouse_model
    from src.models.gaussian_avatar import GaussianAvatar, GaussianAvatarTrainer

    print(f"Loading mouse body model from {mouse_model_path}...")
    body_model = load_mouse_model(str(mouse_model_path), device=device)

    print(f"Loading avatar from {checkpoint_path}...")
    trainer, iteration = GaussianAvatarTrainer.from_checkpoint(
        str(checkpoint_path), body_model, device=device
    )

    print(f"Avatar loaded (trained for {iteration} iterations)")

    return trainer.avatar, body_model


def load_pose(pose_file: Path, device: torch.device = None) -> dict:
    """Load pose from MAMMAL fitting result file.

    Supports two formats:
    1. MAMMAL fitting .pkl format (from results/fitting/.../params/)
       - thetas: [1, 140, 3] joint rotations (axis-angle)
       - trans: [1, 3] global translation
       - rotation: [1, 3] global rotation
       - bone_lengths: [1, 20] bone lengths

    2. Legacy .npz format
       - global_orient: [3] global orientation
       - body_pose: [137*3] joint angles

    Args:
        pose_file: Path to pose file (.pkl or .npz)
        device: Torch device for tensor conversion

    Returns:
        dict with 'pose' tensor [1, 420] and optionally 'bone_lengths' [1, 20]
    """
    pose_file = Path(pose_file)

    if pose_file.suffix == '.pkl':
        # MAMMAL fitting format
        with open(pose_file, 'rb') as f:
            data = pickle.load(f)

        # thetas: [1, 140, 3] -> [1, 420]
        thetas = data['thetas']  # torch.Tensor
        if isinstance(thetas, torch.Tensor):
            pose_tensor = thetas.reshape(1, -1)  # [1, 420]
            if device is not None:
                pose_tensor = pose_tensor.to(device)
        else:
            pose_tensor = torch.tensor(thetas, dtype=torch.float32, device=device).reshape(1, -1)

        result = {'pose': pose_tensor}

        # Optional: bone_lengths
        if 'bone_lengths' in data:
            bone_lengths = data['bone_lengths']
            if isinstance(bone_lengths, torch.Tensor):
                if device is not None:
                    bone_lengths = bone_lengths.to(device)
            else:
                bone_lengths = torch.tensor(bone_lengths, dtype=torch.float32, device=device)
            result['bone_lengths'] = bone_lengths

        return result

    elif pose_file.suffix == '.npz':
        # Legacy npz format
        data = np.load(pose_file)

        global_orient = data.get("global_orient", np.zeros(3))
        body_pose = data.get("body_pose", np.zeros(137 * 3))

        pose_np = np.concatenate([
            global_orient.flatten(),
            body_pose.flatten()
        ])
        pose_tensor = torch.tensor(pose_np, dtype=torch.float32, device=device).unsqueeze(0)

        return {'pose': pose_tensor}

    else:
        raise ValueError(f"Unsupported pose file format: {pose_file.suffix}")


def load_pose_sequence(pose_dir: Path, device: torch.device = None) -> list:
    """Load sequence of poses from a directory.

    Args:
        pose_dir: Directory containing pose files (step_1_frame_*.pkl)
        device: Torch device

    Returns:
        List of pose dicts, sorted by frame number
    """
    pose_dir = Path(pose_dir)

    # Find all pose files
    pkl_files = sorted(pose_dir.glob("step_*_frame_*.pkl"))

    if not pkl_files:
        raise ValueError(f"No pose files found in {pose_dir}")

    poses = []
    for pf in tqdm(pkl_files, desc="Loading poses"):
        poses.append(load_pose(pf, device))

    return poses


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return torch.stack([w, x, y, z], dim=-1)


def apply_coordinate_transform(gaussian_params: dict, device: torch.device, world_scale: float = 160.0) -> dict:
    """Apply world_scale and Y-up to Z-up coordinate transform.

    Body model uses Y-up coordinate system, but our camera uses Z-up.
    This applies:
    1. world_scale to means and scales (model coords -> world coords in mm)
    2. -90 degree rotation around X axis: Y -> Z, Z -> -Y

    This matches the transform applied in GaussianAvatarTrainer._save_visualization()
    """
    import math

    # Apply world_scale to means and scales (like _save_visualization)
    if world_scale != 1.0:
        gaussian_params["means"] = gaussian_params["means"] * world_scale
        gaussian_params["scales"] = gaussian_params["scales"] * world_scale

    # Transform means: [x, y, z] -> [x, z, -y]
    means = gaussian_params["means"]  # [B, N, 3]
    x, y, z = means[..., 0], means[..., 1], means[..., 2]
    gaussian_params["means"] = torch.stack([x, z, -y], dim=-1)

    # Transform quaternions
    quats = gaussian_params["rotations"]  # [B, N, 4]
    # Base rotation: -90 degrees around X axis
    # quaternion = [cos(theta/2), sin(theta/2)*axis]
    # For -90 deg around X: [cos(-45deg), sin(-45deg), 0, 0]
    base_quat = torch.tensor(
        [math.cos(-math.pi/4), math.sin(-math.pi/4), 0, 0],
        dtype=quats.dtype, device=device
    )
    base_quat = base_quat.view(1, 1, 4).expand(quats.shape[0], quats.shape[1], 4)
    gaussian_params["rotations"] = quaternion_multiply(base_quat, quats)

    return gaussian_params


def render_avatar(
    avatar,
    body_model,
    camera: dict,
    pose: torch.Tensor = None,
    bone_lengths: torch.Tensor = None,
    image_size: int = 512,
    device: torch.device = None,
    background_color: tuple = (1, 1, 1),
) -> np.ndarray:
    """Render avatar from a single viewpoint.

    Uses GaussianAvatar's forward() and render() methods.
    Applies coordinate transform from Y-up (body model) to Z-up (camera).

    Args:
        avatar: GaussianAvatar model
        body_model: Mouse body model (used internally by avatar)
        camera: dict with 'viewmat' and 'K' keys
        pose: [1, J*3] joint angles tensor (default: zeros for rest pose)
        bone_lengths: [1, 28] bone lengths (optional)
        image_size: Output image size
        device: Torch device
        background_color: RGB background (default white)

    Returns:
        RGB image as numpy array [H, W, 3] uint8
    """
    # Default to rest pose if not provided
    if pose is None:
        # Use body model's num_joints (140 for MAMMAL mouse model)
        num_joints = avatar.body_model.num_joints
        pose = torch.zeros(1, num_joints * 3, device=device)

    # Camera parameters
    viewmat = torch.tensor(camera["viewmat"], dtype=torch.float32, device=device).unsqueeze(0)
    K = torch.tensor(camera["K"], dtype=torch.float32, device=device).unsqueeze(0)

    # Get Gaussian parameters through forward pass
    with torch.no_grad():
        gaussian_params = avatar.forward(
            pose=pose,
            bone_lengths=bone_lengths,
        )

        # Apply Y-up to Z-up coordinate transform (same as _save_visualization)
        gaussian_params = apply_coordinate_transform(gaussian_params, device)

        # Render using avatar's render method
        rgb, alpha = avatar.render(
            gaussian_params=gaussian_params,
            viewmat=viewmat,
            K=K,
            width=image_size,
            height=image_size,
        )

    # Convert to numpy [B, H, W, 3] -> [H, W, 3]
    rgb = rgb[0].cpu().numpy()
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    return rgb


def create_video(frames: list, output_path: Path, fps: int = 30):
    """Create video from frames."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Saved video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render Gaussian Avatar")
    parser.add_argument("--avatar-checkpoint", type=str, required=True)
    parser.add_argument("--mouse-model", type=str, required=True)
    parser.add_argument("--pose-file", type=str, default=None,
                        help="Single MAMMAL pose file (.pkl or .npz)")
    parser.add_argument("--pose-dir", type=str, default=None,
                        help="Directory with pose sequence (step_*_frame_*.pkl)")
    parser.add_argument("--output-dir", type=str, default="results/avatar_renders")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--elevation", type=float, default=30.0)
    parser.add_argument("--radius", type=float, default=800.0, help="Camera distance in mm")
    parser.add_argument("--video", action="store_true", help="Create rotation video or animation")
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load avatar
    avatar, body_model = load_avatar(
        Path(args.avatar_checkpoint),
        Path(args.mouse_model),
        device,
    )

    # Mode 1: Animation sequence from pose directory
    if args.pose_dir:
        print(f"Loading pose sequence from {args.pose_dir}...")
        poses = load_pose_sequence(Path(args.pose_dir), device)
        print(f"Loaded {len(poses)} poses")

        # Create single camera for animation
        cameras = create_rotation_cameras(
            num_views=1,
            elevation=args.elevation,
            radius=args.radius,
            image_size=args.image_size,
        )
        cam = cameras[0]

        # Render each pose
        frames = []
        for i, pose_data in enumerate(tqdm(poses, desc="Rendering animation")):
            rgb = render_avatar(
                avatar, body_model, cam,
                pose=pose_data['pose'],
                bone_lengths=pose_data.get('bone_lengths'),
                image_size=args.image_size,
                device=device,
            )
            frames.append(rgb)

            # Save individual frame
            cv2.imwrite(
                str(output_dir / f"frame_{i:06d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            )

        # Create animation video
        if args.video and len(frames) > 1:
            create_video(frames, output_dir / "animation.mp4", args.fps)

        print(f"\nAnimation rendering complete!")
        print(f"Output: {output_dir}")
        print(f"Frames: {len(frames)}")
        return

    # Mode 2: Single pose with multiple viewpoints
    pose_tensor = None
    bone_lengths = None
    if args.pose_file:
        pose_data = load_pose(Path(args.pose_file), device)
        pose_tensor = pose_data['pose']
        bone_lengths = pose_data.get('bone_lengths')
        print(f"Loaded pose from {args.pose_file}")

    # Create cameras
    cameras = create_rotation_cameras(
        num_views=args.num_views,
        elevation=args.elevation,
        radius=args.radius,
        image_size=args.image_size,
    )

    # Render
    frames = []
    for i, cam in enumerate(tqdm(cameras, desc="Rendering")):
        rgb = render_avatar(
            avatar, body_model, cam,
            pose=pose_tensor,
            bone_lengths=bone_lengths,
            image_size=args.image_size,
            device=device,
        )
        frames.append(rgb)

        # Save individual image
        cv2.imwrite(
            str(output_dir / f"view_{i:03d}.png"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )

    # Create video if requested
    if args.video and len(frames) > 1:
        create_video(frames, output_dir / "rotation.mp4", args.fps)

    # Create grid
    if len(frames) >= 4:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, min(4, len(frames) // 2), figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            if i < len(frames):
                ax.imshow(frames[i])
                ax.set_title(f"Az: {cameras[i]['azimuth']:.0f}°")
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / "grid.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"\nRendering complete!")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
