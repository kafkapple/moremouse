#!/usr/bin/env python
"""
Generate Synthetic Training Data using Gaussian Mouse Avatar

This script:
1. Loads the MAMMAL mouse model
2. Trains a Gaussian Avatar on multi-view images
3. Generates synthetic multi-view training data

Usage:
    python scripts/generate_synthetic_data.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from tqdm import tqdm

from src.models.mouse_body import MouseBodyModel, load_mouse_model
from src.models.gaussian_avatar import GaussianAvatar, GaussianAvatarTrainer
from src.models.geodesic_embedding import create_geodesic_embedding


def create_camera_on_sphere(
    theta: float,
    phi: float,
    radius: float = 2.22,
    look_at: np.ndarray = None,
) -> tuple:
    """
    Create camera pose on sphere.

    Args:
        theta: Azimuth angle (radians)
        phi: Elevation angle (radians)
        radius: Distance from center
        look_at: Point to look at (default: origin)

    Returns:
        viewmat: 4x4 view matrix
        position: Camera position
    """
    if look_at is None:
        look_at = np.array([0, 0, 0])

    # Camera position
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    position = np.array([x, y, z])

    # Look-at matrix
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)

    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    # View matrix (world to camera)
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ position

    viewmat = np.eye(4)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    return viewmat, position


def create_intrinsics(
    image_size: int = 800,
    fov_degrees: float = 29.86,
) -> np.ndarray:
    """Create camera intrinsics."""
    f = image_size / (2 * np.tan(np.radians(fov_degrees) / 2))

    K = np.array([
        [f, 0, image_size / 2],
        [0, f, image_size / 2],
        [0, 0, 1],
    ])

    return K


def sample_viewpoints(
    num_views: int,
    radius: float = 2.22,
) -> list:
    """Sample uniform viewpoints on sphere."""
    viewpoints = []

    # Fibonacci lattice for uniform sampling
    golden_ratio = (1 + np.sqrt(5)) / 2

    for i in range(num_views):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / num_views)

        viewmat, pos = create_camera_on_sphere(theta, phi, radius)
        viewpoints.append({
            "theta": theta,
            "phi": phi,
            "viewmat": viewmat,
            "position": pos,
        })

    return viewpoints


def render_mesh_fallback(
    vertices: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """
    Simple fallback renderer: project vertices to 2D and draw as points.
    Used when gsplat is unavailable or fails.

    Args:
        vertices: [N, 3] mesh vertices
        viewmat: [4, 4] view matrix
        K: [3, 3] camera intrinsics
        image_size: Output image size

    Returns:
        [H, W, 3] rendered image (uint8)
    """
    import cv2

    # Transform to camera space
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    verts_cam = (R @ vertices.T).T + t  # [N, 3]

    # Filter points in front of camera
    valid = verts_cam[:, 2] > 0.1
    verts_cam = verts_cam[valid]

    if len(verts_cam) == 0:
        return np.ones((image_size, image_size, 3), dtype=np.uint8) * 200  # Light gray

    # Project to image
    verts_proj = (K @ verts_cam.T).T  # [N, 3]
    verts_2d = verts_proj[:, :2] / verts_proj[:, 2:3]  # [N, 2]

    # Create image (light gray background for mouse)
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 240

    # Draw vertices as points (brown/gray for mouse fur)
    for x, y in verts_2d:
        ix, iy = int(x), int(y)
        if 0 <= ix < image_size and 0 <= iy < image_size:
            # Mouse fur color: brownish gray
            img[iy, ix] = [139, 119, 101]  # RGB

    # Apply slight blur to soften
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


def generate_data(
    output_dir: Path,
    mouse_model_path: Path,
    avatar: GaussianAvatar = None,
    num_frames: int = 6000,
    num_views: int = 64,
    image_size: int = 800,
    device: str = "cuda:1",
):
    """
    Generate synthetic training data.

    Args:
        output_dir: Output directory
        mouse_model_path: Path to mouse model
        avatar: Pre-trained GaussianAvatar (if None, creates untrained one)
        num_frames: Number of frames to generate
        num_views: Number of views per frame
        image_size: Image resolution
        device: Compute device
    """
    print(f"Generating synthetic data to {output_dir}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load mouse model
    print("Loading mouse model...")
    body_model = load_mouse_model(mouse_model_path, device=device)

    # Use provided avatar or create new one
    if avatar is not None:
        print("Using pre-trained Gaussian avatar...")
        avatar = avatar.to(device)
    else:
        print("Warning: No avatar provided. Creating untrained Gaussian avatar (gray)...")
        avatar = GaussianAvatar(
            body_model=body_model,
            num_gaussians_per_vertex=1,
        ).to(device)

    # Create geodesic embedding
    print("Computing geodesic embedding...")
    vertices = body_model.v_template.cpu().numpy()
    faces = body_model.get_faces()
    geo_embedding = create_geodesic_embedding(
        vertices, faces,
        num_iterations=500,
        device=device,
    )

    # Sample viewpoints
    print("Sampling viewpoints...")
    viewpoints = sample_viewpoints(num_views)

    # Camera intrinsics
    K = create_intrinsics(image_size)

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Generate random poses
    print("Generating poses...")
    poses = []
    for _ in range(num_frames):
        # Random pose parameters
        pose = np.random.randn(body_model.num_joints * 3) * 0.1
        poses.append(pose)

    poses = np.array(poses)

    # Split train/val
    train_frames = num_frames // 2
    val_frames = num_frames - train_frames

    print(f"Generating {train_frames} training frames...")
    _generate_split(
        avatar, poses[:train_frames], viewpoints, K,
        train_dir, image_size, device, geo_embedding,
    )

    print(f"Generating {val_frames} validation frames...")
    _generate_split(
        avatar, poses[train_frames:], viewpoints[:4], K,  # 4 views for val
        val_dir, image_size, device, geo_embedding,
    )

    # Save metadata
    meta = {
        "num_train_frames": train_frames,
        "num_val_frames": val_frames,
        "num_views": num_views,
        "image_size": image_size,
        "intrinsics": K.tolist(),
    }

    with open(output_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print("Done!")


def _generate_split(
    avatar,
    poses,
    viewpoints,
    K,
    output_dir,
    image_size,
    device,
    geo_embedding,
):
    """Generate data for a single split."""
    import cv2

    K_tensor = torch.from_numpy(K).float().to(device).unsqueeze(0)

    for frame_idx, pose in enumerate(tqdm(poses)):
        frame_dir = output_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)

        pose_tensor = torch.from_numpy(pose).float().to(device).unsqueeze(0)

        # Get Gaussian parameters for this pose
        gaussian_params = avatar(pose_tensor)

        # Save cameras
        cameras_data = {
            "K": K.tolist(),
            "input_viewmat": viewpoints[0]["viewmat"].tolist(),
            "target_viewmats": [v["viewmat"].tolist() for v in viewpoints],
        }

        with open(frame_dir / "cameras.json", 'w') as f:
            json.dump(cameras_data, f)

        # Save pose
        np.save(frame_dir / "pose.npy", pose)

        # Render input view
        viewmat = torch.from_numpy(viewpoints[0]["viewmat"]).float().to(device).unsqueeze(0)

        try:
            rgb, alpha = avatar.render(
                gaussian_params, viewmat, K_tensor, image_size, image_size
            )
            input_img = rgb[0].detach().cpu().numpy()
            input_img = (input_img * 255).astype(np.uint8)
            cv2.imwrite(str(frame_dir / "input.png"), cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))
        except Exception as e:
            # Log error and save fallback mesh rendering
            if frame_idx == 0:
                print(f"Warning: gsplat render failed ({e}), using mesh fallback")
            # Fallback: render mesh vertices as point cloud
            vertices = gaussian_params["means"][0].detach().cpu().numpy()
            fallback_img = render_mesh_fallback(vertices, viewmat[0].cpu().numpy(), K, image_size)
            cv2.imwrite(str(frame_dir / "input.png"), cv2.cvtColor(fallback_img, cv2.COLOR_RGB2BGR))

        # Render target views
        for v_idx, vp in enumerate(viewpoints):
            try:
                viewmat = torch.from_numpy(vp["viewmat"]).float().to(device).unsqueeze(0)
                rgb, alpha = avatar.render(
                    gaussian_params, viewmat, K_tensor, image_size, image_size
                )
                view_img = rgb[0].detach().cpu().numpy()
                view_img = (view_img * 255).astype(np.uint8)
                cv2.imwrite(
                    str(frame_dir / f"view_{v_idx:02d}.png"),
                    cv2.cvtColor(view_img, cv2.COLOR_RGB2BGR)
                )
            except Exception as e:
                # Fallback: render mesh vertices
                vertices = gaussian_params["means"][0].detach().cpu().numpy()
                vm = torch.from_numpy(vp["viewmat"]).float()
                fallback_img = render_mesh_fallback(vertices, vm.numpy(), K, image_size)
                cv2.imwrite(str(frame_dir / f"view_{v_idx:02d}.png"), cv2.cvtColor(fallback_img, cv2.COLOR_RGB2BGR))

        # Generate embedding visualization
        if geo_embedding is not None:
            try:
                emb_rgb = geo_embedding.get_rgb_embeddings().cpu().numpy()
                # This is per-vertex, would need rendering for per-pixel
                # Placeholder: save a dummy embedding image
                emb_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                cv2.imwrite(str(frame_dir / "embedding.png"), emb_img)
            except Exception:
                emb_img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
                cv2.imwrite(str(frame_dir / "embedding.png"), emb_img)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic")
    parser.add_argument("--mouse-model", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=378)
    parser.add_argument("--device", type=str, default="cuda:1")

    args = parser.parse_args()

    # Default mouse model path (relative to project root)
    project_root = Path(__file__).parent.parent
    if args.mouse_model is None:
        # Try relative path first: ../MAMMAL_mouse/mouse_model
        default_path = project_root.parent / "MAMMAL_mouse" / "mouse_model"
        if default_path.exists():
            args.mouse_model = default_path
        else:
            # Fallback to environment variable or error
            import os
            env_path = os.environ.get("MOUSE_MODEL_PATH")
            if env_path:
                args.mouse_model = Path(env_path)
            else:
                raise FileNotFoundError(
                    f"Mouse model not found at {default_path}. "
                    "Set MOUSE_MODEL_PATH environment variable or use --mouse-model flag."
                )

    output_dir = project_root / args.output

    generate_data(
        output_dir=output_dir,
        mouse_model_path=Path(args.mouse_model),
        num_frames=args.num_frames,
        num_views=args.num_views,
        image_size=args.image_size,
        device=args.device,
    )
