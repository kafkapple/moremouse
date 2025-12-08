#!/usr/bin/env python
"""
MoReMouse Inference Script

Run inference on single images or videos.
Outputs novel view synthesis and 3D reconstruction.

Usage:
    python scripts/inference.py --image input.png --checkpoint checkpoints/best.pt
    python scripts/inference.py --video input.mp4 --checkpoint checkpoints/best.pt
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models import MoReMouse


def load_model(checkpoint_path: Path, device: torch.device) -> MoReMouse:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    # Model config
    model_cfg = config.get("model", {})

    encoder_config = {
        "model_name": model_cfg.get("encoder", {}).get("name", "dinov2_vitb14"),
        "freeze": True,
        "input_size": model_cfg.get("encoder", {}).get("input_size", 378),
    }

    triplane_config = {
        "triplane_resolution": model_cfg.get("triplane", {}).get("resolution", 64),
        "triplane_channels": model_cfg.get("triplane", {}).get("channels", 512),
    }

    model = MoReMouse(
        encoder_config=encoder_config,
        triplane_config=triplane_config,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def preprocess_image(
    image: np.ndarray,
    target_size: int = 378,
) -> torch.Tensor:
    """Preprocess image for model input."""
    # Resize
    image = cv2.resize(image, (target_size, target_size))

    # BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # To tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1)

    return image


def create_viewmat(
    theta: float,
    phi: float,
    radius: float = 2.22,
) -> torch.Tensor:
    """Create view matrix for novel view synthesis."""
    # Camera position on sphere
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    position = np.array([x, y, z])

    # Look at origin
    forward = -position / np.linalg.norm(position)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # View matrix
    R = np.stack([right, -up, forward], axis=0)
    t = -R @ position

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t

    return torch.from_numpy(viewmat)


def create_intrinsics(
    image_size: int = 378,
    fov_degrees: float = 29.86,
) -> torch.Tensor:
    """Create camera intrinsics."""
    f = image_size / (2 * np.tan(np.radians(fov_degrees) / 2))

    K = torch.tensor([
        [f, 0, image_size / 2],
        [0, f, image_size / 2],
        [0, 0, 1],
    ], dtype=torch.float32)

    return K


def render_novel_views(
    model: MoReMouse,
    image: torch.Tensor,
    num_views: int = 8,
    output_size: int = 378,
    device: torch.device = None,
) -> list:
    """Render novel views from input image."""
    if device is None:
        device = next(model.parameters()).device

    image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
    K = create_intrinsics(output_size).unsqueeze(0).to(device)

    rendered_views = []

    with torch.no_grad():
        # Encode image to triplane
        triplane = model.encode_image(image)

        # Render from different viewpoints
        for i in range(num_views):
            theta = 2 * np.pi * i / num_views
            phi = np.pi / 3  # 60 degrees elevation

            viewmat = create_viewmat(theta, phi).unsqueeze(0).to(device)

            outputs = model.render_image(
                triplane, viewmat, K,
                height=output_size,
                width=output_size,
            )

            rgb = outputs["rgb"][0].cpu().numpy()  # [H, W, 3]
            rgb = (rgb * 255).astype(np.uint8)

            rendered_views.append({
                "rgb": rgb,
                "theta": theta,
                "phi": phi,
            })

    return rendered_views


def process_image(
    model: MoReMouse,
    image_path: Path,
    output_dir: Path,
    num_views: int = 8,
    device: torch.device = None,
):
    """Process single image."""
    print(f"Processing {image_path}")

    # Load and preprocess image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_tensor = preprocess_image(image)

    # Render novel views
    views = render_novel_views(model, image_tensor, num_views, device=device)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save input
    cv2.imwrite(str(output_dir / "input.png"), image)

    # Save novel views
    for i, view in enumerate(views):
        rgb_bgr = cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"view_{i:02d}.png"), rgb_bgr)

    # Create video from views
    create_rotating_video(views, output_dir / "rotation.mp4")

    print(f"Results saved to {output_dir}")


def process_video(
    model: MoReMouse,
    video_path: Path,
    output_dir: Path,
    num_views: int = 4,
    device: torch.device = None,
):
    """Process video frame by frame."""
    print(f"Processing video {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    all_views = []

    for frame_idx in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = preprocess_image(frame)
        views = render_novel_views(model, image_tensor, num_views, device=device)

        # Save frame views
        frame_dir = frames_dir / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(exist_ok=True)

        for i, view in enumerate(views):
            rgb_bgr = cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_dir / f"view_{i:02d}.png"), rgb_bgr)

        all_views.append(views)

    cap.release()

    # Create output video for each view
    for view_idx in range(num_views):
        output_video_path = output_dir / f"view_{view_idx:02d}.mp4"
        create_video_from_frames(
            [v[view_idx]["rgb"] for v in all_views],
            output_video_path,
            fps,
        )

    print(f"Results saved to {output_dir}")


def create_rotating_video(
    views: list,
    output_path: Path,
    fps: int = 10,
):
    """Create rotating video from views."""
    if not views:
        return

    h, w = views[0]["rgb"].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for view in views:
        rgb_bgr = cv2.cvtColor(view["rgb"], cv2.COLOR_RGB2BGR)
        writer.write(rgb_bgr)

    writer.release()


def create_video_from_frames(
    frames: list,
    output_path: Path,
    fps: int = 30,
):
    """Create video from frames."""
    if not frames:
        return

    h, w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in frames:
        rgb_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(rgb_bgr)

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="MoReMouse Inference")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--video", type=str, help="Input video path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--output", type=str, default="results/inference")
    parser.add_argument("--num-views", type=int, default=8, help="Number of novel views")
    parser.add_argument("--device", type=str, default="cuda:1")

    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("Either --image or --video must be specified")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(Path(args.checkpoint), device)

    # Process
    if args.image:
        process_image(
            model,
            Path(args.image),
            Path(args.output),
            args.num_views,
            device,
        )

    if args.video:
        process_video(
            model,
            Path(args.video),
            Path(args.output),
            args.num_views,
            device,
        )


if __name__ == "__main__":
    main()
