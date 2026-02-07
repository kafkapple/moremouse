#!/usr/bin/env python
"""
MoReMouse 3D Visualization Script

Generates comprehensive 3D visualizations:
1. 360° rotation video
2. Multi-view rendering grid
3. Depth map visualization
4. Normal map visualization
5. Geodesic embedding visualization
6. Mesh export (if DMTet)

Usage:
    python scripts/visualize_3d.py \
        --checkpoint checkpoints/best.pt \
        --input-image test.png \
        --output-dir results/visualization
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

plt.switch_backend('Agg')


def load_model(checkpoint_path: Path, device: torch.device):
    """Load MoReMouse model from checkpoint."""
    from src.models import MoReMouse

    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})

    encoder_cfg = model_cfg.get("encoder", {})
    encoder_config = {
        "model_name": encoder_cfg.get("name", "dinov2_vits14"),
        "freeze": True,
        "input_size": encoder_cfg.get("input_size", 224),
    }

    triplane_cfg = model_cfg.get("triplane", {})
    triplane_config = {
        "triplane_resolution": triplane_cfg.get("resolution", 32),
        "triplane_channels": triplane_cfg.get("channels", 128),
    }

    model = MoReMouse(
        encoder_config=encoder_config,
        triplane_config=triplane_config,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, config


def create_rotation_cameras(
    num_views: int = 36,
    elevation: float = 30.0,
    radius: float = 2.5,
) -> List[Dict]:
    """Create cameras rotating around the object."""
    cameras = []

    for i in range(num_views):
        azimuth = 2 * np.pi * i / num_views
        elev_rad = np.radians(elevation)

        # Camera position
        x = radius * np.cos(elev_rad) * np.cos(azimuth)
        y = radius * np.cos(elev_rad) * np.sin(azimuth)
        z = radius * np.sin(elev_rad)

        cam_pos = np.array([x, y, z])

        # Look at origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # View matrix (world to camera)
        R = np.stack([right, up, -forward], axis=0)
        t = -R @ cam_pos

        viewmat = np.eye(4)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = t

        cameras.append({
            "viewmat": viewmat.astype(np.float32),
            "azimuth": np.degrees(azimuth),
            "elevation": elevation,
            "position": cam_pos,
        })

    return cameras


def render_novel_views(
    model,
    input_image: torch.Tensor,
    cameras: List[Dict],
    K: torch.Tensor,
    image_size: int,
    device: torch.device,
) -> List[Dict]:
    """Render from multiple camera viewpoints."""
    results = []

    with torch.no_grad():
        # Encode image once
        triplane = model.encode_image(input_image.unsqueeze(0).to(device))

        for cam in tqdm(cameras, desc="Rendering views"):
            viewmat = torch.from_numpy(cam["viewmat"]).float().to(device).unsqueeze(0)

            outputs = model.render_image(
                triplane, viewmat, K.unsqueeze(0).to(device),
                height=image_size, width=image_size
            )

            result = {
                "rgb": outputs["rgb"][0].cpu().numpy(),
                "alpha": outputs["alpha"][0].cpu().numpy(),
                "azimuth": cam["azimuth"],
                "elevation": cam["elevation"],
            }

            if "depth" in outputs:
                result["depth"] = outputs["depth"][0].cpu().numpy()

            if "embedding" in outputs:
                result["embedding"] = outputs["embedding"][0].cpu().numpy()

            results.append(result)

    return results


def create_rotation_video(
    frames: List[np.ndarray],
    output_path: Path,
    fps: int = 30,
    image_size: int = 512,
):
    """Create rotation video from frames."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (image_size, image_size))

    for frame in frames:
        # Ensure proper format
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # Resize if needed
        if frame.shape[0] != image_size:
            frame = cv2.resize(frame, (image_size, image_size))

        # Convert to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"Saved video: {output_path}")


def visualize_depth(depth: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Colorize depth map."""
    # Normalize
    valid_mask = depth > 0 if mask is None else mask > 0.5
    if valid_mask.sum() > 0:
        d_min = depth[valid_mask].min()
        d_max = depth[valid_mask].max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
    else:
        depth_norm = depth

    # Apply colormap
    depth_colored = plt.cm.viridis(depth_norm)[:, :, :3]
    depth_colored = (depth_colored * 255).astype(np.uint8)

    # Apply mask
    if mask is not None:
        depth_colored[mask < 0.5] = 255  # White background

    return depth_colored


def visualize_normal(depth: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Compute and visualize normal map from depth."""
    # Compute gradients
    dz_dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

    # Normal = normalize(-dz/dx, -dz/dy, 1)
    normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    normal = normal / norm

    # Map to RGB [0, 255]
    normal_vis = ((normal + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

    # Apply mask
    if mask is not None:
        for c in range(3):
            normal_vis[:, :, c][mask < 0.5] = 128  # Gray background

    return normal_vis


def visualize_embedding(embedding: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Visualize geodesic embedding as RGB."""
    # Normalize each channel
    emb_vis = embedding.copy()
    for c in range(emb_vis.shape[-1]):
        channel = emb_vis[:, :, c]
        if mask is not None:
            valid = channel[mask > 0.5]
            if len(valid) > 0:
                c_min, c_max = valid.min(), valid.max()
                emb_vis[:, :, c] = (channel - c_min) / (c_max - c_min + 1e-8)
        else:
            emb_vis[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

    emb_vis = (emb_vis * 255).clip(0, 255).astype(np.uint8)

    # Apply mask
    if mask is not None:
        for c in range(3):
            emb_vis[:, :, c][mask < 0.5] = 255  # White background

    return emb_vis


def create_multi_view_grid(
    renders: List[Dict],
    input_image: np.ndarray,
    output_path: Path,
    grid_size: Tuple[int, int] = (3, 4),
):
    """Create grid of rendered views."""
    rows, cols = grid_size
    n_views = min(len(renders), rows * cols - 1)  # -1 for input image

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    # Input image in first position
    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title("Input", fontsize=12)
    axes[0, 0].axis('off')

    # Rendered views
    for i in range(n_views):
        r = (i + 1) // cols
        c = (i + 1) % cols
        rgb = renders[i]["rgb"]
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        axes[r, c].imshow(rgb)
        axes[r, c].set_title(f"Az: {renders[i]['azimuth']:.0f}°", fontsize=10)
        axes[r, c].axis('off')

    # Hide unused axes
    for i in range(n_views + 1, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-view grid: {output_path}")


def create_output_comparison(
    input_image: np.ndarray,
    rgb: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    embedding: np.ndarray,
    alpha: np.ndarray,
    output_path: Path,
):
    """Create comprehensive output comparison."""
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.1)

    # Row 1: Input, RGB, Alpha, Depth, Normal
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(input_image)
    ax1.set_title("Input", fontsize=12)
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    rgb_vis = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1 else rgb
    ax2.imshow(rgb_vis)
    ax2.set_title("Rendered RGB", fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    alpha_vis = plt.cm.gray(alpha)[:, :, :3]
    ax3.imshow((alpha_vis * 255).astype(np.uint8))
    ax3.set_title("Alpha/Mask", fontsize=12)
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[0, 3])
    if depth is not None:
        depth_vis = visualize_depth(depth, alpha)
        ax4.imshow(depth_vis)
    ax4.set_title("Depth", fontsize=12)
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[0, 4])
    if normal is not None:
        ax5.imshow(normal)
    ax5.set_title("Normal", fontsize=12)
    ax5.axis('off')

    # Row 2: Embedding channels and combined
    if embedding is not None and embedding.shape[-1] >= 3:
        ax6 = fig.add_subplot(gs[1, 0])
        emb_r = visualize_embedding(embedding[:, :, 0:1].repeat(3, axis=-1), alpha)
        ax6.imshow(emb_r)
        ax6.set_title("Embedding Ch.0", fontsize=12)
        ax6.axis('off')

        ax7 = fig.add_subplot(gs[1, 1])
        emb_g = visualize_embedding(embedding[:, :, 1:2].repeat(3, axis=-1), alpha)
        ax7.imshow(emb_g)
        ax7.set_title("Embedding Ch.1", fontsize=12)
        ax7.axis('off')

        ax8 = fig.add_subplot(gs[1, 2])
        emb_b = visualize_embedding(embedding[:, :, 2:3].repeat(3, axis=-1), alpha)
        ax8.imshow(emb_b)
        ax8.set_title("Embedding Ch.2", fontsize=12)
        ax8.axis('off')

        ax9 = fig.add_subplot(gs[1, 3])
        emb_combined = visualize_embedding(embedding, alpha)
        ax9.imshow(emb_combined)
        ax9.set_title("Geodesic Embedding (RGB)", fontsize=12)
        ax9.axis('off')

        # Histogram of embedding values
        ax10 = fig.add_subplot(gs[1, 4])
        for c, color in enumerate(['red', 'green', 'blue']):
            valid = embedding[:, :, c][alpha > 0.5].flatten()
            ax10.hist(valid, bins=50, alpha=0.5, color=color, label=f'Ch.{c}')
        ax10.set_title("Embedding Distribution", fontsize=12)
        ax10.legend(fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved output comparison: {output_path}")


def create_training_evolution_gif(
    checkpoint_dir: Path,
    model,
    input_tensor: torch.Tensor,
    K: torch.Tensor,
    device: torch.device,
    output_path: Path,
    image_size: int = 224,
):
    """Create GIF showing training evolution."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        print("No periodic checkpoints found for evolution visualization")
        return

    frames = []
    labels = []

    for ckpt_path in tqdm(checkpoints[:16], desc="Loading checkpoints"):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            step = checkpoint.get("global_step", 0)

            with torch.no_grad():
                viewmat = torch.eye(4).unsqueeze(0).to(device)
                outputs = model(
                    input_tensor.unsqueeze(0).to(device),
                    viewmats=viewmat,
                    Ks=K.unsqueeze(0).to(device),
                )
                rgb = outputs["rgb"][0].cpu().numpy()
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

                # Add text label
                frame = rgb.copy()
                cv2.putText(
                    frame,
                    f"Epoch {epoch}, Step {step}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                frames.append(frame)
                labels.append(f"Epoch {epoch}")

        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    # Save as GIF
    if frames:
        import imageio
        imageio.mimsave(str(output_path), frames, duration=0.5)
        print(f"Saved training evolution GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="MoReMouse 3D Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/visualization")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-views", type=int, default=36)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory with periodic checkpoints for evolution GIF")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model(Path(args.checkpoint), device)

    # Load input image
    print(f"Loading input image: {args.input_image}")
    img = cv2.imread(args.input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (args.image_size, args.image_size))
    input_np = img.copy()

    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # [C, H, W]

    # Create camera intrinsics
    fx = fy = args.image_size / (2 * np.tan(np.radians(30) / 2))
    cx = cy = args.image_size / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    # =======================
    # 1. Render novel views
    # =======================
    print("\n=== Rendering Novel Views ===")
    cameras = create_rotation_cameras(args.num_views, elevation=30.0)
    renders = render_novel_views(model, img_tensor, cameras, K, args.image_size, device)

    # =======================
    # 2. Create rotation video
    # =======================
    print("\n=== Creating Rotation Video ===")
    rgb_frames = [(r["rgb"] * 255).clip(0, 255).astype(np.uint8) for r in renders]
    create_rotation_video(rgb_frames, output_dir / "rotation_rgb.mp4", args.video_fps, args.image_size)

    # Depth video if available
    if "depth" in renders[0]:
        depth_frames = [visualize_depth(r["depth"], r["alpha"]) for r in renders]
        create_rotation_video(depth_frames, output_dir / "rotation_depth.mp4", args.video_fps, args.image_size)

    # =======================
    # 3. Multi-view grid
    # =======================
    print("\n=== Creating Multi-View Grid ===")
    create_multi_view_grid(renders, input_np, output_dir / "multi_view_grid.png")

    # =======================
    # 4. Comprehensive output
    # =======================
    print("\n=== Creating Output Comparison ===")
    front_render = renders[0]
    rgb = front_render["rgb"]
    alpha = front_render["alpha"]
    depth = front_render.get("depth")
    embedding = front_render.get("embedding")

    normal = visualize_normal(depth, alpha) if depth is not None else None

    create_output_comparison(
        input_np, rgb, depth, normal, embedding, alpha,
        output_dir / "output_comparison.png"
    )

    # =======================
    # 5. Save individual outputs
    # =======================
    print("\n=== Saving Individual Outputs ===")

    # Save RGB views
    views_dir = output_dir / "views"
    views_dir.mkdir(exist_ok=True)
    for i, r in enumerate(renders):
        rgb_frame = (r["rgb"] * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(views_dir / f"view_{i:03d}.png"), cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))

    # =======================
    # 6. Training evolution (if checkpoints available)
    # =======================
    if args.checkpoint_dir:
        print("\n=== Creating Training Evolution GIF ===")
        try:
            create_training_evolution_gif(
                Path(args.checkpoint_dir),
                model,
                img_tensor,
                K,
                device,
                output_dir / "training_evolution.gif",
                args.image_size,
            )
        except ImportError:
            print("imageio not available, skipping GIF generation")

    print(f"\n=== Visualization Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
