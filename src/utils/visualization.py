"""
Visualization Utilities for MoReMouse

Includes:
- Novel view rendering
- Normal map computation and visualization
- Rotation video generation
- Comparison grids
- Depth map visualization
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def compute_normal_map(
    depth: torch.Tensor,
    K: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute normal map from depth map.

    Args:
        depth: [B, H, W] or [H, W] depth map
        K: [3, 3] camera intrinsics (optional, for perspective correction)

    Returns:
        [B, H, W, 3] or [H, W, 3] normal map in [-1, 1]
    """
    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    B, H, W = depth.shape
    device = depth.device

    # Compute gradients using Sobel
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=depth.dtype, device=device).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=depth.dtype, device=device).view(1, 1, 3, 3) / 8.0

    depth_4d = depth.unsqueeze(1)  # [B, 1, H, W]
    dz_dx = F.conv2d(depth_4d, sobel_x, padding=1).squeeze(1)  # [B, H, W]
    dz_dy = F.conv2d(depth_4d, sobel_y, padding=1).squeeze(1)  # [B, H, W]

    # Construct normal vectors
    # n = (-dz/dx, -dz/dy, 1), normalized
    normals = torch.stack([-dz_dx, -dz_dy, torch.ones_like(dz_dx)], dim=-1)
    normals = F.normalize(normals, dim=-1)

    if squeeze:
        normals = normals.squeeze(0)

    return normals


def normal_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    """
    Convert normal map to RGB visualization.

    Args:
        normals: [..., 3] normal vectors in [-1, 1]

    Returns:
        [..., 3] RGB colors in [0, 1]
    """
    # Map [-1, 1] to [0, 1]
    return (normals + 1) / 2


def create_rotation_cameras(
    num_views: int = 36,
    radius: float = 2.22,
    elevation: float = 30.0,
    center: np.ndarray = None,
) -> List[Dict]:
    """
    Create camera poses for 360-degree rotation.

    Args:
        num_views: Number of views (frames)
        radius: Distance from center
        elevation: Elevation angle in degrees
        center: Look-at point (default: origin)

    Returns:
        List of camera dicts with viewmat, position
    """
    if center is None:
        center = np.array([0, 0, 0])

    cameras = []
    phi = np.radians(90 - elevation)  # Convert to polar angle

    for i in range(num_views):
        theta = 2 * np.pi * i / num_views

        # Camera position
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        position = np.array([x, y, z])

        # Look at center
        forward = center - position
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

        cameras.append({
            "viewmat": viewmat,
            "position": position,
            "theta": theta,
            "phi": phi,
        })

    return cameras


def render_novel_views(
    model,
    input_image: torch.Tensor,
    cameras: List[Dict],
    K: torch.Tensor,
    image_size: int = 378,
    device: torch.device = None,
) -> List[np.ndarray]:
    """
    Render novel views from a single input image.

    Args:
        model: MoReMouse model
        input_image: [C, H, W] or [1, C, H, W] input image
        cameras: List of camera dicts from create_rotation_cameras
        K: [3, 3] camera intrinsics
        image_size: Output image size
        device: Torch device

    Returns:
        List of [H, W, 3] RGB images (numpy, uint8)
    """
    if device is None:
        device = next(model.parameters()).device

    if input_image.dim() == 3:
        input_image = input_image.unsqueeze(0)

    input_image = input_image.to(device)
    K = K.to(device).unsqueeze(0) if K.dim() == 2 else K.to(device)

    model.eval()
    rendered_views = []

    with torch.no_grad():
        for cam in cameras:
            viewmat = torch.from_numpy(cam["viewmat"]).float().to(device).unsqueeze(0)

            outputs = model(input_image, viewmats=viewmat, Ks=K)
            rgb = outputs["rgb"][0]  # [H, W, 3]

            # Convert to numpy uint8
            rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
            rendered_views.append(rgb_np)

    return rendered_views


def create_rotation_video(
    views: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = "mp4v",
):
    """
    Create rotation video from rendered views.

    Args:
        views: List of [H, W, 3] RGB images
        output_path: Output video path (.mp4)
        fps: Frames per second
        codec: Video codec
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    H, W = views[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for view in views:
        # RGB to BGR for OpenCV
        frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()
    print(f"Saved video to {output_path}")


def create_comparison_grid(
    images: List[np.ndarray],
    titles: List[str] = None,
    ncols: int = 4,
    padding: int = 10,
    title_height: int = 30,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Create comparison grid of images.

    Args:
        images: List of [H, W, 3] images
        titles: Optional titles for each image
        ncols: Number of columns
        padding: Padding between images
        title_height: Height for title text
        font_scale: Font scale for titles

    Returns:
        [H, W, 3] grid image
    """
    n = len(images)
    nrows = (n + ncols - 1) // ncols

    # Assume all images same size
    h, w = images[0].shape[:2]

    # Create canvas
    grid_h = nrows * (h + title_height + padding) + padding
    grid_w = ncols * (w + padding) + padding
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255

    for i, img in enumerate(images):
        row = i // ncols
        col = i % ncols

        y = row * (h + title_height + padding) + padding + title_height
        x = col * (w + padding) + padding

        canvas[y:y+h, x:x+w] = img

        # Add title
        if titles and i < len(titles):
            cv2.putText(
                canvas, titles[i],
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1
            )

    return canvas


def visualize_depth(
    depth: torch.Tensor,
    colormap: int = cv2.COLORMAP_VIRIDIS,
    min_val: float = None,
    max_val: float = None,
) -> np.ndarray:
    """
    Visualize depth map with colormap.

    Args:
        depth: [H, W] depth map
        colormap: OpenCV colormap
        min_val: Minimum depth value (for normalization)
        max_val: Maximum depth value

    Returns:
        [H, W, 3] colored depth visualization (uint8)
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()

    # Normalize
    if min_val is None:
        min_val = depth[depth > 0].min() if (depth > 0).any() else 0
    if max_val is None:
        max_val = depth.max()

    depth_norm = (depth - min_val) / (max_val - min_val + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)

    # Apply colormap
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, colormap)

    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def create_inference_visualization(
    input_image: np.ndarray,
    novel_views: List[np.ndarray],
    depth_map: np.ndarray = None,
    normal_map: np.ndarray = None,
    output_path: Union[str, Path] = None,
) -> np.ndarray:
    """
    Create comprehensive inference visualization.

    Layout:
        +-------------------+-------------------+
        |      Input        |    Novel View 0   |
        +-------------------+-------------------+
        |  Novel View 1     |    Novel View 2   |
        +-------------------+-------------------+
        |     Depth         |     Normal        |
        +-------------------+-------------------+

    Args:
        input_image: [H, W, 3] input image
        novel_views: List of [H, W, 3] novel view images
        depth_map: [H, W] depth map (optional)
        normal_map: [H, W, 3] normal map (optional)
        output_path: Save path (optional)

    Returns:
        [H, W, 3] visualization image
    """
    images = [input_image] + novel_views[:3]
    titles = ["Input", "View 0°", "View 90°", "View 180°"]

    if depth_map is not None:
        images.append(visualize_depth(depth_map))
        titles.append("Depth")

    if normal_map is not None:
        if isinstance(normal_map, torch.Tensor):
            normal_map = normal_map.cpu().numpy()
        # Map [-1, 1] to [0, 255]
        normal_vis = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        images.append(normal_vis)
        titles.append("Normal")

    grid = create_comparison_grid(images, titles, ncols=2)

    if output_path:
        cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"Saved visualization to {output_path}")

    return grid


class VideoRenderer:
    """
    Video renderer for frame sequences.

    Supports rendering sequences of poses with novel views.
    """

    def __init__(
        self,
        model,
        image_size: int = 378,
        device: torch.device = None,
    ):
        self.model = model
        self.image_size = image_size
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)
        self.model.eval()

    def render_sequence(
        self,
        input_images: List[torch.Tensor],
        viewmats: List[torch.Tensor],
        K: torch.Tensor,
        output_dir: Union[str, Path],
        save_video: bool = True,
        fps: int = 30,
    ) -> List[np.ndarray]:
        """
        Render a sequence of frames.

        Args:
            input_images: List of [C, H, W] input images
            viewmats: List of [4, 4] view matrices (one per frame)
            K: [3, 3] camera intrinsics
            output_dir: Output directory
            save_video: Whether to save as video
            fps: Video FPS

        Returns:
            List of rendered frames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        K = K.to(self.device).unsqueeze(0) if K.dim() == 2 else K.to(self.device)

        frames = []
        with torch.no_grad():
            for i, (img, viewmat) in enumerate(zip(input_images, viewmats)):
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                img = img.to(self.device)
                viewmat = viewmat.to(self.device).unsqueeze(0)

                outputs = self.model(img, viewmats=viewmat, Ks=K)
                rgb = outputs["rgb"][0].cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)

                frames.append(rgb)

                # Save individual frame
                cv2.imwrite(
                    str(output_dir / f"frame_{i:04d}.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )

        # Create video
        if save_video and frames:
            create_rotation_video(
                frames,
                output_dir / "sequence.mp4",
                fps=fps,
            )

        return frames

    def render_rotation(
        self,
        input_image: torch.Tensor,
        K: torch.Tensor,
        num_views: int = 72,
        elevation: float = 30.0,
        output_path: Union[str, Path] = None,
        fps: int = 30,
    ) -> List[np.ndarray]:
        """
        Render 360-degree rotation from single input.

        Args:
            input_image: [C, H, W] input image
            K: [3, 3] camera intrinsics
            num_views: Number of rotation frames
            elevation: Camera elevation angle
            output_path: Video output path
            fps: Video FPS

        Returns:
            List of rendered frames
        """
        cameras = create_rotation_cameras(num_views, elevation=elevation)
        frames = render_novel_views(
            self.model, input_image, cameras, K,
            image_size=self.image_size, device=self.device
        )

        if output_path:
            create_rotation_video(frames, output_path, fps=fps)

        return frames
