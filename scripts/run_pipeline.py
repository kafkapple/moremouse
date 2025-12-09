#!/usr/bin/env python
"""
MoReMouse Full Pipeline Script

Unified script for running all stages of the MoReMouse pipeline:
1. Train Gaussian Avatar (AGAM) on multi-view images
2. Generate synthetic training data
3. Train MoReMouse network (NeRF + DMTet)
4. Evaluate and visualize results

Usage:
    # Full pipeline
    python scripts/run_pipeline.py --stage all

    # Individual stages
    python scripts/run_pipeline.py --stage avatar --data-dir ../MAMMAL_mouse/data
    python scripts/run_pipeline.py --stage synthetic --avatar-checkpoint checkpoints/avatar/avatar_final.pt
    python scripts/run_pipeline.py --stage train --data-dir data/synthetic
    python scripts/run_pipeline.py --stage evaluate --checkpoint checkpoints/best.pt
    python scripts/run_pipeline.py --stage visualize --checkpoint checkpoints/best.pt --image test.png
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm


def train_avatar(args):
    """Stage 1: Train Gaussian Avatar on multi-view images."""
    print("\n" + "=" * 60)
    print("Stage 1: Training Gaussian Avatar (AGAM)")
    print("=" * 60)

    from src.models.mouse_body import load_mouse_model
    from src.models.gaussian_avatar import GaussianAvatar, GaussianAvatarTrainer
    from src.data import create_mammal_dataloader

    device = torch.device(args.device)

    # Load body model
    print(f"Loading mouse body model from {args.mouse_model}...")
    body_model = load_mouse_model(args.mouse_model, device=device)

    # Create avatar
    avatar = GaussianAvatar(
        body_model=body_model,
        num_gaussians_per_vertex=args.gaussians_per_vertex,
    )

    # Create trainer
    trainer = GaussianAvatarTrainer(
        avatar=avatar,
        lr=args.avatar_lr,
        device=device,
    )

    # Load data
    print(f"Loading multi-view data from {args.data_dir}...")
    dataloader = create_mammal_dataloader(
        args.data_dir,
        batch_size=1,  # Multi-view, so batch_size=1 means 1 frame with all views
        num_workers=args.num_workers,
        num_frames=args.avatar_frames,
        image_size=args.avatar_image_size,  # Paper: 800x800 for avatar
    )

    # Train
    trainer.train(
        dataloader=dataloader,
        num_iterations=args.avatar_iterations,
        checkpoint_dir=args.avatar_checkpoint_dir,
        vis_dir=args.avatar_vis_dir,
        log_freq=100,
        save_freq=args.save_freq,
        vis_freq=args.vis_freq,
    )

    print(f"\nAvatar training complete!")
    print(f"Checkpoint: {args.avatar_checkpoint_dir}/avatar_final.pt")


def generate_synthetic(args):
    """Stage 2: Generate synthetic training data using trained avatar."""
    print("\n" + "=" * 60)
    print("Stage 2: Generating Synthetic Data")
    print("=" * 60)

    from src.models.mouse_body import load_mouse_model
    from src.models.gaussian_avatar import GaussianAvatar, GaussianAvatarTrainer
    from src.models.geodesic_embedding import create_geodesic_embedding

    device = torch.device(args.device)

    # Load body model
    print(f"Loading mouse body model from {args.mouse_model}...")
    body_model = load_mouse_model(args.mouse_model, device=device)

    # Load or create avatar
    if args.avatar_checkpoint and Path(args.avatar_checkpoint).exists():
        print(f"Loading trained avatar from {args.avatar_checkpoint}...")
        trainer, _ = GaussianAvatarTrainer.from_checkpoint(
            args.avatar_checkpoint, body_model, device=device
        )
        avatar = trainer.avatar
    else:
        print("Warning: No avatar checkpoint found. Using untrained avatar (gray).")
        avatar = GaussianAvatar(body_model=body_model).to(device)

    # Create geodesic embedding
    print("Computing geodesic embedding...")
    vertices = body_model.v_template.cpu().numpy()
    faces = body_model.get_faces()
    geo_embedding = create_geodesic_embedding(
        vertices, faces,
        num_iterations=500,
        device=device,
    )

    # Generate data (reuse existing function)
    from scripts.generate_synthetic_data import generate_data

    output_dir = Path(args.synthetic_output)
    generate_data(
        output_dir=output_dir,
        mouse_model_path=Path(args.mouse_model),
        num_frames=args.num_frames,
        num_views=args.num_views,
        image_size=args.image_size,
        device=args.device,
    )

    print(f"\nSynthetic data generation complete!")
    print(f"Output: {output_dir}")


def train_moremouse(args):
    """Stage 3: Train MoReMouse network."""
    print("\n" + "=" * 60)
    print("Stage 3: Training MoReMouse Network")
    print("=" * 60)

    # Use the existing train.py with hydra
    import subprocess

    cmd = [
        "python", "scripts/train.py",
        f"experiment.name={args.experiment_name}",
        f"experiment.device={args.device}",
        f"train.stages.nerf.epochs={args.nerf_epochs}",
        f"train.stages.dmtet.epochs={args.dmtet_epochs}",
        f"data.dataloader.batch_size={args.batch_size}",
        f"logging.use_wandb={str(args.use_wandb).lower()}",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))


def evaluate(args):
    """Stage 4: Evaluate trained model."""
    print("\n" + "=" * 60)
    print("Stage 4: Evaluation")
    print("=" * 60)

    import subprocess

    cmd = [
        "python", "scripts/evaluate.py",
        f"--checkpoint={args.checkpoint}",
        f"--dataset={args.eval_dataset}",
        f"--device={args.device}",
        f"--output={args.eval_output}",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))


def visualize(args):
    """Stage 5: Generate visualizations."""
    print("\n" + "=" * 60)
    print("Stage 5: Visualization")
    print("=" * 60)

    import cv2
    from src.models import MoReMouse
    from src.utils import (
        create_rotation_cameras,
        render_novel_views,
        create_rotation_video,
        create_inference_visualization,
        compute_normal_map,
        VideoRenderer,
    )

    device = torch.device(args.device)
    output_dir = Path(args.vis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = MoReMouse()  # Default config
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Default intrinsics
    fx = fy = args.image_size / (2 * np.tan(np.radians(30) / 2))
    cx = cy = args.image_size / 2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    if args.image:
        # Single image visualization
        print(f"Processing {args.image}...")
        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.image_size, args.image_size))
        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)  # [C, H, W]

        # Create video renderer
        renderer = VideoRenderer(model, args.image_size, device)

        # Render rotation
        print("Rendering 360° rotation...")
        frames = renderer.render_rotation(
            img, K,
            num_views=72,
            elevation=30.0,
            output_path=output_dir / "rotation.mp4",
            fps=30,
        )

        # Create comparison visualization
        print("Creating visualization grid...")
        cameras = create_rotation_cameras(8, elevation=30.0)
        views = render_novel_views(model, img, cameras, K, args.image_size, device)

        # Get depth and normal from model output
        with torch.no_grad():
            viewmat = torch.from_numpy(cameras[0]["viewmat"]).float().to(device).unsqueeze(0)
            outputs = model(img.unsqueeze(0).to(device), viewmats=viewmat, Ks=K.unsqueeze(0).to(device))

            if "depth" in outputs:
                depth = outputs["depth"][0].cpu()
                normal = compute_normal_map(depth)
            else:
                depth = None
                normal = None

        input_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        create_inference_visualization(
            input_np,
            views[:4],
            depth_map=depth,
            normal_map=normal,
            output_path=output_dir / "visualization.png",
        )

        # Save individual views
        for i, view in enumerate(views):
            cv2.imwrite(
                str(output_dir / f"view_{i:02d}.png"),
                cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
            )

    print(f"\nVisualization complete!")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="MoReMouse Full Pipeline")

    # Stage selection
    parser.add_argument("--stage", type=str, default="all",
                       choices=["all", "avatar", "synthetic", "train", "evaluate", "visualize"],
                       help="Pipeline stage to run")

    # Common args
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image-size", type=int, default=378,
                       help="Image size for MoReMouse training (378 for paper)")
    parser.add_argument("--avatar-image-size", type=int, default=800,
                       help="Image size for avatar training (800 for paper)")
    parser.add_argument("--num-workers", type=int, default=4)

    # Data paths (gpu05 server)
    # Video format: /home/joon/data/markerless_mouse_1_nerf/videos_undist/
    # Image format: /home/joon/data/markerless_mouse/mouse1/
    parser.add_argument("--data-dir", type=str, default="/home/joon/data/markerless_mouse_1_nerf")
    parser.add_argument("--mouse-model", type=str, default="/home/joon/data/MAMMAL_mouse/mouse_model")

    # Avatar training
    parser.add_argument("--avatar-frames", type=int, default=800)
    parser.add_argument("--avatar-iterations", type=int, default=400000)
    parser.add_argument("--avatar-lr", type=float, default=1e-3)
    parser.add_argument("--gaussians-per-vertex", type=int, default=1)
    parser.add_argument("--avatar-checkpoint-dir", type=str, default="checkpoints/avatar")
    parser.add_argument("--avatar-vis-dir", type=str, default="outputs/avatar_vis")
    parser.add_argument("--avatar-checkpoint", type=str, default=None,
                       help="Pre-trained avatar checkpoint for synthetic data generation")

    # Synthetic data generation
    parser.add_argument("--synthetic-output", type=str, default="data/synthetic")
    parser.add_argument("--num-frames", type=int, default=1000)
    parser.add_argument("--num-views", type=int, default=64)

    # MoReMouse training
    parser.add_argument("--experiment-name", type=str, default="moremouse_exp")
    parser.add_argument("--nerf-epochs", type=int, default=60)
    parser.add_argument("--dmtet-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-wandb", action="store_true")

    # Evaluation
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--eval-dataset", type=str, default="synthetic")
    parser.add_argument("--eval-output", type=str, default="results/evaluation.json")

    # Visualization
    parser.add_argument("--image", type=str, default=None, help="Input image for visualization")
    parser.add_argument("--vis-output", type=str, default="results/visualization")

    # Checkpoint frequency
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--vis-freq", type=int, default=1000)

    args = parser.parse_args()

    # Run selected stage(s)
    if args.stage == "all":
        train_avatar(args)
        generate_synthetic(args)
        train_moremouse(args)
        evaluate(args)
    elif args.stage == "avatar":
        train_avatar(args)
    elif args.stage == "synthetic":
        generate_synthetic(args)
    elif args.stage == "train":
        train_moremouse(args)
    elif args.stage == "evaluate":
        evaluate(args)
    elif args.stage == "visualize":
        visualize(args)


if __name__ == "__main__":
    main()
