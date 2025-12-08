#!/usr/bin/env python
"""
MoReMouse Evaluation Script

Evaluates model on synthetic and real datasets.
Computes PSNR, SSIM, LPIPS, and IoU metrics.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --dataset real
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.models import MoReMouse
from src.data import SyntheticDataset, RealDataset, get_transforms
from src.utils import compute_metrics


def load_model(checkpoint_path: Path, device: torch.device) -> MoReMouse:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Recreate model
    model_cfg = config["model"]

    encoder_config = {
        "model_name": model_cfg["encoder"]["name"],
        "freeze": True,
        "input_size": model_cfg["encoder"]["input_size"],
    }

    triplane_config = {
        "triplane_resolution": model_cfg["triplane"]["resolution"],
        "triplane_channels": model_cfg["triplane"]["channels"],
    }

    model = MoReMouse(
        encoder_config=encoder_config,
        triplane_config=triplane_config,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_synthetic(
    model: MoReMouse,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 4,
    num_workers: int = 4,
) -> dict:
    """Evaluate on synthetic dataset."""
    print("Evaluating on synthetic dataset...")

    transform = get_transforms(mode="eval")
    dataset = SyntheticDataset(
        data_dir=data_dir,
        split="val",
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_metrics = []

    # LPIPS model
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
    except ImportError:
        lpips_fn = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Synthetic"):
            input_images = batch["input_image"].to(device)
            target_images = batch["target_images"][:, 0].to(device)
            viewmats = batch["target_viewmats"][:, 0].to(device)
            Ks = batch["K"].to(device)

            # Forward pass
            outputs = model(input_images, viewmats=viewmats, Ks=Ks)

            # Compute metrics for each sample in batch
            for i in range(input_images.shape[0]):
                pred = {
                    "rgb": outputs["rgb"][i:i+1],
                    "alpha": outputs["alpha"][i:i+1],
                }
                target = {
                    "rgb": target_images[i:i+1].permute(0, 2, 3, 1),
                    "mask": (target_images[i:i+1].sum(dim=1) > 0).float(),
                }

                metrics = compute_metrics(pred, target, lpips_fn)
                all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    return avg_metrics


def evaluate_real(
    model: MoReMouse,
    data_dir: Path,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 2,
) -> dict:
    """Evaluate on real captured dataset."""
    print("Evaluating on real dataset...")

    transform = get_transforms(mode="eval")
    dataset = RealDataset(
        data_dir=data_dir,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    all_metrics = []

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
    except ImportError:
        lpips_fn = None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Real"):
            images = batch["images"]  # [B, C, 3, H, W] - C cameras
            viewmats = batch["viewmats"]  # [B, C, 4, 4]
            Ks = batch["K"]  # [B, 3, 3]

            B, C = images.shape[:2]

            # Use first camera as input
            input_images = images[:, 0].to(device)

            # Evaluate on all other cameras
            for c in range(1, C):
                target_images = images[:, c].to(device)
                target_viewmats = viewmats[:, c].to(device)

                outputs = model(
                    input_images,
                    viewmats=target_viewmats,
                    Ks=Ks.to(device),
                )

                pred = {
                    "rgb": outputs["rgb"],
                    "alpha": outputs["alpha"],
                }
                target = {
                    "rgb": target_images.permute(0, 2, 3, 1),
                    "mask": (target_images.sum(dim=1) > 0).float(),
                }

                metrics = compute_metrics(pred, target, lpips_fn)
                all_metrics.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoReMouse")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "real", "both"])
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/evaluation.json")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(Path(args.checkpoint), device)

    results = {}

    # Evaluate
    if args.dataset in ["synthetic", "both"]:
        synthetic_dir = Path(args.data_dir) / "synthetic"
        results["synthetic"] = evaluate_synthetic(
            model, synthetic_dir, device, args.batch_size
        )
        print("\nSynthetic Results:")
        for k, v in results["synthetic"].items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    if args.dataset in ["real", "both"]:
        real_dir = Path(args.data_dir) / "real"
        results["real"] = evaluate_real(
            model, real_dir, device, args.batch_size
        )
        print("\nReal Results:")
        for k, v in results["real"].items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
