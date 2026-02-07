#!/usr/bin/env python
"""
MoReMouse Comprehensive Evaluation & Report Generation

Generates complete evaluation report including:
1. Training progress analysis (loss curves, metrics over time)
2. Per-stage evaluation (NeRF vs DMTet comparison)
3. Image quality metrics (PSNR, SSIM, LPIPS)
4. 3D reconstruction quality (mesh, depth, normal)
5. Geodesic embedding visualization
6. 2D/3D rendering comparisons
7. Sample evolution throughout training

Usage:
    python scripts/comprehensive_eval.py \
        --checkpoint checkpoints/best.pt \
        --log-dir outputs/moremouse_default/2025-12-14_03-13-15 \
        --output-dir results/comprehensive_report
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Set matplotlib backend for non-interactive use
plt.switch_backend('Agg')


class TrainingAnalyzer:
    """Analyze training logs and generate progress plots."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / "train.log"
        self.metrics = self._parse_logs()

    def _parse_logs(self) -> Dict[str, List]:
        """Parse training log file."""
        metrics = {
            "epoch": [],
            "loss": [],
            "stage": [],  # 'nerf' or 'dmtet'
            "timestamp": [],
        }

        if not self.log_file.exists():
            print(f"Warning: Log file not found: {self.log_file}")
            return metrics

        # Pattern: [2025-12-14 15:34:07,208][__main__][INFO] - Epoch 103 - Loss: 0.2720
        pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\].*Epoch (\d+) - Loss: ([\d.]+)'

        with open(self.log_file, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    timestamp_str, epoch, loss = match.groups()
                    epoch = int(epoch)
                    loss = float(loss)

                    metrics["epoch"].append(epoch)
                    metrics["loss"].append(loss)
                    metrics["stage"].append("nerf" if epoch < 60 else "dmtet")
                    metrics["timestamp"].append(timestamp_str)

        return metrics

    def plot_loss_curve(self, output_path: Path) -> None:
        """Plot training loss curve with stage separation."""
        if not self.metrics["epoch"]:
            print("No training data to plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = np.array(self.metrics["epoch"])
        losses = np.array(self.metrics["loss"])

        # Full training curve
        ax = axes[0]
        nerf_mask = epochs < 60
        dmtet_mask = epochs >= 60

        ax.plot(epochs[nerf_mask], losses[nerf_mask], 'b-', label='NeRF Stage', linewidth=2)
        ax.plot(epochs[dmtet_mask], losses[dmtet_mask], 'r-', label='DMTet Stage', linewidth=2)
        ax.axvline(x=60, color='gray', linestyle='--', alpha=0.7, label='Stage Transition')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Full Training Progress', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # NeRF stage detail
        ax = axes[1]
        if nerf_mask.sum() > 0:
            ax.plot(epochs[nerf_mask], losses[nerf_mask], 'b-', linewidth=2)
            ax.fill_between(epochs[nerf_mask], losses[nerf_mask], alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('NeRF Stage (Epochs 0-59)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # DMTet stage detail
        ax = axes[2]
        if dmtet_mask.sum() > 0:
            ax.plot(epochs[dmtet_mask], losses[dmtet_mask], 'r-', linewidth=2)
            ax.fill_between(epochs[dmtet_mask], losses[dmtet_mask], alpha=0.3)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('DMTet Stage (Epochs 60-159)', fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved loss curve: {output_path}")

    def plot_training_time(self, output_path: Path) -> None:
        """Plot training time per epoch."""
        if len(self.metrics["timestamp"]) < 2:
            return

        times = []
        for i, ts in enumerate(self.metrics["timestamp"]):
            times.append(datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"))

        durations = []
        for i in range(1, len(times)):
            delta = (times[i] - times[i-1]).total_seconds() / 60  # minutes
            durations.append(delta)

        fig, ax = plt.subplots(figsize=(12, 4))
        epochs = self.metrics["epoch"][1:]

        ax.bar(epochs, durations, color=['blue' if e < 60 else 'red' for e in epochs], alpha=0.7)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Time (minutes)', fontsize=12)
        ax.set_title('Training Time per Epoch', fontsize=14)
        ax.axhline(y=np.mean(durations), color='green', linestyle='--', label=f'Mean: {np.mean(durations):.1f} min')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training time plot: {output_path}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics."""
        if not self.metrics["epoch"]:
            return {}

        epochs = np.array(self.metrics["epoch"])
        losses = np.array(self.metrics["loss"])

        nerf_losses = losses[epochs < 60]
        dmtet_losses = losses[epochs >= 60]

        return {
            "total_epochs": int(max(epochs)) + 1,
            "nerf_epochs": int(min(60, max(epochs) + 1)),
            "dmtet_epochs": int(max(0, max(epochs) - 59)),
            "final_loss": float(losses[-1]) if len(losses) > 0 else None,
            "nerf_final_loss": float(nerf_losses[-1]) if len(nerf_losses) > 0 else None,
            "nerf_min_loss": float(nerf_losses.min()) if len(nerf_losses) > 0 else None,
            "dmtet_final_loss": float(dmtet_losses[-1]) if len(dmtet_losses) > 0 else None,
            "dmtet_min_loss": float(dmtet_losses.min()) if len(dmtet_losses) > 0 else None,
            "loss_reduction_nerf": float((nerf_losses[0] - nerf_losses[-1]) / nerf_losses[0] * 100) if len(nerf_losses) > 1 else None,
            "loss_reduction_dmtet": float((dmtet_losses[0] - dmtet_losses[-1]) / dmtet_losses[0] * 100) if len(dmtet_losses) > 1 else None,
        }


class ModelEvaluator:
    """Comprehensive model evaluation."""

    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.config = None
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        from src.models import MoReMouse

        print(f"Loading model from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.config = checkpoint.get("config", {})
        model_cfg = self.config.get("model", {})

        # Build model config
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

        self.model = MoReMouse(
            encoder_config=encoder_config,
            triplane_config=triplane_config,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Store training info
        self.training_info = {
            "epoch": checkpoint.get("epoch", "unknown"),
            "global_step": checkpoint.get("global_step", "unknown"),
            "best_metric": checkpoint.get("best_metric", "unknown"),
        }

    def compute_image_metrics(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        pred_mask: torch.Tensor = None,
        target_mask: torch.Tensor = None,
    ) -> Dict[str, float]:
        """Compute image quality metrics."""
        metrics = {}

        # Ensure same shape
        if pred_rgb.shape != target_rgb.shape:
            target_rgb = F.interpolate(
                target_rgb.permute(0, 3, 1, 2),
                size=pred_rgb.shape[1:3],
                mode='bilinear',
            ).permute(0, 2, 3, 1)

        # PSNR
        mse = F.mse_loss(pred_rgb, target_rgb)
        psnr = 10 * torch.log10(1.0 / mse)
        metrics["psnr"] = psnr.item()

        # SSIM (simplified)
        from src.losses import SSIMLoss
        try:
            ssim_loss = SSIMLoss()
            ssim_val = 1.0 - ssim_loss(
                pred_rgb.permute(0, 3, 1, 2),
                target_rgb.permute(0, 3, 1, 2)
            )
            metrics["ssim"] = ssim_val.item()
        except:
            metrics["ssim"] = None

        # LPIPS
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
            lpips_val = lpips_fn(
                pred_rgb.permute(0, 3, 1, 2) * 2 - 1,
                target_rgb.permute(0, 3, 1, 2) * 2 - 1
            )
            metrics["lpips"] = lpips_val.mean().item()
        except:
            metrics["lpips"] = None

        # IoU for masks
        if pred_mask is not None and target_mask is not None:
            pred_binary = (pred_mask > 0.5).float()
            target_binary = (target_mask > 0.5).float()
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum() - intersection
            iou = intersection / (union + 1e-8)
            metrics["iou"] = iou.item()

        return metrics

    def render_novel_views(
        self,
        input_image: torch.Tensor,
        K: torch.Tensor,
        num_views: int = 8,
        elevation: float = 30.0,
        image_size: int = 224,
    ) -> List[np.ndarray]:
        """Render novel views around the object."""
        from src.utils import create_rotation_cameras

        cameras = create_rotation_cameras(num_views, elevation=elevation)
        views = []

        with torch.no_grad():
            # Encode once
            triplane = self.model.encode_image(input_image.unsqueeze(0).to(self.device))

            for cam in cameras:
                viewmat = torch.from_numpy(cam["viewmat"]).float().to(self.device).unsqueeze(0)

                outputs = self.model.render_image(
                    triplane, viewmat, K.unsqueeze(0).to(self.device),
                    height=image_size, width=image_size
                )

                rgb = outputs["rgb"][0].cpu().numpy()
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                views.append(rgb)

        return views

    def generate_depth_normal(
        self,
        input_image: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        image_size: int = 224,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate RGB, depth, and normal maps."""
        with torch.no_grad():
            outputs = self.model(
                input_image.unsqueeze(0).to(self.device),
                viewmats=viewmat.unsqueeze(0).to(self.device),
                Ks=K.unsqueeze(0).to(self.device),
                render_size=(image_size, image_size),
            )

        rgb = outputs["rgb"][0].cpu().numpy()
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        depth = outputs["depth"][0].cpu().numpy() if "depth" in outputs else None

        # Compute normal from depth
        if depth is not None:
            normal = self._compute_normal_from_depth(depth)
        else:
            normal = None

        return rgb, depth, normal

    def _compute_normal_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """Compute normal map from depth."""
        # Sobel gradients
        dz_dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

        # Normal = normalize(-dz/dx, -dz/dy, 1)
        normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
        normal = normal / norm

        # Map to [0, 255]
        normal = ((normal + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        return normal


class ReportGenerator:
    """Generate comprehensive markdown report with visualizations."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.report_sections = []

    def add_section(self, title: str, content: str):
        """Add a section to the report."""
        self.report_sections.append({"title": title, "content": content})

    def save_figure(self, fig_or_array, name: str) -> str:
        """Save figure and return relative path."""
        path = self.figures_dir / name

        if isinstance(fig_or_array, np.ndarray):
            if fig_or_array.dtype != np.uint8:
                fig_or_array = (fig_or_array * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(path), cv2.cvtColor(fig_or_array, cv2.COLOR_RGB2BGR))
        else:
            fig_or_array.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig_or_array)

        return f"figures/{name}"

    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        name: str,
        cols: int = 4,
    ) -> str:
        """Create a grid of comparison images."""
        n = len(images)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (img, title) in enumerate(zip(images, titles)):
            r, c = i // cols, i % cols
            axes[r, c].imshow(img)
            axes[r, c].set_title(title, fontsize=10)
            axes[r, c].axis('off')

        # Hide empty axes
        for i in range(n, rows * cols):
            r, c = i // cols, i % cols
            axes[r, c].axis('off')

        plt.tight_layout()
        return self.save_figure(fig, name)

    def generate_report(self, filename: str = "report.md"):
        """Generate the final markdown report."""
        report_path = self.output_dir / filename

        with open(report_path, 'w') as f:
            f.write("# MoReMouse Comprehensive Evaluation Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Table of Contents
            f.write("## Table of Contents\n\n")
            for i, section in enumerate(self.report_sections, 1):
                anchor = section["title"].lower().replace(" ", "-").replace(".", "")
                f.write(f"{i}. [{section['title']}](#{anchor})\n")
            f.write("\n---\n\n")

            # Sections
            for section in self.report_sections:
                f.write(f"## {section['title']}\n\n")
                f.write(section['content'])
                f.write("\n\n")

        print(f"Report saved: {report_path}")
        return report_path


def create_training_evolution_visualization(
    checkpoint_dir: Path,
    evaluator: ModelEvaluator,
    test_image: torch.Tensor,
    K: torch.Tensor,
    output_dir: Path,
) -> List[str]:
    """Create visualization of training evolution."""
    # Find periodic checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pt"))

    if not checkpoints:
        print("No periodic checkpoints found")
        return []

    images = []
    labels = []

    for ckpt_path in checkpoints[:8]:  # Limit to 8 checkpoints
        try:
            # Load this checkpoint
            checkpoint = torch.load(ckpt_path, map_location=evaluator.device)
            evaluator.model.load_state_dict(checkpoint["model_state_dict"])

            step = checkpoint.get("global_step", 0)
            epoch = checkpoint.get("epoch", 0)

            # Render
            with torch.no_grad():
                outputs = evaluator.model(
                    test_image.unsqueeze(0).to(evaluator.device),
                    viewmats=torch.eye(4).unsqueeze(0).to(evaluator.device),
                    Ks=K.unsqueeze(0).to(evaluator.device),
                )
                rgb = outputs["rgb"][0].cpu().numpy()
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                images.append(rgb)
                labels.append(f"Epoch {epoch}\nStep {step}")
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    return images, labels


def main():
    parser = argparse.ArgumentParser(description="Comprehensive MoReMouse Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/comprehensive_report")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test-image", type=str, default=None, help="Test image for visualization")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    # Initialize report generator
    report = ReportGenerator(output_dir)

    # =============================================
    # 1. Training Progress Analysis
    # =============================================
    print("\n=== Analyzing Training Progress ===")
    analyzer = TrainingAnalyzer(Path(args.log_dir))

    # Generate loss plots
    loss_curve_path = output_dir / "figures" / "loss_curves.png"
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.plot_loss_curve(loss_curve_path)

    time_plot_path = output_dir / "figures" / "training_time.png"
    analyzer.plot_training_time(time_plot_path)

    # Get summary stats
    train_stats = analyzer.get_summary_stats()

    # Add to report
    content = f"""
### Training Summary

| Metric | Value |
|--------|-------|
| Total Epochs | {train_stats.get('total_epochs', 'N/A')} |
| NeRF Epochs | {train_stats.get('nerf_epochs', 'N/A')} |
| DMTet Epochs | {train_stats.get('dmtet_epochs', 'N/A')} |
| Final Loss | {train_stats.get('final_loss', 'N/A'):.4f if train_stats.get('final_loss') else 'N/A'} |

### Stage-wise Performance

| Stage | Final Loss | Min Loss | Loss Reduction |
|-------|------------|----------|----------------|
| NeRF | {train_stats.get('nerf_final_loss', 'N/A'):.4f if train_stats.get('nerf_final_loss') else 'N/A'} | {train_stats.get('nerf_min_loss', 'N/A'):.4f if train_stats.get('nerf_min_loss') else 'N/A'} | {train_stats.get('loss_reduction_nerf', 'N/A'):.1f}% if train_stats.get('loss_reduction_nerf') else 'N/A' |
| DMTet | {train_stats.get('dmtet_final_loss', 'N/A'):.4f if train_stats.get('dmtet_final_loss') else 'N/A'} | {train_stats.get('dmtet_min_loss', 'N/A'):.4f if train_stats.get('dmtet_min_loss') else 'N/A'} | {train_stats.get('loss_reduction_dmtet', 'N/A'):.1f}% if train_stats.get('loss_reduction_dmtet') else 'N/A' |

### Loss Curves

![Loss Curves](figures/loss_curves.png)

### Training Time per Epoch

![Training Time](figures/training_time.png)
"""
    report.add_section("1. Training Progress Analysis", content)

    # =============================================
    # 2. Model Evaluation
    # =============================================
    print("\n=== Loading Model ===")
    evaluator = ModelEvaluator(Path(args.checkpoint), device)

    content = f"""
### Model Configuration

| Parameter | Value |
|-----------|-------|
| Checkpoint | `{args.checkpoint}` |
| Epoch | {evaluator.training_info.get('epoch', 'N/A')} |
| Global Step | {evaluator.training_info.get('global_step', 'N/A')} |
| Best Metric | {evaluator.training_info.get('best_metric', 'N/A')} |
"""
    report.add_section("2. Model Information", content)

    # =============================================
    # 3. Visual Quality Evaluation
    # =============================================
    if args.test_image and Path(args.test_image).exists():
        print("\n=== Generating Visualizations ===")

        # Load test image
        img = cv2.imread(args.test_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # [C, H, W]

        # Default intrinsics
        fx = fy = 224 / (2 * np.tan(np.radians(30) / 2))
        cx = cy = 112
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

        # Render novel views
        print("Rendering novel views...")
        novel_views = evaluator.render_novel_views(img_tensor, K)

        # Create comparison grid
        titles = [f"View {i}" for i in range(len(novel_views))]
        grid_path = report.create_comparison_grid(novel_views, titles, "novel_views.png")

        # Generate depth and normal
        print("Generating depth and normal maps...")
        viewmat = torch.eye(4)
        rgb, depth, normal = evaluator.generate_depth_normal(img_tensor, viewmat, K)

        # Save individual outputs
        outputs_grid = [img]  # Input
        outputs_titles = ["Input"]

        if rgb is not None:
            outputs_grid.append(rgb)
            outputs_titles.append("Rendered RGB")

        if depth is not None:
            # Normalize depth for visualization
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_vis = (plt.cm.viridis(depth_vis)[:, :, :3] * 255).astype(np.uint8)
            outputs_grid.append(depth_vis)
            outputs_titles.append("Depth")

        if normal is not None:
            outputs_grid.append(normal)
            outputs_titles.append("Normal")

        outputs_path = report.create_comparison_grid(outputs_grid, outputs_titles, "outputs.png", cols=4)

        content = f"""
### Novel View Synthesis

The model was tested with the following input image and rendered from 8 different viewpoints:

![Novel Views]({grid_path})

### RGB, Depth, and Normal Maps

![Outputs]({outputs_path})
"""
        report.add_section("3. Visual Quality Evaluation", content)
    else:
        report.add_section("3. Visual Quality Evaluation",
                          "*No test image provided. Add --test-image to generate visualizations.*")

    # =============================================
    # 4. Recommendations
    # =============================================
    content = """
### Next Steps

1. **Additional Training**: If loss hasn't converged, consider training for more epochs
2. **Hyperparameter Tuning**: Adjust learning rate, loss weights based on results
3. **Data Augmentation**: Add more training data variety for better generalization
4. **Real Data Testing**: Evaluate on real captured mouse images

### Known Limitations

- Current evaluation uses synthetic data
- 3D mesh quality depends on DMTet grid resolution
- Geodesic embedding quality depends on mesh topology
"""
    report.add_section("4. Recommendations", content)

    # Generate final report
    report.generate_report()

    # Save JSON summary
    summary = {
        "training_stats": train_stats,
        "model_info": evaluator.training_info,
        "generated_at": datetime.now().isoformat(),
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Report Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Main report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
