#!/usr/bin/env python
"""
MoReMouse Final Report Generator

Generates a comprehensive final training report including:
1. Pipeline overview
2. Training progress (per stage)
3. Loss curves and metrics evolution
4. Validation metrics (PSNR, SSIM, LPIPS, IoU)
5. Sample visualizations (2D/3D)
6. Model comparison (NeRF vs DMTet)
7. Recommendations and next steps

Usage:
    python scripts/generate_final_report.py \
        --checkpoint checkpoints/best.pt \
        --log-dir outputs/moremouse_default/2025-12-14_03-13-15 \
        --output-dir results/final_report \
        --test-images data/test_samples/
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

plt.switch_backend('Agg')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


class FinalReportGenerator:
    """Generate comprehensive final training report."""

    def __init__(
        self,
        checkpoint_path: Path,
        log_dir: Path,
        output_dir: Path,
        device: torch.device,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.device = device

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir = self.output_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        self.report_data = {
            "generated_at": datetime.now().isoformat(),
            "sections": [],
        }

    def parse_training_logs(self) -> Dict:
        """Parse training logs for metrics."""
        log_file = self.log_dir / "train.log"
        if not log_file.exists():
            print(f"Warning: Log file not found: {log_file}")
            return {}

        data = {
            "epochs": [],
            "losses": [],
            "stages": [],
            "timestamps": [],
            "loss_components": {},
        }

        # Main pattern
        main_pattern = r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\].*Epoch (\d+) - Loss: ([\d.]+)'

        # Component loss patterns (if available)
        component_patterns = {
            "mse": r'mse_loss: ([\d.]+)',
            "lpips": r'lpips_loss: ([\d.]+)',
            "mask": r'mask_loss: ([\d.]+)',
            "geodesic": r'geodesic_loss: ([\d.]+)',
        }

        with open(log_file, 'r') as f:
            for line in f:
                match = re.search(main_pattern, line)
                if match:
                    timestamp, epoch, loss = match.groups()
                    epoch = int(epoch)
                    loss = float(loss)

                    data["epochs"].append(epoch)
                    data["losses"].append(loss)
                    data["stages"].append("nerf" if epoch < 60 else "dmtet")
                    data["timestamps"].append(timestamp)

                    # Parse component losses
                    for name, pattern in component_patterns.items():
                        comp_match = re.search(pattern, line)
                        if comp_match:
                            if name not in data["loss_components"]:
                                data["loss_components"][name] = []
                            data["loss_components"][name].append(float(comp_match.group(1)))

        return data

    def compute_training_stats(self, log_data: Dict) -> Dict:
        """Compute training statistics."""
        if not log_data.get("epochs"):
            return {}

        epochs = np.array(log_data["epochs"])
        losses = np.array(log_data["losses"])
        timestamps = log_data["timestamps"]

        nerf_mask = epochs < 60
        dmtet_mask = epochs >= 60

        nerf_losses = losses[nerf_mask]
        dmtet_losses = losses[dmtet_mask]

        # Compute training duration
        if len(timestamps) >= 2:
            start = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            end = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
            total_duration = (end - start).total_seconds() / 3600  # hours
        else:
            total_duration = 0

        stats = {
            "total_epochs": int(max(epochs)) + 1 if len(epochs) > 0 else 0,
            "nerf_epochs": int(nerf_mask.sum()),
            "dmtet_epochs": int(dmtet_mask.sum()),
            "total_duration_hours": round(total_duration, 2),

            # Final losses
            "final_loss": float(losses[-1]) if len(losses) > 0 else None,
            "best_loss": float(losses.min()) if len(losses) > 0 else None,
            "best_epoch": int(np.argmin(losses)) if len(losses) > 0 else None,

            # NeRF stage
            "nerf_initial_loss": float(nerf_losses[0]) if len(nerf_losses) > 0 else None,
            "nerf_final_loss": float(nerf_losses[-1]) if len(nerf_losses) > 0 else None,
            "nerf_best_loss": float(nerf_losses.min()) if len(nerf_losses) > 0 else None,
            "nerf_improvement": float((nerf_losses[0] - nerf_losses[-1]) / nerf_losses[0] * 100) if len(nerf_losses) > 1 else None,

            # DMTet stage
            "dmtet_initial_loss": float(dmtet_losses[0]) if len(dmtet_losses) > 0 else None,
            "dmtet_final_loss": float(dmtet_losses[-1]) if len(dmtet_losses) > 0 else None,
            "dmtet_best_loss": float(dmtet_losses.min()) if len(dmtet_losses) > 0 else None,
            "dmtet_improvement": float((dmtet_losses[0] - dmtet_losses[-1]) / dmtet_losses[0] * 100) if len(dmtet_losses) > 1 else None,

            # Convergence
            "convergence_epoch": self._find_convergence_epoch(losses),
            "is_converged": self._check_convergence(losses),
        }

        return stats

    def _find_convergence_epoch(self, losses: np.ndarray, window: int = 10, threshold: float = 0.001) -> Optional[int]:
        """Find epoch where loss stabilized."""
        if len(losses) < window:
            return None

        for i in range(window, len(losses)):
            window_std = np.std(losses[i-window:i])
            window_mean = np.mean(losses[i-window:i])
            if window_std / window_mean < threshold:
                return i
        return None

    def _check_convergence(self, losses: np.ndarray, window: int = 20, threshold: float = 0.01) -> bool:
        """Check if training has converged."""
        if len(losses) < window:
            return False

        recent = losses[-window:]
        return np.std(recent) / np.mean(recent) < threshold

    def plot_training_curves(self, log_data: Dict, stats: Dict) -> str:
        """Create comprehensive training curve plot."""
        if not log_data.get("epochs"):
            return None

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        epochs = np.array(log_data["epochs"])
        losses = np.array(log_data["losses"])
        nerf_mask = epochs < 60
        dmtet_mask = epochs >= 60

        # 1. Full loss curve
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(epochs[nerf_mask], losses[nerf_mask], 'b-', label='NeRF', linewidth=2)
        ax1.plot(epochs[dmtet_mask], losses[dmtet_mask], 'r-', label='DMTet', linewidth=2)
        ax1.axvline(x=60, color='gray', linestyle='--', alpha=0.7, label='Stage Transition')

        if stats.get("best_epoch"):
            ax1.axvline(x=stats["best_epoch"], color='green', linestyle=':', alpha=0.7, label=f'Best (Epoch {stats["best_epoch"]})')

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Loss distribution
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(losses[nerf_mask], bins=30, alpha=0.7, label='NeRF', color='blue')
        ax2.hist(losses[dmtet_mask], bins=30, alpha=0.7, label='DMTet', color='red')
        ax2.set_xlabel('Loss')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Loss Distribution')
        ax2.legend()

        # 3. NeRF stage detail
        ax3 = fig.add_subplot(gs[1, 0])
        if nerf_mask.sum() > 0:
            ax3.plot(epochs[nerf_mask], losses[nerf_mask], 'b-', linewidth=2)
            ax3.fill_between(epochs[nerf_mask], losses[nerf_mask], alpha=0.3)
            if stats.get("nerf_initial_loss") and stats.get("nerf_final_loss"):
                ax3.axhline(y=stats["nerf_initial_loss"], color='gray', linestyle='--', alpha=0.5, label=f'Initial: {stats["nerf_initial_loss"]:.4f}')
                ax3.axhline(y=stats["nerf_final_loss"], color='blue', linestyle='--', alpha=0.5, label=f'Final: {stats["nerf_final_loss"]:.4f}')
            ax3.legend(fontsize=8)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title(f'NeRF Stage (↓{stats.get("nerf_improvement", 0):.1f}%)')
        ax3.grid(True, alpha=0.3)

        # 4. DMTet stage detail
        ax4 = fig.add_subplot(gs[1, 1])
        if dmtet_mask.sum() > 0:
            ax4.plot(epochs[dmtet_mask], losses[dmtet_mask], 'r-', linewidth=2)
            ax4.fill_between(epochs[dmtet_mask], losses[dmtet_mask], alpha=0.3)
            if stats.get("dmtet_initial_loss") and stats.get("dmtet_final_loss"):
                ax4.axhline(y=stats["dmtet_initial_loss"], color='gray', linestyle='--', alpha=0.5, label=f'Initial: {stats["dmtet_initial_loss"]:.4f}')
                ax4.axhline(y=stats["dmtet_final_loss"], color='red', linestyle='--', alpha=0.5, label=f'Final: {stats["dmtet_final_loss"]:.4f}')
            ax4.legend(fontsize=8)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title(f'DMTet Stage (↓{stats.get("dmtet_improvement", 0):.1f}%)')
        ax4.grid(True, alpha=0.3)

        # 5. Log scale
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.semilogy(epochs, losses, 'k-', linewidth=1)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss (log scale)')
        ax5.set_title('Loss (Log Scale)')
        ax5.grid(True, alpha=0.3)

        # 6. Rolling average
        ax6 = fig.add_subplot(gs[2, 0])
        window = 5
        if len(losses) >= window:
            rolling_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax6.plot(epochs[:len(rolling_avg)], rolling_avg, 'g-', linewidth=2)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Loss')
        ax6.set_title(f'Rolling Average (window={window})')
        ax6.grid(True, alpha=0.3)

        # 7. Loss change rate
        ax7 = fig.add_subplot(gs[2, 1])
        if len(losses) > 1:
            loss_diff = np.diff(losses)
            ax7.plot(epochs[1:], loss_diff, 'purple', linewidth=1, alpha=0.7)
            ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Δ Loss')
        ax7.set_title('Loss Change Rate')
        ax7.grid(True, alpha=0.3)

        # 8. Training time per epoch
        ax8 = fig.add_subplot(gs[2, 2])
        if len(log_data["timestamps"]) > 1:
            times = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in log_data["timestamps"]]
            durations = [(times[i] - times[i-1]).total_seconds() / 60 for i in range(1, len(times))]
            colors = ['blue' if e < 60 else 'red' for e in epochs[1:]]
            ax8.bar(epochs[1:len(durations)+1], durations, color=colors, alpha=0.7)
            ax8.axhline(y=np.mean(durations), color='green', linestyle='--', label=f'Mean: {np.mean(durations):.1f} min')
            ax8.legend()
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Time (min)')
        ax8.set_title('Time per Epoch')
        ax8.grid(True, alpha=0.3, axis='y')

        plt.suptitle('MoReMouse Training Analysis', fontsize=16, fontweight='bold', y=0.98)

        path = self.figures_dir / "training_curves.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves: {path}")
        return str(path.relative_to(self.output_dir))

    def generate_sample_comparisons(self, test_images_dir: Optional[Path] = None) -> List[str]:
        """Generate sample input/output comparisons."""
        if test_images_dir is None or not Path(test_images_dir).exists():
            print("No test images provided")
            return []

        from src.models import MoReMouse

        # Load model
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        model_cfg = config.get("model", {})

        encoder_cfg = model_cfg.get("encoder", {})
        model = MoReMouse(
            encoder_config={
                "model_name": encoder_cfg.get("name", "dinov2_vits14"),
                "freeze": True,
                "input_size": encoder_cfg.get("input_size", 224),
            },
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        # Find test images
        test_dir = Path(test_images_dir)
        image_files = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
        image_files = image_files[:6]  # Max 6 samples

        if not image_files:
            print("No test images found")
            return []

        saved_paths = []
        for img_path in tqdm(image_files, desc="Processing samples"):
            try:
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))

                img_tensor = torch.from_numpy(img).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)

                # Create intrinsics
                fx = fy = 224 / (2 * np.tan(np.radians(30) / 2))
                K = torch.tensor([[fx, 0, 112], [0, fy, 112], [0, 0, 1]], dtype=torch.float32)

                with torch.no_grad():
                    viewmat = torch.eye(4).unsqueeze(0).to(self.device)
                    outputs = model(
                        img_tensor.unsqueeze(0).to(self.device),
                        viewmats=viewmat,
                        Ks=K.unsqueeze(0).to(self.device),
                    )

                # Create comparison figure
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                axes[0].imshow(img)
                axes[0].set_title("Input")
                axes[0].axis('off')

                rgb = outputs["rgb"][0].cpu().numpy()
                axes[1].imshow((rgb * 255).clip(0, 255).astype(np.uint8))
                axes[1].set_title("Rendered RGB")
                axes[1].axis('off')

                if "alpha" in outputs:
                    alpha = outputs["alpha"][0].cpu().numpy()
                    axes[2].imshow(alpha, cmap='gray')
                    axes[2].set_title("Alpha")
                    axes[2].axis('off')

                if "depth" in outputs:
                    depth = outputs["depth"][0].cpu().numpy()
                    axes[3].imshow(depth, cmap='viridis')
                    axes[3].set_title("Depth")
                    axes[3].axis('off')

                plt.tight_layout()
                save_path = self.figures_dir / f"sample_{img_path.stem}.png"
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                saved_paths.append(str(save_path.relative_to(self.output_dir)))

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        return saved_paths

    def generate_markdown_report(self, log_data: Dict, stats: Dict, sample_paths: List[str], curve_path: str):
        """Generate final markdown report."""
        report_path = self.output_dir / "final_report.md"

        with open(report_path, 'w') as f:
            f.write("# MoReMouse Training Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Epochs**: {stats.get('total_epochs', 'N/A')}\n")
            f.write(f"- **Training Duration**: {stats.get('total_duration_hours', 'N/A')} hours\n")
            f.write(f"- **Final Loss**: {stats.get('final_loss', 'N/A'):.4f if stats.get('final_loss') else 'N/A'}\n")
            f.write(f"- **Best Loss**: {stats.get('best_loss', 'N/A'):.4f if stats.get('best_loss') else 'N/A'} (Epoch {stats.get('best_epoch', 'N/A')})\n")
            f.write(f"- **Converged**: {'Yes' if stats.get('is_converged') else 'No'}\n\n")

            # Training Progress
            f.write("## Training Progress\n\n")

            if curve_path:
                f.write(f"![Training Curves]({curve_path})\n\n")

            # Stage-wise Analysis
            f.write("### Stage-wise Analysis\n\n")
            f.write("| Stage | Epochs | Initial Loss | Final Loss | Improvement |\n")
            f.write("|-------|--------|--------------|------------|-------------|\n")
            f.write(f"| NeRF | {stats.get('nerf_epochs', 'N/A')} | {stats.get('nerf_initial_loss', 'N/A'):.4f if stats.get('nerf_initial_loss') else 'N/A'} | {stats.get('nerf_final_loss', 'N/A'):.4f if stats.get('nerf_final_loss') else 'N/A'} | {stats.get('nerf_improvement', 'N/A'):.1f}% if stats.get('nerf_improvement') else 'N/A' |\n")
            f.write(f"| DMTet | {stats.get('dmtet_epochs', 'N/A')} | {stats.get('dmtet_initial_loss', 'N/A'):.4f if stats.get('dmtet_initial_loss') else 'N/A'} | {stats.get('dmtet_final_loss', 'N/A'):.4f if stats.get('dmtet_final_loss') else 'N/A'} | {stats.get('dmtet_improvement', 'N/A'):.1f}% if stats.get('dmtet_improvement') else 'N/A' |\n\n")

            # Sample Visualizations
            if sample_paths:
                f.write("## Sample Visualizations\n\n")
                for i, path in enumerate(sample_paths):
                    f.write(f"### Sample {i+1}\n")
                    f.write(f"![Sample {i+1}]({path})\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if not stats.get('is_converged'):
                f.write("- **Not Converged**: Consider training for more epochs\n")
            if stats.get('dmtet_improvement', 0) < 5:
                f.write("- **Limited DMTet Improvement**: May need to tune DMTet hyperparameters\n")
            f.write("- **Validate on Real Data**: Test model on actual mouse images\n")
            f.write("- **Visualize 3D**: Use `visualize_3d.py` for novel view synthesis\n\n")

            # Technical Details
            f.write("## Technical Details\n\n")
            f.write(f"- **Checkpoint**: `{self.checkpoint_path}`\n")
            f.write(f"- **Log Directory**: `{self.log_dir}`\n")
            f.write(f"- **Device**: {self.device}\n\n")

            f.write("---\n")
            f.write("*Report generated by MoReMouse evaluation pipeline*\n")

        print(f"Report saved: {report_path}")

        # Also save JSON summary
        summary = {
            "training_stats": stats,
            "checkpoint": str(self.checkpoint_path),
            "log_dir": str(self.log_dir),
            "generated_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return report_path


def main():
    parser = argparse.ArgumentParser(description="Generate MoReMouse Final Report")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/final_report")
    parser.add_argument("--test-images", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Initialize generator
    generator = FinalReportGenerator(
        checkpoint_path=Path(args.checkpoint),
        log_dir=Path(args.log_dir),
        output_dir=Path(args.output_dir),
        device=device,
    )

    print("\n=== Parsing Training Logs ===")
    log_data = generator.parse_training_logs()

    print("\n=== Computing Statistics ===")
    stats = generator.compute_training_stats(log_data)

    print("\n=== Generating Training Curves ===")
    curve_path = generator.plot_training_curves(log_data, stats)

    print("\n=== Generating Sample Comparisons ===")
    sample_paths = []
    if args.test_images:
        sample_paths = generator.generate_sample_comparisons(Path(args.test_images))

    print("\n=== Generating Final Report ===")
    report_path = generator.generate_markdown_report(log_data, stats, sample_paths, curve_path)

    print(f"\n=== Report Complete ===")
    print(f"Output: {args.output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
