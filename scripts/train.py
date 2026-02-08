#!/usr/bin/env python
"""
MoReMouse Training Script

Two-stage training:
1. NeRF stage (60 epochs): Volumetric rendering
2. DMTet stage (100 epochs): Surface extraction

Usage:
    python scripts/train.py
    python scripts/train.py experiment.name=my_exp
    python scripts/train.py train.stages.nerf.epochs=30
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import numpy as np
from PIL import Image
import torchvision.utils as vutils

from src.models import MoReMouse
from src.data import SyntheticDataset, get_transforms
from src.losses import MoReMouseLoss
from src.utils import setup_logging, get_logger, compute_metrics


def create_visualization_grid(
    input_img: torch.Tensor,
    pred_img: torch.Tensor,
    target_img: torch.Tensor = None,
    nrow: int = 4,
) -> np.ndarray:
    """
    Create visualization grid: Input | Prediction | (Target) | (Diff)

    Args:
        input_img: [B, C, H, W] or [B, H, W, C] input images
        pred_img: [B, H, W, C] predicted images
        target_img: [B, C, H, W] or [B, H, W, C] target images (optional)

    Returns:
        [H, W, 3] numpy array for visualization
    """
    # Ensure BCHW format
    if input_img.dim() == 4 and input_img.shape[-1] == 3:
        input_img = input_img.permute(0, 3, 1, 2)
    if pred_img.dim() == 4 and pred_img.shape[-1] == 3:
        pred_img = pred_img.permute(0, 3, 1, 2)
    if target_img is not None:
        if target_img.dim() == 4 and target_img.shape[-1] == 3:
            target_img = target_img.permute(0, 3, 1, 2)

    # Clamp to [0, 1]
    input_img = torch.clamp(input_img, 0, 1)
    pred_img = torch.clamp(pred_img, 0, 1)

    # Build grid
    images = [input_img[:nrow], pred_img[:nrow]]
    if target_img is not None:
        target_img = torch.clamp(target_img, 0, 1)
        images.append(target_img[:nrow])
        # Add difference map
        diff = torch.abs(pred_img[:nrow] - target_img[:nrow])
        diff = diff * 3  # Amplify for visibility
        images.append(torch.clamp(diff, 0, 1))

    # Concatenate vertically
    all_imgs = torch.cat(images, dim=0)
    grid = vutils.make_grid(all_imgs, nrow=nrow, padding=2, normalize=False)

    # To numpy
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    grid_np = (grid_np * 255).astype(np.uint8)

    return grid_np


def create_comprehensive_visualization(
    outputs: dict,
    input_img: torch.Tensor,
    target_img: torch.Tensor = None,
    nrow: int = 4,
) -> dict:
    """
    Create comprehensive visualization for all model outputs.

    Creates separate visualizations for:
    - RGB: Input | Prediction | Target | Diff
    - Depth: Predicted depth map (colorized)
    - Alpha: Predicted opacity/mask
    - Embedding: Geodesic embedding (as RGB)

    Args:
        outputs: Model outputs containing rgb, depth, alpha, embedding
        input_img: [B, C, H, W] or [B, H, W, C] input images
        target_img: [B, C, H, W] or [B, H, W, C] target images (optional)
        nrow: Number of images per row

    Returns:
        Dictionary with visualization arrays for each modality
    """
    vis_dict = {}
    B = input_img.shape[0]
    n = min(nrow, B)

    # 1. RGB visualization (existing logic)
    vis_dict["rgb"] = create_visualization_grid(
        input_img, outputs["rgb"], target_img, nrow=n
    )

    # 2. Depth visualization
    if "depth" in outputs and outputs["depth"] is not None:
        depth = outputs["depth"].detach()  # [B, H, W]
        if depth.dim() == 4:
            depth = depth.squeeze(1)

        # Normalize depth to [0, 1] for visualization
        depth_min = depth.min()
        depth_max = depth.max()
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)

        # Apply colormap (viridis-like)
        depth_colored = _apply_depth_colormap(depth_norm[:n])  # [n, H, W, 3]
        depth_colored = depth_colored.permute(0, 3, 1, 2)  # [n, 3, H, W]

        grid = vutils.make_grid(depth_colored, nrow=n, padding=2, normalize=False)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        vis_dict["depth"] = (grid_np * 255).astype(np.uint8)

    # 3. Alpha/Opacity visualization
    if "alpha" in outputs and outputs["alpha"] is not None:
        alpha = outputs["alpha"].detach()  # [B, H, W]
        if alpha.dim() == 4:
            alpha = alpha.squeeze(1)

        # Clamp to [0, 1]
        alpha = torch.clamp(alpha, 0, 1)

        # Convert to 3-channel grayscale
        alpha_rgb = alpha[:n].unsqueeze(1).repeat(1, 3, 1, 1)  # [n, 3, H, W]

        grid = vutils.make_grid(alpha_rgb, nrow=n, padding=2, normalize=False)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        vis_dict["alpha"] = (grid_np * 255).astype(np.uint8)

    # 4. Geodesic Embedding visualization
    if "embedding" in outputs and outputs["embedding"] is not None:
        embedding = outputs["embedding"].detach()  # [B, H, W, 3]
        if embedding.dim() == 4 and embedding.shape[-1] == 3:
            # Already in [B, H, W, 3] format
            embedding = embedding.permute(0, 3, 1, 2)  # [B, 3, H, W]

        # Normalize embedding to [0, 1] for visualization
        emb_min = embedding.min()
        emb_max = embedding.max()
        embedding_norm = (embedding - emb_min) / (emb_max - emb_min + 1e-8)
        embedding_norm = torch.clamp(embedding_norm, 0, 1)

        grid = vutils.make_grid(embedding_norm[:n], nrow=n, padding=2, normalize=False)
        grid_np = grid.cpu().numpy().transpose(1, 2, 0)
        vis_dict["embedding"] = (grid_np * 255).astype(np.uint8)

    return vis_dict


def _apply_depth_colormap(depth: torch.Tensor) -> torch.Tensor:
    """
    Apply viridis-like colormap to depth tensor.

    Args:
        depth: [B, H, W] normalized depth in [0, 1]

    Returns:
        [B, H, W, 3] colored depth
    """
    # Simple viridis-like colormap (blue -> green -> yellow)
    B, H, W = depth.shape
    device = depth.device

    # RGB values for viridis at key points
    colors = torch.tensor([
        [0.267, 0.004, 0.329],  # Dark purple (near)
        [0.282, 0.140, 0.458],  # Purple
        [0.254, 0.265, 0.530],  # Blue-purple
        [0.207, 0.372, 0.553],  # Blue
        [0.164, 0.471, 0.558],  # Blue-green
        [0.128, 0.567, 0.551],  # Teal
        [0.135, 0.659, 0.518],  # Green-teal
        [0.267, 0.749, 0.441],  # Green
        [0.478, 0.821, 0.318],  # Yellow-green
        [0.741, 0.873, 0.150],  # Yellow
        [0.993, 0.906, 0.144],  # Bright yellow (far)
    ], device=device)

    # Interpolate colors based on depth value
    depth_flat = depth.flatten()  # [B*H*W]
    idx_float = depth_flat * (len(colors) - 1)
    idx_low = idx_float.long().clamp(0, len(colors) - 2)
    idx_high = (idx_low + 1).clamp(0, len(colors) - 1)
    t = (idx_float - idx_low.float()).unsqueeze(-1)  # [B*H*W, 1]

    color_low = colors[idx_low]  # [B*H*W, 3]
    color_high = colors[idx_high]  # [B*H*W, 3]
    colored = color_low * (1 - t) + color_high * t  # [B*H*W, 3]

    return colored.reshape(B, H, W, 3)


class Trainer:
    """
    MoReMouse Trainer

    Handles two-stage training with NeRF and DMTet.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = get_logger(__name__)

        # Setup device
        self.device = torch.device(cfg.experiment.device)
        self.logger.info(f"Using device: {self.device}")

        # Setup mixed precision (AMP)
        self.use_amp = cfg.train.training.mixed_precision and self.device.type == "cuda"
        # PyTorch 2.0.x uses torch.cuda.amp, 2.1+ uses torch.amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            self.logger.info("Mixed precision training enabled (AMP)")

        # Create model
        self.model = self._create_model()
        self.model.to(self.device)

        # Create loss
        self.criterion = MoReMouseLoss(
            lambda_mse=cfg.train.loss.mse,
            lambda_lpips=cfg.train.loss.lpips,
            lambda_mask=cfg.train.loss.mask,
            lambda_smooth=cfg.train.loss.smooth_l1,
            lambda_depth=cfg.train.loss.depth,
            lambda_geo=cfg.train.loss.geodesic,
        )

        # Create optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Create dataloaders
        self.train_loader, self.val_loader = self._create_dataloaders()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')

        # Load checkpoint if resume path provided
        if hasattr(cfg, 'resume') and cfg.resume:
            self._load_checkpoint(cfg.resume)

        # Setup logging
        self._setup_logging()

    def _create_model(self) -> nn.Module:
        """Create MoReMouse model."""
        model_cfg = self.cfg.model

        encoder_config = {
            "model_name": model_cfg.encoder.name,
            "freeze": model_cfg.encoder.freeze,
            "input_size": model_cfg.encoder.input_size,
        }

        triplane_config = {
            "triplane_resolution": model_cfg.triplane.resolution,
            "internal_resolution": model_cfg.triplane.get("internal_resolution", 64),
            "triplane_channels": model_cfg.triplane.channels,
            "output_channels": model_cfg.triplane.get("output_channels", 80),
            "num_heads": model_cfg.transformer.num_heads,
            "head_dim": model_cfg.transformer.get("head_dim", 64),
            "num_layers": model_cfg.transformer.num_layers,
            "mlp_dim": model_cfg.transformer.mlp_dim,
        }

        decoder_config = {
            "hidden_dim": model_cfg.mlp_decoder.hidden_dim,
            "num_shared_layers": model_cfg.mlp_decoder.hidden_layers,
        }

        model = MoReMouse(
            encoder_config=encoder_config,
            triplane_config=triplane_config,
            decoder_config=decoder_config,
            render_mode="nerf",  # Start with NeRF
        )

        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        opt_cfg = self.cfg.train.optimizer

        if opt_cfg.name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_cfg.lr,
            )

        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        sched_cfg = self.cfg.train.scheduler

        if sched_cfg.name == "cosine_annealing":
            # Total steps
            total_epochs = (
                self.cfg.train.stages.nerf.epochs +
                self.cfg.train.stages.dmtet.epochs
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs,
                eta_min=sched_cfg.min_lr,
            )
        else:
            scheduler = None

        return scheduler

    def _create_dataloaders(self):
        """Create train and validation dataloaders."""
        data_cfg = self.cfg.data
        loader_cfg = data_cfg.dataloader

        transform = get_transforms(
            mode="train",
            image_size=self.cfg.model.encoder.input_size,
        )

        # Training dataset
        train_dataset = SyntheticDataset(
            data_dir=self.cfg.paths.data,
            split="train",
            transform=transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=True,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
        )

        # Validation dataset
        val_transform = get_transforms(mode="eval")
        val_dataset = SyntheticDataset(
            data_dir=self.cfg.paths.data,
            split="val",
            transform=val_transform,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=loader_cfg.batch_size,
            shuffle=False,
            num_workers=loader_cfg.num_workers,
            pin_memory=loader_cfg.pin_memory,
        )

        return train_loader, val_loader

    def _setup_logging(self):
        """Setup experiment logging."""
        if self.cfg.logging.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.cfg.logging.wandb_project,
                entity=self.cfg.logging.wandb_entity,
                name=self.cfg.experiment.name,
                config=OmegaConf.to_container(self.cfg),
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def train(self):
        """Run full training."""
        self.logger.info("Starting training...")

        # Stage 1: NeRF
        self.logger.info("=== Stage 1: NeRF Training ===")
        self.model.render_mode = "nerf"
        nerf_epochs = self.cfg.train.stages.nerf.epochs

        for epoch in range(nerf_epochs):
            self.current_epoch = epoch
            self._train_epoch()

            if (epoch + 1) % self.cfg.logging.eval_freq == 0:
                self._validate()

            if self.scheduler is not None:
                self.scheduler.step()

        # Stage 2: DMTet (not yet implemented)
        dmtet_epochs = self.cfg.train.stages.dmtet.epochs
        if dmtet_epochs > 0:
            self.logger.warning(
                "=== Stage 2: DMTet Training SKIPPED === "
                "DMTet renderer is not yet implemented. "
                f"Skipping {dmtet_epochs} DMTet epochs. "
                "The model will use NeRF rendering only."
            )

        self.logger.info("Training completed!")

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_images = batch["input_image"].to(self.device)
            target_images = batch["target_images"][:, 0].to(self.device)  # First target view
            viewmats = batch["target_viewmats"][:, 0].to(self.device)
            Ks = batch["K"].to(self.device)
            target_embedding = batch["embedding"].to(self.device)

            # Forward pass with AMP autocast
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(
                    input_images,
                    viewmats=viewmats,
                    Ks=Ks,
                )

                # Prepare targets
                targets = {
                    "rgb": target_images.permute(0, 2, 3, 1),  # [B, H, W, 3]
                    "mask": (target_images.sum(dim=1) > 0).float(),  # Simple mask
                    "embedding": target_embedding.permute(0, 2, 3, 1),
                }

                # Prepare predictions
                preds = {
                    "rgb": outputs["rgb"],
                    "alpha": outputs["alpha"],
                    "embedding": outputs.get("embedding"),
                }

                # Compute loss
                losses = self.criterion(preds, targets)

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()

            # Gradient clipping (unscale first for proper clipping)
            if self.cfg.train.training.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.train.training.gradient_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            for k, v in losses.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v.item())

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })

            # Logging
            self.global_step += 1
            if self.global_step % self.cfg.logging.log_freq == 0:
                self._log_step(losses)

            # Periodic visualization (every eval_freq steps)
            if self.global_step % self.cfg.logging.eval_freq == 0:
                self._save_train_visualization(input_images, outputs, target_images)

            # Save checkpoint
            if self.global_step % self.cfg.logging.save_freq == 0:
                self._save_checkpoint()

        # Epoch summary
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        self.logger.info(
            f"Epoch {self.current_epoch} - "
            f"Loss: {avg_losses['total']:.4f}"
        )

    def _validate(self):
        """Run validation with visualization."""
        self.model.eval()
        metrics_list = []
        vis_samples = None  # Store first batch for visualization

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                input_images = batch["input_image"].to(self.device)
                target_images = batch["target_images"][:, 0].to(self.device)
                viewmats = batch["target_viewmats"][:, 0].to(self.device)
                Ks = batch["K"].to(self.device)

                # Forward pass with AMP autocast
                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    outputs = self.model(
                        input_images,
                        viewmats=viewmats,
                        Ks=Ks,
                    )

                # Store first batch for visualization (all modalities)
                if batch_idx == 0:
                    vis_samples = {
                        "input": input_images.detach(),
                        "target": target_images.detach(),
                        "outputs": {k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()},
                    }

                # Compute metrics
                metrics = compute_metrics(
                    pred={"rgb": outputs["rgb"], "alpha": outputs["alpha"]},
                    target={
                        "rgb": target_images.permute(0, 2, 3, 1),
                        "mask": (target_images.sum(dim=1) > 0).float(),
                    },
                )
                metrics_list.append(metrics)

        # Average metrics
        avg_metrics = {}
        for k in metrics_list[0].keys():
            avg_metrics[k] = sum(m[k] for m in metrics_list) / len(metrics_list)

        self.logger.info(f"Validation - PSNR: {avg_metrics['psnr']:.2f}, SSIM: {avg_metrics['ssim']:.4f}")

        # Create and save comprehensive visualization (all modalities)
        if vis_samples is not None:
            vis_dir = Path(self.cfg.paths.output) / self.cfg.experiment.name / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            # Create comprehensive visualization for all modalities
            vis_dict = create_comprehensive_visualization(
                outputs=vis_samples["outputs"],
                input_img=vis_samples["input"],
                target_img=vis_samples["target"],
                nrow=min(4, vis_samples["input"].shape[0]),
            )

            # Save each modality
            for modality, vis_array in vis_dict.items():
                vis_path = vis_dir / f"val_epoch{self.current_epoch:03d}_step{self.global_step:06d}_{modality}.png"
                Image.fromarray(vis_array).save(vis_path)
            self.logger.info(f"Saved validation visualizations to {vis_dir}")

        # Log to wandb
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in avg_metrics.items()}, step=self.global_step)
            # Log visualization images for all modalities
            if vis_samples is not None:
                wandb_images = {}
                captions = {
                    "rgb": "Input | Pred | Target | Diff",
                    "depth": "Predicted Depth (viridis)",
                    "alpha": "Predicted Opacity/Mask",
                    "embedding": "Geodesic Embedding (RGB)",
                }
                for modality, vis_array in vis_dict.items():
                    wandb_images[f"val/{modality}"] = wandb.Image(
                        vis_array, caption=captions.get(modality, modality)
                    )
                wandb.log(wandb_images, step=self.global_step)

        # Save best model
        if avg_metrics.get('lpips', float('inf')) < self.best_metric:
            self.best_metric = avg_metrics['lpips']
            self._save_checkpoint(is_best=True)

    def _save_train_visualization(
        self,
        input_images: torch.Tensor,
        outputs: dict,
        target_images: torch.Tensor,
    ):
        """Save comprehensive training visualization (RGB, depth, alpha, embedding)."""
        vis_dir = Path(self.cfg.paths.output) / self.cfg.experiment.name / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Create comprehensive visualization for all modalities
        vis_dict = create_comprehensive_visualization(
            outputs=outputs,
            input_img=input_images.detach(),
            target_img=target_images.detach(),
            nrow=min(4, input_images.shape[0]),
        )

        # Save each modality
        for modality, vis_array in vis_dict.items():
            vis_path = vis_dir / f"train_step{self.global_step:06d}_{modality}.png"
            Image.fromarray(vis_array).save(vis_path)

        # Log to wandb
        if self.use_wandb:
            wandb_images = {}
            captions = {
                "rgb": "Input | Pred | Target | Diff",
                "depth": "Predicted Depth (viridis)",
                "alpha": "Predicted Opacity/Mask",
                "embedding": "Geodesic Embedding (RGB)",
            }
            for modality, vis_array in vis_dict.items():
                wandb_images[f"train/{modality}"] = wandb.Image(
                    vis_array, caption=captions.get(modality, modality)
                )
            wandb.log(wandb_images, step=self.global_step)

    def _log_step(self, losses: dict):
        """Log training step."""
        if self.use_wandb:
            log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
            log_dict["train/lr"] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict, step=self.global_step)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint for resuming training."""
        checkpoint_path = Path(checkpoint_path)

        # Auto-detect latest checkpoint if directory provided
        if checkpoint_path.is_dir():
            latest_path = checkpoint_path / "latest.pt"
            if latest_path.exists():
                checkpoint_path = latest_path
            else:
                self.logger.warning(f"No latest.pt found in {checkpoint_path}")
                return

        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return

        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore training state
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float('inf'))

        self.logger.info(
            f"Resumed from epoch {self.current_epoch}, step {self.global_step}"
        )

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.cfg.paths.checkpoints)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": OmegaConf.to_container(self.cfg),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest
        torch.save(checkpoint, checkpoint_dir / "latest.pt")

        # Save best
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        # Save periodic
        if self.global_step % (self.cfg.logging.save_freq * 5) == 0:
            torch.save(
                checkpoint,
                checkpoint_dir / f"checkpoint_{self.global_step:08d}.pt"
            )

            # Delete old checkpoints (keep_last)
            keep_last = self.cfg.checkpoint.get("keep_last", 5)
            existing_checkpoints = sorted(
                checkpoint_dir.glob("checkpoint_*.pt"),
                key=lambda p: int(p.stem.split("_")[1])
            )
            if len(existing_checkpoints) > keep_last:
                for old_ckpt in existing_checkpoints[:-keep_last]:
                    old_ckpt.unlink()
                    self.logger.info(f"Deleted old checkpoint: {old_ckpt.name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Setup logging
    setup_logging(Path(cfg.paths.logs))

    # Set seed
    torch.manual_seed(cfg.experiment.seed)

    # Create trainer and run
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
