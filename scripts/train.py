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

from src.models import MoReMouse
from src.data import SyntheticDataset, get_transforms
from src.losses import MoReMouseLoss
from src.utils import setup_logging, get_logger, compute_metrics


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
            "triplane_channels": model_cfg.triplane.channels,
            "num_heads": model_cfg.transformer.num_heads,
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

        # Stage 2: DMTet (if implemented)
        self.logger.info("=== Stage 2: DMTet Training ===")
        self.model.render_mode = "dmtet"
        dmtet_epochs = self.cfg.train.stages.dmtet.epochs

        for epoch in range(dmtet_epochs):
            self.current_epoch = nerf_epochs + epoch
            self._train_epoch()

            if (epoch + 1) % self.cfg.logging.eval_freq == 0:
                self._validate()

            if self.scheduler is not None:
                self.scheduler.step()

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

            # Forward pass
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

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()

            # Gradient clipping
            if self.cfg.train.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.train.training.gradient_clip,
                )

            self.optimizer.step()

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
        """Run validation."""
        self.model.eval()
        metrics_list = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_images = batch["input_image"].to(self.device)
                target_images = batch["target_images"][:, 0].to(self.device)
                viewmats = batch["target_viewmats"][:, 0].to(self.device)
                Ks = batch["K"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_images,
                    viewmats=viewmats,
                    Ks=Ks,
                )

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

        # Log to wandb
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in avg_metrics.items()}, step=self.global_step)

        # Save best model
        if avg_metrics.get('lpips', float('inf')) < self.best_metric:
            self.best_metric = avg_metrics['lpips']
            self._save_checkpoint(is_best=True)

    def _log_step(self, losses: dict):
        """Log training step."""
        if self.use_wandb:
            log_dict = {f"train/{k}": v.item() for k, v in losses.items()}
            log_dict["train/lr"] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict, step=self.global_step)

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
