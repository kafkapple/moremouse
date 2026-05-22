"""Train the anchor-based AGAM proxy and generate visual diagnostics."""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import torch
from loguru import logger
from omegaconf import OmegaConf

from moremouse.data.agam import (
    build_agam_template,
    build_target_avatar,
    move_torch_avatar,
    stack_torch_avatars,
)
from moremouse.models.agam import AgamAvatarModel
from moremouse.training.agam_losses import gaussian_avatar_loss
from moremouse.training.agam_pipeline import build_batch, render_dense_preview, render_eval, scope_note
from moremouse.training.reproducibility import seed_everything


def main() -> None:
    """Run a compact AGAM training loop with preview outputs."""
    default_cfg = OmegaConf.load("configs/default.yaml")
    exp_cfg = OmegaConf.load("configs/experiments/moremouse_author_level.yaml")
    dataset_cfg = OmegaConf.load(exp_cfg.dataset_config).dataset
    seed_everything(int(default_cfg.seed))
    torch.manual_seed(int(default_cfg.seed))

    exp = exp_cfg.experiment
    output_dir = Path(default_cfg.paths.gpu_result_root) / exp.output_subdir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "frames").mkdir(parents=True)
    (output_dir / "grids").mkdir(parents=True)
    (output_dir / "dense_views").mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = build_batch(dataset_cfg, exp, output_dir)
    template = build_agam_template(batch["canonical_mesh"], int(exp.anchor_count), int(exp.geodesic_anchor_count))
    target_batch = stack_torch_avatars([build_target_avatar(mesh, template) for mesh in batch["frame_meshes"]])
    target_batch = move_torch_avatar(target_batch, device)

    model = AgamAvatarModel(
        template=template,
        hidden_dim=int(exp.hidden_dim),
        use_dino=bool(exp.use_dino),
        transformer_layers=int(exp.transformer_layers),
        transformer_heads=int(exp.transformer_heads),
    ).to(device)
    images = torch.from_numpy(batch["images"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp.learning_rate))

    logs: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(int(exp.epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(images)
        losses = gaussian_avatar_loss(prediction.avatar, target_batch)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite AGAM loss")
        total.backward()
        optimizer.step()
        loss_value = float(total.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % int(exp.train_log_interval) == 0:
            logs.append({
                "epoch": epoch + 1,
                "total": loss_value,
                "center": float(losses["center"].detach().cpu()),
                "color": float(losses["color"].detach().cpu()),
                "scale": float(losses["scale"].detach().cpu()),
                "opacity": float(losses["opacity"].detach().cpu()),
                "rotation": float(losses["rotation"].detach().cpu()),
            })
            logger.info("epoch {} total {:.6f}", epoch + 1, loss_value)

    if best_state is None:
        raise RuntimeError("AGAM training never recorded a finite state")
    model.load_state_dict(best_state)

    eval_rows = render_eval(batch, exp, model, target_batch, device, output_dir, dataset_cfg)
    dense_preview = render_dense_preview(batch, exp, model, target_batch, device, output_dir)
    report = {
        "device": str(device),
        "template": {
            "anchors": int(template.anchor_indices.shape[0]),
            "geodesic_anchors": int(template.geodesic_anchor_indices.shape[0]),
        },
        "best_total_loss": best_loss,
        "logs": logs,
        "eval_rows": eval_rows,
        "dense_preview": dense_preview,
        "scope": scope_note(),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote AGAM report to {}", output_dir / "report.json")


if __name__ == "__main__":
    main()
