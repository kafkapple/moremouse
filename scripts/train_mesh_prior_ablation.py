"""Train mesh-free vs mesh-prior pixel MLP feasibility ablation."""

from pathlib import Path
import json
import pickle
import shutil

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from moremouse.data.video_frames import extract_frame
from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import binary_iou, project_vertices, rasterize_projected_silhouette
from moremouse.training.reproducibility import seed_everything


def main() -> None:
    """Train two small MLPs and compare held-out-view mask IoU."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    abl = cfg.feasibility_ablation
    seed_everything(int(OmegaConf.load("configs/default.yaml").seed))
    torch.manual_seed(int(OmegaConf.load("configs/default.yaml").seed))
    output_dir = Path(abl.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_dataset(cfg, abl, output_dir)
    results = {}
    for use_mesh in [False, True]:
        name = "mesh_prior" if use_mesh else "mesh_free"
        model = PixelMlp(input_dim=4 if use_mesh else 3, hidden_dim=int(abl.hidden_dim)).to(device)
        train_model(model, data["train"], use_mesh, abl, device)
        metrics = evaluate_model(model, data["eval"], use_mesh, float(abl.threshold), device)
        results[name] = metrics
        logger.info("{} eval mean IoU: {:.4f}", name, metrics["mean_iou"])
    report = {
        "device": str(device),
        "frames": [int(frame) for frame in abl.frames],
        "train_views": [int(view) for view in abl.train_views],
        "eval_views": [int(view) for view in abl.eval_views],
        "results": results,
        "best_mesh_sources": data["best_sources"],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote mesh-prior ablation report to {}", output_dir / "report.json")


def build_dataset(cfg: dict, abl: dict, output_dir: Path) -> dict:
    """Build sampled train/eval tensors from masks and best-source mesh silhouettes."""
    root = Path(cfg.root)
    cameras = pickle.load((root / cfg.source.cameras.primary).open("rb"))
    best_sources = load_best_sources(Path(abl.mesh_source_report))
    samples = {"train": [], "eval": []}
    for frame_id in [int(frame) for frame in abl.frames]:
        mesh = load_obj_mesh(Path(best_sources[str(frame_id)]["obj_path"]))
        for split_name, views_key, sample_key in [
            ("train", "train_views", "samples_per_view"),
            ("eval", "eval_views", "eval_samples_per_view"),
        ]:
            for view in [int(view) for view in abl[views_key]]:
                mask = load_mask(root, cfg, frame_id, view, output_dir / "frames")
                uv, _ = project_vertices(mesh.vertices, cameras[view])
                silhouette = rasterize_projected_silhouette(uv, mesh.faces, mask.size)
                samples[split_name].append(sample_pixels(mask, silhouette, frame_id, int(abl[sample_key])))
    return {"train": stack_samples(samples["train"]), "eval": stack_samples(samples["eval"]), "best_sources": best_sources}


def load_best_sources(report_path: Path) -> dict:
    """Load best mesh source rows from source-search report."""
    if not report_path.exists():
        raise FileNotFoundError(f"Missing mesh source report: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))["best_by_frame"]


def load_mask(root: Path, cfg: dict, frame_id: int, view: int, frame_dir: Path) -> Image.Image:
    """Extract one mask frame."""
    mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
    extract_frame(root / cfg.source.segmentation_masks / f"{view}.mp4", frame_id, mask_path)
    return Image.open(mask_path).convert("L")


def sample_pixels(mask: Image.Image, silhouette: np.ndarray, frame_id: int, count: int) -> dict:
    """Sample random pixels with coordinate, frame, mesh, and target features."""
    mask_array = (np.asarray(mask) > np.iinfo(np.uint8).max // 2).astype(np.float32)
    height, width = mask_array.shape
    ys = np.random.randint(0, height, size=count)
    xs = np.random.randint(0, width, size=count)
    xy = np.stack([xs / (width - 1), ys / (height - 1)], axis=1).astype(np.float32)
    frame_feature = np.full((count, 1), frame_id / 18000.0, dtype=np.float32)
    mesh_feature = silhouette[ys, xs].astype(np.float32)[:, None]
    target = mask_array[ys, xs][:, None]
    return {"base": np.concatenate([xy, frame_feature], axis=1), "mesh": mesh_feature, "target": target}


def stack_samples(samples: list[dict]) -> dict:
    """Stack sampled pixel dictionaries."""
    return {key: torch.from_numpy(np.concatenate([sample[key] for sample in samples], axis=0))
            for key in ["base", "mesh", "target"]}


class PixelMlp(torch.nn.Module):
    """Small pixel classifier for feasibility ablation."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize the MLP layers."""
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Predict mask logits."""
        return self.net(inputs)


def train_model(model: PixelMlp, data: dict, use_mesh: bool, abl: dict, device: torch.device) -> None:
    """Train one MLP."""
    inputs = make_inputs(data, use_mesh).to(device)
    targets = data["target"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(abl.learning_rate))
    for _ in range(int(abl.epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(inputs), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ablation loss")
        loss.backward()
        optimizer.step()


def evaluate_model(model: PixelMlp, data: dict, use_mesh: bool, threshold: float, device: torch.device) -> dict:
    """Evaluate one model on sampled held-out pixels."""
    inputs = make_inputs(data, use_mesh).to(device)
    targets = data["target"].cpu().numpy()[:, 0] > threshold
    with torch.no_grad():
        probs = torch.sigmoid(model(inputs)).cpu().numpy()[:, 0]
    predictions = probs > threshold
    return {"mean_iou": binary_iou(targets, predictions)}


def make_inputs(data: dict, use_mesh: bool) -> torch.Tensor:
    """Assemble model inputs."""
    if use_mesh:
        return torch.cat([data["base"], data["mesh"]], dim=1).float()
    return data["base"].float()


if __name__ == "__main__":
    main()
