"""Train single-view RGB to 3D mesh PCA reconstruction MVP."""

from pathlib import Path
import json
import pickle
import shutil

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from moremouse.data.video_frames import extract_frame
from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import binary_iou, project_vertices, rasterize_projected_silhouette
from moremouse.models.single_view_mesh import SingleViewMeshMvp
from moremouse.training.reproducibility import seed_everything
from moremouse.visualization.overlay import load_rgb, overlay_mask
from scripts.audit_camera_projection import overlay_mesh_silhouette


def main() -> None:
    """Run the single-view mesh MVP training and visualization."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    default_cfg = OmegaConf.load("configs/default.yaml")
    seed_everything(int(default_cfg.seed))
    torch.manual_seed(int(default_cfg.seed))
    exp = cfg.single_view_mesh_mvp
    output_dir = Path(exp.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "frames").mkdir(parents=True)
    (output_dir / "grids").mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_training_data(cfg, exp, output_dir)
    model, basis = train_model(data, exp, device)
    rows = render_eval_grids(cfg, exp, data, model, basis, output_dir, device)
    report = {"device": str(device), "rows": rows, "scope": scope_note()}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote single-view mesh MVP report to {}", output_dir / "report.json")


def build_training_data(cfg: dict, exp: dict, output_dir: Path) -> dict:
    """Load input images and target best-source meshes."""
    report = json.loads(Path(exp.mesh_source_report).read_text(encoding="utf-8"))
    best_rows = [report["best_by_frame"][frame_id] for frame_id in sorted(report["best_by_frame"], key=int)]
    images, vertices, frames = [], [], []
    for row in best_rows:
        frame_id = int(row["frame_id"])
        frames.append(frame_id)
        images.append(load_input_image(cfg, exp, frame_id, output_dir / "frames"))
        vertices.append(load_obj_mesh(Path(row["obj_path"])).vertices.astype(np.float32))
    return {"images": np.stack(images), "vertices": np.stack(vertices), "frames": frames, "best_rows": best_rows}


def load_input_image(cfg: dict, exp: dict, frame_id: int, frame_dir: Path) -> np.ndarray:
    """Extract and normalize one input RGB image."""
    root = Path(cfg.root)
    view = int(exp.input_view)
    rgb_path = frame_dir / f"input_v{view}_f{frame_id:06d}.png"
    extract_frame(root / cfg.source.rgb_videos / f"{view}.mp4", frame_id, rgb_path)
    image = Image.open(rgb_path).convert("RGB").resize(tuple(int(v) for v in exp.image_size))
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.iinfo(np.uint8).max


def pca_fit(vertices: np.ndarray, components: int) -> dict:
    """Fit PCA over flattened vertex arrays."""
    flat = vertices.reshape(vertices.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:components].astype(np.float32)
    coeffs = centered @ basis.T
    coeff_mean = coeffs.mean(axis=0, keepdims=True)
    coeff_std = coeffs.std(axis=0, keepdims=True) + 1e-6
    coeffs = (coeffs - coeff_mean) / coeff_std
    return {"mean": mean.astype(np.float32), "basis": basis, "coeffs": coeffs.astype(np.float32),
            "coeff_mean": coeff_mean.astype(np.float32), "coeff_std": coeff_std.astype(np.float32)}


def train_model(data: dict, exp: dict, device: torch.device) -> tuple[SingleViewMeshMvp, dict]:
    """Train image encoder to predict mesh PCA coefficients."""
    basis = pca_fit(data["vertices"], int(exp.pca_components))
    images = torch.from_numpy(data["images"]).to(device)
    targets = torch.from_numpy(basis["coeffs"]).to(device)
    model = SingleViewMeshMvp(int(exp.pca_components), int(exp.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp.learning_rate))
    for epoch in range(int(exp.epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(images), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite single-view mesh loss")
        loss.backward()
        optimizer.step()
        if (epoch + 1) % int(exp.train_log_interval) == 0:
            logger.info("epoch {} coeff_mse {:.6f}", epoch + 1, float(loss.detach().cpu()))
    return model, basis


def render_eval_grids(
    cfg: dict,
    exp: dict,
    data: dict,
    model: SingleViewMeshMvp,
    basis: dict,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Render predicted meshes for configured evaluation frames."""
    cameras = pickle.load((Path(cfg.root) / cfg.source.cameras.primary).open("rb"))
    rows = []
    frame_to_index = {frame: index for index, frame in enumerate(data["frames"])}
    for frame_id in [int(frame) for frame in exp.eval_frames]:
        index = frame_to_index[frame_id]
        image = torch.from_numpy(data["images"][index:index + 1]).to(device)
        with torch.no_grad():
            coeff = model(image).cpu().numpy()
        vertices = reconstruct_vertices(coeff, basis, data["vertices"].shape[1])
        row = render_frame(cfg, exp, frame_id, vertices, cameras, output_dir)
        row["target_source"] = data["best_rows"][index]["source"]
        rows.append(row)
    return rows


def reconstruct_vertices(coeff: np.ndarray, basis: dict, vertex_count: int) -> np.ndarray:
    """Reconstruct vertices from predicted PCA coefficients."""
    coeff = coeff * basis["coeff_std"] + basis["coeff_mean"]
    flat = coeff @ basis["basis"] + basis["mean"]
    return flat.reshape(vertex_count, 3).astype(np.float32)


def render_frame(cfg: dict, exp: dict, frame_id: int, vertices: np.ndarray, cameras: list[dict], output_dir: Path) -> dict:
    """Render one predicted mesh into six camera views."""
    root = Path(cfg.root)
    panels, scores = [], []
    faces = load_obj_mesh(Path(json.loads(Path(exp.mesh_source_report).read_text())["best_by_frame"][str(frame_id)]["obj_path"])).faces
    panels.append(input_panel(cfg, exp, frame_id, output_dir / "frames"))
    for view in [int(view) for view in cfg.views]:
        rgb_path, mask_path = extract_view_frames(cfg, frame_id, view, output_dir / "frames")
        rgb = load_rgb(rgb_path)
        mask = Image.open(mask_path).convert("L")
        uv, _ = project_vertices(vertices, cameras[view])
        silhouette = rasterize_projected_silhouette(uv, faces, mask.size)
        score = binary_iou(np.asarray(mask) > int(exp.mask_threshold), silhouette)
        scores.append(score)
        panels.append(render_panel(rgb, mask, silhouette, int(exp.mask_threshold), frame_id, view, score, exp))
    grid_path = output_dir / "grids" / f"single_view_mesh_mvp_frame_{frame_id:06d}.png"
    save_grid(panels, grid_path)
    return {"frame_id": frame_id, "mean_silhouette_iou": float(np.mean(scores)), "grid_path": str(grid_path)}


def extract_view_frames(cfg: dict, frame_id: int, view: int, frame_dir: Path) -> tuple[Path, Path]:
    """Extract RGB and mask frames."""
    root = Path(cfg.root)
    rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
    mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
    extract_frame(root / cfg.source.rgb_videos / f"{view}.mp4", frame_id, rgb_path)
    extract_frame(root / cfg.source.segmentation_masks / f"{view}.mp4", frame_id, mask_path)
    return rgb_path, mask_path


def input_panel(cfg: dict, exp: dict, frame_id: int, frame_dir: Path) -> Image.Image:
    """Create the single-view input panel."""
    rgb_path, mask_path = extract_view_frames(cfg, frame_id, int(exp.input_view), frame_dir)
    image = overlay_mask(load_rgb(rgb_path), Image.open(mask_path).convert("L"), threshold=int(exp.mask_threshold))
    return label(image, f"single RGB input v{int(exp.input_view)} f{frame_id:06d}", exp)


def render_panel(rgb: Image.Image, mask: Image.Image, silhouette: np.ndarray, threshold: int,
                 frame_id: int, view: int, score: float, exp: dict) -> Image.Image:
    """Create one predicted render panel."""
    image = overlay_mesh_silhouette(overlay_mask(rgb, mask, threshold=threshold), silhouette)
    return label(image, f"pred render v{view} f{frame_id:06d} IoU={score:.3f}", exp)


def label(image: Image.Image, text: str, exp: dict) -> Image.Image:
    """Resize and label a panel."""
    output = image.resize(tuple(int(value) for value in exp.panel_size)).convert("RGB")
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 34), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return output


def save_grid(panels: list[Image.Image], output_path: Path) -> None:
    """Save seven panels as a 4x2 grid."""
    width, height = panels[0].size
    output = Image.new("RGB", (width * 4, height * 2), (18, 18, 18))
    for index, panel in enumerate(panels):
        output.paste(panel, ((index % 4) * width, (index // 4) * height))
    output.save(output_path)


def scope_note() -> str:
    """Return scope note for this MVP."""
    return "Single-view RGB encoder predicts PCA coefficients of best-source MAMMAL mesh vertices."

if __name__ == "__main__":
    main()
