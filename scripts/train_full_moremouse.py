"""Train the paper-aligned MoReMouse full-stack local implementation."""

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
from moremouse.geometry.geodesic_surface import farthest_point_anchors, geodesic_anchor_distances, geodesic_rgb_embedding
from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import binary_iou, project_vertices, rasterize_projected_silhouette
from moremouse.models.triplane_reconstruction import MoReMouseTriplane
from moremouse.rendering.mesh_raster import face_normals, rasterize_face_colors, vertex_to_face_colors
from moremouse.training.reproducibility import seed_everything
from moremouse.visualization.grid import save_pil_grid
from moremouse.visualization.overlay import load_rgb, overlay_mask
from scripts.audit_camera_projection import overlay_mesh_silhouette
from scripts.train_single_view_mesh_mvp import pca_fit, reconstruct_vertices


def main() -> None:
    """Train and evaluate the full-stack local MoReMouse implementation."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    seed = int(OmegaConf.load("configs/default.yaml").seed)
    seed_everything(seed)
    torch.manual_seed(seed)
    exp = cfg.full_moremouse
    output_dir = Path(exp.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "frames").mkdir(parents=True)
    (output_dir / "grids").mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_data(cfg, exp, output_dir)
    model, basis, losses = train_model(data, exp, device)
    rows = render_eval(cfg, exp, data, model, basis, output_dir, device)
    report = {"device": str(device), "losses": losses, "rows": rows, "scope": scope_note(exp)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote full MoReMouse local report to {}", output_dir / "report.json")


def build_data(cfg: dict, exp: dict, output_dir: Path) -> dict:
    """Load RGB inputs, mesh targets, and geodesic correspondence colors."""
    report = json.loads(Path(exp.mesh_source_report).read_text(encoding="utf-8"))
    best_rows = [report["best_by_frame"][key] for key in sorted(report["best_by_frame"], key=int)]
    images, vertices, frames = [], [], []
    for row in best_rows:
        frame_id = int(row["frame_id"])
        frames.append(frame_id)
        images.append(load_input(cfg, exp, frame_id, output_dir / "frames"))
        vertices.append(load_obj_mesh(Path(row["obj_path"])).vertices.astype(np.float32))
    mesh = load_obj_mesh(Path(best_rows[0]["obj_path"]))
    colors = build_geodesic_colors(mesh.vertices, mesh.faces, int(exp.geodesic_anchors))
    return {"images": np.stack(images), "vertices": np.stack(vertices), "frames": frames,
            "best_rows": best_rows, "faces": mesh.faces, "geodesic_colors": colors}


def build_geodesic_colors(vertices: np.ndarray, faces: np.ndarray, anchor_count: int) -> np.ndarray:
    """Create paper-style geodesic correspondence colors on the template mesh."""
    anchors = farthest_point_anchors(vertices, anchor_count)
    distances = geodesic_anchor_distances(vertices, faces, anchors)
    return geodesic_rgb_embedding(distances)


def load_input(cfg: dict, exp: dict, frame_id: int, frame_dir: Path) -> np.ndarray:
    """Extract and normalize one monocular RGB input."""
    rgb_path = frame_dir / f"input_v{int(exp.input_view)}_f{frame_id:06d}.png"
    video = Path(cfg.root) / cfg.source.rgb_videos / f"{int(exp.input_view)}.mp4"
    extract_frame(video, frame_id, rgb_path)
    image = Image.open(rgb_path).convert("RGB").resize(tuple(int(v) for v in exp.image_size))
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.iinfo(np.uint8).max


def train_model(data: dict, exp: dict, device: torch.device) -> tuple[MoReMouseTriplane, dict, list[dict]]:
    """Train image-to-triplane and mesh coefficient heads."""
    basis = pca_fit(data["vertices"], int(exp.pca_components))
    images = torch.from_numpy(data["images"]).to(device)
    targets = torch.from_numpy(basis["coeffs"]).to(device)
    model = MoReMouseTriplane(
        int(exp.pca_components), int(exp.hidden_dim), int(exp.plane_channels),
        int(exp.plane_size), int(exp.transformer_layers), int(exp.transformer_heads),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(exp.learning_rate))
    losses, best_loss, best_state = [], float("inf"), None
    for epoch in range(int(exp.epochs)):
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        coeff_loss = torch.nn.functional.mse_loss(output["coeffs"], targets)
        plane_loss = output["triplanes"].square().mean()
        loss = coeff_loss + float(exp.triplane_l2_weight) * plane_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite full MoReMouse loss")
        loss.backward()
        optimizer.step()
        if (epoch + 1) % int(exp.train_log_interval) == 0:
            item = {"epoch": epoch + 1, "coeff_mse": float(coeff_loss.detach().cpu())}
            losses.append(item)
            logger.info("epoch {} coeff_mse {:.6f}", item["epoch"], item["coeff_mse"])
        if float(coeff_loss.detach().cpu()) < best_loss:
            best_loss = float(coeff_loss.detach().cpu())
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No finite best model state was recorded")
    model.load_state_dict(best_state)
    losses.append({"epoch": "best", "coeff_mse": best_loss})
    return model, basis, losses


def render_eval(cfg: dict, exp: dict, data: dict, model: MoReMouseTriplane,
                basis: dict, output_dir: Path, device: torch.device) -> list[dict]:
    """Render learned mesh, geodesic embedding, and normals for evaluation frames."""
    cameras = pickle.load((Path(cfg.root) / cfg.source.cameras.primary).open("rb"))
    frame_to_index = {frame: index for index, frame in enumerate(data["frames"])}
    rows = []
    for frame_id in [int(frame) for frame in exp.eval_frames]:
        index = frame_to_index[frame_id]
        with torch.no_grad():
            output = model(torch.from_numpy(data["images"][index:index + 1]).to(device))
        vertices = reconstruct_vertices(output["coeffs"].cpu().numpy(), basis, data["vertices"].shape[1])
        row = render_frame(cfg, exp, frame_id, vertices, data, cameras, output_dir)
        row["target_source"] = data["best_rows"][index]["source"]
        rows.append(row)
    return rows


def render_frame(cfg: dict, exp: dict, frame_id: int, vertices: np.ndarray,
                 data: dict, cameras: list[dict], output_dir: Path) -> dict:
    """Render one full-stack reconstruction grid."""
    panels, scores = [input_panel(cfg, exp, frame_id, output_dir / "frames")], []
    geodesic_faces = vertex_to_face_colors(data["geodesic_colors"], data["faces"])
    normal_faces = face_normals(vertices, data["faces"])
    for view in [int(view) for view in cfg.views]:
        rgb_path, mask_path = extract_view(cfg, frame_id, view, output_dir / "frames")
        mask = Image.open(mask_path).convert("L")
        uv, _ = project_vertices(vertices, cameras[view])
        silhouette = rasterize_projected_silhouette(uv, data["faces"], mask.size)
        score = binary_iou(np.asarray(mask) > int(exp.mask_threshold), silhouette)
        scores.append(score)
        panels.append(render_panel(load_rgb(rgb_path), mask, silhouette, frame_id, view, score, exp))
        if view == int(exp.input_view):
            save_attribute_views(uv, data["faces"], geodesic_faces, normal_faces, mask.size, output_dir, frame_id)
    grid_path = output_dir / "grids" / f"full_moremouse_frame_{frame_id:06d}.png"
    save_pil_grid(panels, 4, grid_path, (18, 18, 18))
    return {"frame_id": frame_id, "mean_silhouette_iou": float(np.mean(scores)), "grid_path": str(grid_path)}


def save_attribute_views(uv: np.ndarray, faces: np.ndarray, geodesic_faces: np.ndarray, normal_faces: np.ndarray,
                         size: tuple[int, int], output_dir: Path, frame_id: int) -> None:
    """Save geodesic and normal synthetic supervision previews."""
    rasterize_face_colors(uv, faces, geodesic_faces, size).save(output_dir / "grids" / f"geodesic_f{frame_id:06d}.png")
    rasterize_face_colors(uv, faces, normal_faces, size).save(output_dir / "grids" / f"normal_f{frame_id:06d}.png")


def extract_view(cfg: dict, frame_id: int, view: int, frame_dir: Path) -> tuple[Path, Path]:
    """Extract RGB and mask frame files."""
    rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
    mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
    extract_frame(Path(cfg.root) / cfg.source.rgb_videos / f"{view}.mp4", frame_id, rgb_path)
    extract_frame(Path(cfg.root) / cfg.source.segmentation_masks / f"{view}.mp4", frame_id, mask_path)
    return rgb_path, mask_path


def input_panel(cfg: dict, exp: dict, frame_id: int, frame_dir: Path) -> Image.Image:
    """Create the monocular input panel."""
    rgb_path, mask_path = extract_view(cfg, frame_id, int(exp.input_view), frame_dir)
    image = overlay_mask(load_rgb(rgb_path), Image.open(mask_path).convert("L"), threshold=int(exp.mask_threshold))
    return label(image, f"single RGB input v{int(exp.input_view)} f{frame_id:06d}", exp)


def render_panel(rgb: Image.Image, mask: Image.Image, silhouette: np.ndarray,
                 frame_id: int, view: int, score: float, exp: dict) -> Image.Image:
    """Create one predicted render panel."""
    image = overlay_mesh_silhouette(overlay_mask(rgb, mask, threshold=int(exp.mask_threshold)), silhouette)
    return label(image, f"full pred v{view} f{frame_id:06d} IoU={score:.3f}", exp)


def label(image: Image.Image, text: str, exp: dict) -> Image.Image:
    """Resize and label a panel."""
    output = image.resize(tuple(int(value) for value in exp.panel_size)).convert("RGB")
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 34), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    return output


def scope_note(exp: dict) -> str:
    """Describe implemented paper modules."""
    return ("Local full-stack MoReMouse: MAMMAL best-source mesh as AGAM proxy, geodesic correspondence "
            f"with {int(exp.geodesic_anchors)} anchors, transformer-triplane image encoder, mesh extraction proxy.")

if __name__ == "__main__":
    main()
