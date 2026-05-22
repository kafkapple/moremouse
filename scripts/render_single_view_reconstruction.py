"""Render single-view input and six-view mesh reconstruction proxy panels."""

from pathlib import Path
import json
import pickle
import shutil

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from moremouse.data.video_frames import extract_frame
from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import binary_iou, project_vertices, rasterize_projected_silhouette
from moremouse.visualization.overlay import load_rgb, overlay_mask
from scripts.audit_camera_projection import overlay_mesh_silhouette


def main() -> None:
    """Write single-view-to-six-view reconstruction proxy visualizations."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    viz = cfg.single_view_reconstruction_viz
    root = Path(cfg.root)
    output_dir = Path(viz.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    frame_dir = output_dir / "frames"
    grid_dir = output_dir / "grids"
    frame_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)
    cameras = pickle.load((root / cfg.source.cameras.primary).open("rb"))
    best_sources = json.loads(Path(viz.mesh_source_report).read_text(encoding="utf-8"))["best_by_frame"]
    rows = []
    for frame_id in [int(frame) for frame in viz.frames]:
        row = render_frame(cfg, viz, best_sources, cameras, frame_id, frame_dir, grid_dir)
        rows.append(row)
    report = {"rows": rows, "input_view": int(viz.input_view), "note": reconstruction_note()}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote reconstruction proxy report to {}", output_dir / "report.json")


def render_frame(
    cfg: dict,
    viz: dict,
    best_sources: dict,
    cameras: list[dict],
    frame_id: int,
    frame_dir: Path,
    grid_dir: Path,
) -> dict:
    """Render one input panel plus six projected reconstruction panels."""
    source = best_sources[str(frame_id)]
    mesh = load_obj_mesh(Path(source["obj_path"]))
    views = [int(view) for view in cfg.views]
    threshold = int(cfg.visualization.mask_binary_threshold)
    cell_size = validate_panel_size([int(value) for value in viz.panel_size])
    panels = [input_panel(cfg, viz, frame_id, frame_dir, cell_size)]
    view_scores = []
    for view in views:
        rgb_path, mask_path = extract_rgb_mask(cfg, frame_id, view, frame_dir)
        rgb = load_rgb(rgb_path)
        mask = Image.open(mask_path).convert("L")
        uv, _ = project_vertices(mesh.vertices, cameras[view])
        silhouette = rasterize_projected_silhouette(uv, mesh.faces, mask.size)
        score = binary_iou(np.asarray(mask) > threshold, silhouette)
        view_scores.append(score)
        panels.append(render_panel(rgb, mask, silhouette, threshold, frame_id, view, score, cell_size))
    output_path = grid_dir / f"single_view_recon_frame_{frame_id:06d}.png"
    save_grid(panels, output_path)
    return {
        "frame_id": frame_id,
        "source": source["source"],
        "setting": source["setting"],
        "obj_path": source["obj_path"],
        "mean_silhouette_iou": float(np.mean(view_scores)),
        "view_silhouette_iou": [float(score) for score in view_scores],
        "grid_path": str(output_path),
    }


def input_panel(cfg: dict, viz: dict, frame_id: int, frame_dir: Path, cell_size: tuple[int, int]) -> Image.Image:
    """Build the single-view input panel."""
    view = int(viz.input_view)
    rgb_path, mask_path = extract_rgb_mask(cfg, frame_id, view, frame_dir)
    image = overlay_mask(load_rgb(rgb_path), Image.open(mask_path).convert("L"),
                         threshold=int(cfg.visualization.mask_binary_threshold))
    return label(image, f"single input v{view} f{frame_id:06d}", cell_size)


def extract_rgb_mask(cfg: dict, frame_id: int, view: int, frame_dir: Path) -> tuple[Path, Path]:
    """Extract RGB and mask frames if needed."""
    root = Path(cfg.root)
    rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
    mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
    extract_frame(root / cfg.source.rgb_videos / f"{view}.mp4", frame_id, rgb_path)
    extract_frame(root / cfg.source.segmentation_masks / f"{view}.mp4", frame_id, mask_path)
    return rgb_path, mask_path


def render_panel(
    rgb: Image.Image,
    mask: Image.Image,
    silhouette: np.ndarray,
    threshold: int,
    frame_id: int,
    view: int,
    score: float,
    cell_size: tuple[int, int],
) -> Image.Image:
    """Build one six-view reconstruction render panel."""
    image = overlay_mask(rgb, mask, threshold=threshold)
    image = overlay_mesh_silhouette(image, silhouette)
    text = f"render v{view} f{frame_id:06d} IoU={score:.3f}"
    return label(image, text, cell_size)


def label(image: Image.Image, text: str, cell_size: tuple[int, int]) -> Image.Image:
    """Resize a panel and add a readable title."""
    output = image.resize(cell_size).convert("RGB")
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 34), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255), font=font())
    return output


def save_grid(panels: list[Image.Image], output_path: Path) -> None:
    """Save seven panels as a 4x2 grid."""
    width, height = panels[0].size
    output = Image.new("RGB", (width * 4, height * 2), (18, 18, 18))
    for index, panel in enumerate(panels):
        output.paste(panel, ((index % 4) * width, (index // 4) * height))
    output.save(output_path)


def validate_panel_size(panel_size: list[int]) -> tuple[int, int]:
    """Validate panel size from config."""
    if len(panel_size) != 2:
        raise ValueError("panel_size must contain width and height")
    width, height = panel_size
    if width <= 0 or height <= 0:
        raise ValueError("panel_size values must be positive")
    return width, height


def font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable label font."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def reconstruction_note() -> str:
    """Return the scope note for these visualizations."""
    return (
        "This is the current best-source MAMMAL mesh reconstruction proxy, "
        "not a trained final MoReMouse single-image model output."
    )


if __name__ == "__main__":
    main()
