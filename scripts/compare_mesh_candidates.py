"""Compare candidate MAMMAL meshes with camera-projected mask IoU."""

from pathlib import Path
import json
import pickle
import shutil

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from moremouse.data.video_frames import extract_frame
from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import binary_iou, project_vertices, rasterize_projected_silhouette
from moremouse.visualization.candidate_panel import save_candidate_panels
from moremouse.visualization.overlay import load_rgb
from scripts.audit_camera_projection import (
    overlay_mesh_silhouette,
)


def main() -> None:
    """Render candidate comparison grids and write ranking report."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    root = Path(cfg.root)
    output_root = Path(cfg.outputs.mesh_candidate_audit_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    cameras = pickle.load((root / cfg.source.cameras.primary).open("rb"))
    threshold = int(cfg.visualization.mask_binary_threshold)
    views = [int(view) for view in cfg.views]
    rows = []
    for frame_cfg in cfg.mesh_candidate_audit.frames:
        frame_id = int(frame_cfg.frame_id)
        frame_dir = output_root / "frames"
        masks = load_masks(root, frame_id, views, frame_dir)
        rgbs = load_rgbs(root, frame_id, views, frame_dir, str(cfg.source.rgb_videos))
        frame_rows = []
        for candidate in frame_cfg.candidates:
            obj_path = Path(candidate.obj_path)
            if not obj_path.exists():
                logger.warning("Skipping missing candidate: {}", obj_path)
                continue
            mesh = load_obj_mesh(obj_path)
            cells = []
            scores = []
            for view in views:
                uv, _ = project_vertices(mesh.vertices, cameras[view])
                silhouette = rasterize_projected_silhouette(uv, mesh.faces, masks[view].size)
                target = np.asarray(masks[view]) > threshold
                score = binary_iou(target, silhouette)
                scores.append(score)
                image = overlay_mesh_silhouette(Image.new("RGB", masks[view].size, (20, 20, 20)), silhouette)
                image = composite_mask_outline(image, masks[view], threshold)
                cells.append(label(image.resize((256, 228)), str(candidate.name), view, score))
            row = {
                "frame_id": frame_id,
                "candidate": str(candidate.name),
                "setting": str(candidate.setting),
                "obj_path": str(obj_path),
                "mean_silhouette_iou": float(np.mean(scores)),
                "view_silhouette_iou": [float(score) for score in scores],
            }
            rows.append(row)
            frame_rows.append((row, cells))
        save_frame_grid(frame_rows, output_root / f"frame_{frame_id:06d}_candidate_grid.png")
        save_candidate_panels(
            frame_rows,
            views,
            rgbs,
            masks,
            threshold,
            cameras,
            [int(value) for value in cfg.mesh_candidate_audit.panel_cell_size],
            OmegaConf.to_container(cfg.mesh_candidate_audit.panel_style, resolve=True),
            output_root / f"frame_{frame_id:06d}_candidate_panels.png",
        )
    report = {"rows": rows, "best_by_frame": best_by_frame(rows)}
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote mesh candidate report to {}", output_root / "report.json")


def load_masks(root: Path, frame_id: int, views: list[int], frame_dir: Path) -> dict[int, Image.Image]:
    """Extract and load mask frames for all views."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    masks = {}
    for view in views:
        mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
        extract_frame(root / "simpleclick_undist" / f"{view}.mp4", frame_id, mask_path)
        masks[view] = Image.open(mask_path).convert("L")
    return masks


def load_rgbs(
    root: Path,
    frame_id: int,
    views: list[int],
    frame_dir: Path,
    video_dir: str,
) -> dict[int, Image.Image]:
    """Extract and load RGB frames for all views."""
    rgbs = {}
    for view in views:
        rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
        extract_frame(root / video_dir / f"{view}.mp4", frame_id, rgb_path)
        rgbs[view] = load_rgb(rgb_path)
    return rgbs


def composite_mask_outline(image: Image.Image, mask: Image.Image, threshold: int) -> Image.Image:
    """Overlay target mask in green under the candidate silhouette."""
    target = np.asarray(mask) > threshold
    target_img = Image.fromarray((target.astype(np.uint8) * 180), "L")
    green = Image.new("RGB", image.size, (40, 220, 120))
    blended = Image.blend(image.convert("RGB"), green, 0.35)
    return Image.composite(blended, image.convert("RGB"), target_img)


def label(image: Image.Image, name: str, view: int, score: float) -> Image.Image:
    """Add candidate/view score label."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 26), fill=(0, 0, 0))
    draw.text((5, 6), f"{name} v{view} IoU={score:.2f}", fill=(255, 255, 255))
    return output


def save_frame_grid(frame_rows: list[tuple[dict, list[Image.Image]]], output_path: Path) -> None:
    """Save one candidate-vs-view grid for a frame."""
    if not frame_rows:
        raise ValueError("No candidate rows to save")
    cell_w = frame_rows[0][1][0].width
    cell_h = frame_rows[0][1][0].height
    output = Image.new("RGB", (cell_w * len(frame_rows[0][1]), cell_h * len(frame_rows)), (15, 15, 15))
    for row_index, (_, cells) in enumerate(frame_rows):
        for col_index, cell in enumerate(cells):
            output.paste(cell, (col_index * cell_w, row_index * cell_h))
    output.save(output_path)


def best_by_frame(rows: list[dict]) -> dict[str, dict]:
    """Return the highest mean-IoU candidate for each frame."""
    best = {}
    for row in rows:
        key = str(row["frame_id"])
        if key not in best or row["mean_silhouette_iou"] > best[key]["mean_silhouette_iou"]:
            best[key] = row
    return best


if __name__ == "__main__":
    main()
