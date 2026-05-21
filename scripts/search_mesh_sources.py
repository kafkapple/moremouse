"""Search MAMMAL mesh sources for best projected silhouette candidates."""

from pathlib import Path
import json
import pickle
import re
import shutil

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from moremouse.data.video_frames import extract_frame
from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import binary_iou, project_vertices, rasterize_projected_silhouette

FRAME_PATTERN = re.compile(r"frame_(\d+)\.obj$")


def main() -> None:
    """Compare available mesh source directories and write best-source report."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    root = Path(cfg.root)
    search_cfg = cfg.mesh_source_search
    output_dir = Path(search_cfg.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    cameras = pickle.load((root / cfg.source.cameras.primary).open("rb"))
    threshold = int(cfg.visualization.mask_binary_threshold)
    views = [int(view) for view in cfg.views]
    index = build_index(search_cfg.sources)
    frame_ids = select_frame_ids(index, search_cfg)
    rows = compare_frames(root, frame_ids, index, cameras, views, threshold, output_dir)
    report = {
        "frame_ids": frame_ids,
        "rows": rows,
        "best_by_frame": best_by_frame(rows),
        "source_counts": source_counts(index),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote mesh source search report to {}", output_dir / "report.json")


def build_index(sources: list[dict]) -> dict[int, list[dict]]:
    """Index OBJ candidates by frame id."""
    index: dict[int, list[dict]] = {}
    for source in sources:
        obj_dir = Path(source.obj_dir)
        for obj_path in sorted(obj_dir.glob("*.obj")):
            match = FRAME_PATTERN.search(obj_path.name)
            if not match:
                continue
            frame_id = int(match.group(1))
            index.setdefault(frame_id, []).append({
                "source": str(source.name),
                "setting": str(source.setting),
                "obj_path": str(obj_path),
            })
    return index


def select_frame_ids(index: dict[int, list[dict]], search_cfg: dict) -> list[int]:
    """Select explicit frames plus high-candidate-count frames."""
    explicit = {int(frame_id) for frame_id in search_cfg.frame_ids}
    min_candidates = int(search_cfg.min_candidates)
    ranked = sorted(index, key=lambda frame: (-len(index[frame]), frame))
    auto = [frame for frame in ranked if len(index[frame]) >= min_candidates]
    frame_ids = list(explicit)
    for frame_id in auto[: int(search_cfg.max_auto_frames)]:
        frame_ids.append(frame_id)
    return sorted({frame for frame in frame_ids if frame in index})


def compare_frames(
    root: Path,
    frame_ids: list[int],
    index: dict[int, list[dict]],
    cameras: list[dict],
    views: list[int],
    threshold: int,
    output_dir: Path,
) -> list[dict]:
    """Compare all indexed candidates for selected frames."""
    rows = []
    mask_dir = output_dir / "masks"
    for frame_id in frame_ids:
        masks = load_masks(root, frame_id, views, mask_dir)
        for candidate in index[frame_id]:
            mesh = load_obj_mesh(Path(candidate["obj_path"]))
            scores = []
            for view in views:
                uv, _ = project_vertices(mesh.vertices, cameras[view])
                silhouette = rasterize_projected_silhouette(uv, mesh.faces, masks[view].size)
                target = np.asarray(masks[view]) > threshold
                scores.append(binary_iou(target, silhouette))
            rows.append({
                "frame_id": frame_id,
                "source": candidate["source"],
                "setting": candidate["setting"],
                "obj_path": candidate["obj_path"],
                "mean_silhouette_iou": float(np.mean(scores)),
                "view_silhouette_iou": [float(score) for score in scores],
            })
    return rows


def load_masks(root: Path, frame_id: int, views: list[int], mask_dir: Path) -> dict[int, Image.Image]:
    """Extract and load mask frames for all views."""
    masks = {}
    for view in views:
        mask_path = mask_dir / f"mask_v{view}_f{frame_id:06d}.png"
        extract_frame(root / "simpleclick_undist" / f"{view}.mp4", frame_id, mask_path)
        masks[view] = Image.open(mask_path).convert("L")
    return masks


def best_by_frame(rows: list[dict]) -> dict[str, dict]:
    """Return best candidate row for each frame."""
    best = {}
    for row in rows:
        key = str(row["frame_id"])
        if key not in best or row["mean_silhouette_iou"] > best[key]["mean_silhouette_iou"]:
            best[key] = row
    return best


def source_counts(index: dict[int, list[dict]]) -> dict[str, int]:
    """Count indexed OBJ files by source."""
    counts: dict[str, int] = {}
    for candidates in index.values():
        for candidate in candidates:
            counts[candidate["source"]] = counts.get(candidate["source"], 0) + 1
    return counts


if __name__ == "__main__":
    main()
