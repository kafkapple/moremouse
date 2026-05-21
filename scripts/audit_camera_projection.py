"""Audit MAMMAL mesh projection against RGB masks and camera conventions."""

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
from moremouse.visualization.overlay import load_rgb, overlay_mask


def main() -> None:
    """Project fitted meshes with dataset cameras and save overlay diagnostics."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    root = Path(cfg.root)
    output_root = Path(cfg.outputs.camera_projection_audit_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    frame_dir = output_root / "frames"
    overlay_dir = output_root / "overlays"
    frame_dir.mkdir(parents=True)
    overlay_dir.mkdir(parents=True)

    cameras = pickle.load((root / cfg.source.cameras.primary).open("rb"))
    manifest = json.loads(Path(cfg.manifest).read_text(encoding="utf-8"))
    assets = {int(asset["frame_id"]): asset for asset in manifest["assets"]}
    frame_ids = [0, 2000, 6000, 12000, 17980]
    views = [int(view) for view in cfg.views]
    threshold = int(cfg.visualization.mask_binary_threshold)
    rows = []
    for frame_id in frame_ids:
        mesh = load_obj_mesh(Path(assets[frame_id]["obj_path"]))
        cells = []
        for view in views:
            rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
            mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
            extract_frame(root / cfg.source.rgb_videos / f"{view}.mp4", frame_id, rgb_path)
            extract_frame(root / cfg.source.segmentation_masks / f"{view}.mp4", frame_id, mask_path)
            uv, inside_ratio = project_vertices(mesh.vertices, cameras[view])
            mask = Image.open(mask_path).convert("L")
            mask_bbox = mask_bbox_xyxy(mask, threshold)
            mesh_bbox = uv_bbox_xyxy(uv, mask.size)
            bbox_iou = xyxy_iou(mask_bbox, mesh_bbox)
            image = overlay_mask(load_rgb(rgb_path), mask, threshold=threshold)
            image = draw_projection(image, uv, mesh.faces)
            cells.append(label(image.resize((320, 285)), frame_id, view, bbox_iou))
            rows.append(
                {
                    "frame_id": frame_id,
                    "view": view,
                    "source": assets[frame_id]["source"],
                    "inside_ratio": inside_ratio,
                    "bbox_iou": bbox_iou,
                    "mask_bbox": mask_bbox,
                    "mesh_bbox": mesh_bbox,
                }
            )
        save_grid(cells, overlay_dir / f"projection_frame_{frame_id:06d}.png")
    report = {
        "camera_convention": "OpenCV world-to-camera: x_cam = R @ x_world + T; pixel = K @ x_cam",
        "mask_threshold": threshold,
        "mean_bbox_iou": float(np.mean([row["bbox_iou"] for row in rows])),
        "rows": rows,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote projection audit to {}", output_root / "report.json")


def project_vertices(vertices: np.ndarray, camera: dict) -> tuple[np.ndarray, float]:
    """Project world vertices using MAMMAL camera R/T/K."""
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["T"], dtype=np.float64)
    intrinsic = np.asarray(camera["K"], dtype=np.float64)
    camera_points = (rotation @ vertices.astype(np.float64).T + translation[:, None]).T
    positive_depth = camera_points[:, 2] > np.finfo(np.float32).eps
    projected = np.full((vertices.shape[0], 2), np.nan, dtype=np.float64)
    homogeneous = (intrinsic @ camera_points[positive_depth].T).T
    projected[positive_depth] = homogeneous[:, :2] / homogeneous[:, 2:3]
    width = int(camera["mapx"].shape[1])
    height = int(camera["mapx"].shape[0])
    inside = (
        positive_depth
        & (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    return projected, float(inside.mean())


def mask_bbox_xyxy(mask: Image.Image, threshold: int) -> list[float]:
    """Compute foreground mask bbox."""
    array = np.asarray(mask)
    ys, xs = np.where(array > threshold)
    if len(xs) == 0:
        raise ValueError("Mask has no foreground")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def uv_bbox_xyxy(uv: np.ndarray, size: tuple[int, int]) -> list[float]:
    """Compute projected vertex bbox inside image bounds."""
    width, height = size
    finite = np.isfinite(uv).all(axis=1)
    inside = finite & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    if not inside.any():
        raise ValueError("Projection has no in-frame vertices")
    points = uv[inside]
    return [float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max())]


def xyxy_iou(first: list[float], second: list[float]) -> float:
    """Compute bbox IoU for xyxy boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0.0:
        raise ValueError("Degenerate bbox union")
    return float(intersection / union)


def draw_projection(image: Image.Image, uv: np.ndarray, faces: np.ndarray) -> Image.Image:
    """Draw projected mesh edges on an image."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for face in faces[:3500]:
        valid = [int(index) for index in face if index >= 0 and np.isfinite(uv[int(index)]).all()]
        for start, end in zip(valid, valid[1:] + valid[:1]):
            draw.line((tuple(uv[start]), tuple(uv[end])), fill=(255, 80, 60), width=1)
    return output


def label(image: Image.Image, frame_id: int, view: int, bbox_iou: float) -> Image.Image:
    """Add frame/view metadata to one audit cell."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    text = f"f{frame_id:06d} v{view} bboxIoU={bbox_iou:.2f}"
    draw.rectangle((0, 0, output.width, 24), fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return output


def save_grid(cells: list[Image.Image], output_path: Path) -> None:
    """Save six view cells as a 3x2 grid."""
    width = cells[0].width
    height = cells[0].height
    output = Image.new("RGB", (width * 3, height * 2), (20, 20, 20))
    for index, cell in enumerate(cells):
        output.paste(cell, ((index % 3) * width, (index // 3) * height))
    output.save(output_path)


if __name__ == "__main__":
    main()
