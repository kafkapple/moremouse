"""Candidate mesh comparison panel rendering."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from moremouse.geometry.obj import load_obj_mesh
from moremouse.geometry.projection import project_vertices, rasterize_projected_silhouette


def save_candidate_panels(
    frame_rows: list[tuple[dict, list[Image.Image]]],
    views: list[int],
    rgbs: dict[int, Image.Image],
    masks: dict[int, Image.Image],
    threshold: int,
    cameras: list[dict],
    cell_size: list[int],
    output_path: Path,
) -> None:
    """Save per-candidate RGB/mask/silhouette/error panels for one representative view."""
    if not frame_rows:
        raise ValueError("No candidate rows to save")
    view_index, view = choose_representative_view(frame_rows, views)
    cell_w, cell_h = validate_cell_size(cell_size)
    output = Image.new("RGB", (cell_w * 4, cell_h * len(frame_rows)), (18, 18, 18))
    target = np.asarray(masks[view]) > threshold
    for row_index, (row, _) in enumerate(frame_rows):
        mesh = load_obj_mesh(Path(row["obj_path"]))
        uv, _ = project_vertices(mesh.vertices, cameras[view])
        silhouette = rasterize_projected_silhouette(uv, mesh.faces, masks[view].size)
        panels = [
            title(rgbs[view], f"{row['candidate']} RGB v{view}"),
            title(mask_panel(target), "GT mask"),
            title(mask_panel(silhouette), f"mesh IoU={row['view_silhouette_iou'][view_index]:.2f}"),
            title(error_overlay(rgbs[view], target, silhouette), "green overlap / red FP / blue FN"),
        ]
        for col_index, panel in enumerate(panels):
            output.paste(panel.resize((cell_w, cell_h)), (col_index * cell_w, row_index * cell_h))
    output.save(output_path)


def choose_representative_view(
    frame_rows: list[tuple[dict, list[Image.Image]]],
    views: list[int],
) -> tuple[int, int]:
    """Choose the configured view with the largest candidate IoU spread."""
    scores = np.asarray([row["view_silhouette_iou"] for row, _ in frame_rows])
    view_index = int(np.argmax(scores.max(axis=0) - scores.min(axis=0)))
    return view_index, views[view_index]


def validate_cell_size(cell_size: list[int]) -> tuple[int, int]:
    """Validate panel cell size from config."""
    if len(cell_size) != 2:
        raise ValueError("panel_cell_size must contain width and height")
    width, height = cell_size
    if width <= 0 or height <= 0:
        raise ValueError("panel_cell_size values must be positive")
    return width, height


def mask_panel(mask: np.ndarray) -> Image.Image:
    """Render a binary mask as a high-contrast RGB panel."""
    array = mask.astype(np.uint8) * 255
    return Image.fromarray(array, "L").convert("RGB")


def error_overlay(rgb: Image.Image, target: np.ndarray, silhouette: np.ndarray) -> Image.Image:
    """Render target/candidate agreement and errors over RGB."""
    base = rgb.convert("RGB")
    colors = np.zeros((base.height, base.width, 3), dtype=np.uint8)
    colors[np.logical_and(target, silhouette)] = (60, 230, 110)
    colors[np.logical_and(~target, silhouette)] = (255, 60, 45)
    colors[np.logical_and(target, ~silhouette)] = (65, 130, 255)
    overlay = Image.fromarray(colors, "RGB")
    mask = Image.fromarray((np.logical_or(target, silhouette).astype(np.uint8) * 180), "L")
    return Image.composite(Image.blend(base, overlay, 0.65), base, mask)


def title(image: Image.Image, text: str) -> Image.Image:
    """Add a compact title bar to a panel."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 28), fill=(0, 0, 0))
    draw.text((6, 7), text, fill=(255, 255, 255))
    return output
