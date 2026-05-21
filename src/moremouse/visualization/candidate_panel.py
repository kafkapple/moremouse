"""Candidate mesh comparison panel rendering."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    """Save per-candidate RGB/mask/silhouette panels for one representative view."""
    if not frame_rows:
        raise ValueError("No candidate rows to save")
    view_index, view = choose_representative_view(frame_rows, views)
    cell_w, cell_h = validate_cell_size(cell_size)
    output = Image.new("RGB", (cell_w * 6, cell_h * len(frame_rows)), (18, 18, 18))
    detail_dir = output_path.parent / "candidate_details" / output_path.stem
    detail_dir.mkdir(parents=True, exist_ok=True)
    target = np.asarray(masks[view]) > threshold
    for row_index, (row, _) in enumerate(frame_rows):
        mesh = load_obj_mesh(Path(row["obj_path"]))
        uv, _ = project_vertices(mesh.vertices, cameras[view])
        silhouette = rasterize_projected_silhouette(uv, mesh.faces, masks[view].size)
        panels = build_panels(row, view, view_index, rgbs[view], target, silhouette)
        for col_index, panel in enumerate(panels):
            output.paste(panel.resize((cell_w, cell_h)), (col_index * cell_w, row_index * cell_h))
        detail_path = detail_dir / f"{sanitize_filename(row['candidate'])}.png"
        save_detail_row(panels, cell_w, cell_h, detail_path)
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


def build_panels(
    row: dict,
    view: int,
    view_index: int,
    rgb: Image.Image,
    target: np.ndarray,
    silhouette: np.ndarray,
) -> list[Image.Image]:
    """Build separate visual panels so comparison colors do not overlap."""
    candidate = str(row["candidate"])
    score = float(row["view_silhouette_iou"][view_index])
    return [
        title(rgb, f"{candidate} | RGB | view {view}"),
        title(mask_panel(target), "GT mask only"),
        title(mask_panel(silhouette), f"mesh mask only | IoU {score:.3f}"),
        title(color_mask(rgb, np.logical_and(target, silhouette), (60, 230, 110)), "overlap only"),
        title(color_mask(rgb, np.logical_and(~target, silhouette), (255, 60, 45)), "mesh-only FP"),
        title(color_mask(rgb, np.logical_and(target, ~silhouette), (65, 130, 255)), "GT-only FN"),
    ]


def color_mask(rgb: Image.Image, mask_array: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    """Overlay one colored binary mask on RGB."""
    base = rgb.convert("RGB")
    overlay = Image.new("RGB", base.size, color)
    mask = Image.fromarray((mask_array.astype(np.uint8) * 190), "L")
    return Image.composite(Image.blend(base, overlay, 0.65), base, mask)


def title(image: Image.Image, text: str) -> Image.Image:
    """Add a compact title bar to a panel."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    bar_h = 44
    draw.rectangle((0, 0, output.width, bar_h), fill=(0, 0, 0))
    draw.text((10, 7), text, fill=(255, 255, 255), font=label_font())
    return output


def label_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font with a portable fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        return ImageFont.load_default()


def save_detail_row(panels: list[Image.Image], cell_w: int, cell_h: int, output_path: Path) -> None:
    """Save one candidate detail row as a separate file."""
    output = Image.new("RGB", (cell_w * len(panels), cell_h), (18, 18, 18))
    for col_index, panel in enumerate(panels):
        output.paste(panel.resize((cell_w, cell_h)), (col_index * cell_w, 0))
    output.save(output_path)


def sanitize_filename(name: str) -> str:
    """Return a conservative filename for a candidate name."""
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in name]
    return "".join(allowed)
