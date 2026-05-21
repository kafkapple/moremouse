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
    style: dict,
    output_path: Path,
) -> None:
    """Save per-candidate RGB/mask/silhouette panels for one representative view."""
    if not frame_rows:
        raise ValueError("No candidate rows to save")
    view_index, view = choose_representative_view(frame_rows, views)
    cell_w, cell_h = validate_cell_size(cell_size)
    panel_style = validate_style(style)
    output = Image.new("RGB", (cell_w * 6, cell_h * len(frame_rows)), (18, 18, 18))
    detail_dir = output_path.parent / "candidate_details" / output_path.stem
    detail_dir.mkdir(parents=True, exist_ok=True)
    target = np.asarray(masks[view]) > threshold
    for row_index, (row, _) in enumerate(frame_rows):
        mesh = load_obj_mesh(Path(row["obj_path"]))
        uv, _ = project_vertices(mesh.vertices, cameras[view])
        silhouette = rasterize_projected_silhouette(uv, mesh.faces, masks[view].size)
        panels = build_panels(row, view, view_index, rgbs[view], target, silhouette, panel_style)
        for col_index, panel in enumerate(panels):
            output.paste(panel.resize((cell_w, cell_h)), (col_index * cell_w, row_index * cell_h))
        detail_path = detail_dir / f"{sanitize_filename(row['candidate'])}.png"
        save_detail_row(panels, str(row["setting"]), cell_w, cell_h, panel_style, detail_path)
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


def validate_style(style: dict) -> dict:
    """Validate panel style values from config."""
    required = {
        "title_bar_height",
        "title_font_size",
        "detail_header_height",
        "detail_header_font_size",
        "detail_meta_font_size",
        "detail_grid_size",
    }
    missing = required - set(style)
    if missing:
        raise ValueError(f"Missing panel style keys: {sorted(missing)}")
    grid_size = [int(value) for value in style["detail_grid_size"]]
    if len(grid_size) != 2:
        raise ValueError("detail_grid_size must contain columns and rows")
    return {
        "title_bar_height": int(style["title_bar_height"]),
        "title_font_size": int(style["title_font_size"]),
        "detail_header_height": int(style["detail_header_height"]),
        "detail_header_font_size": int(style["detail_header_font_size"]),
        "detail_meta_font_size": int(style["detail_meta_font_size"]),
        "detail_grid_columns": grid_size[0],
        "detail_grid_rows": grid_size[1],
    }


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
    style: dict,
) -> list[Image.Image]:
    """Build separate visual panels so comparison colors do not overlap."""
    candidate = str(row["candidate"])
    score = float(row["view_silhouette_iou"][view_index])
    return [
        title(rgb, f"{candidate} | RGB | view {view}", style),
        title(mask_panel(target), "GT mask only", style),
        title(mask_panel(silhouette), f"mesh mask only | IoU {score:.3f}", style),
        title(color_mask(rgb, np.logical_and(target, silhouette), (60, 230, 110)), "overlap only", style),
        title(color_mask(rgb, np.logical_and(~target, silhouette), (255, 60, 45)), "mesh-only FP", style),
        title(color_mask(rgb, np.logical_and(target, ~silhouette), (65, 130, 255)), "GT-only FN", style),
    ]


def color_mask(rgb: Image.Image, mask_array: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    """Overlay one colored binary mask on RGB."""
    base = rgb.convert("RGB")
    overlay = Image.new("RGB", base.size, color)
    mask = Image.fromarray((mask_array.astype(np.uint8) * 190), "L")
    return Image.composite(Image.blend(base, overlay, 0.65), base, mask)


def title(image: Image.Image, text: str, style: dict) -> Image.Image:
    """Add a compact title bar to a panel."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, style["title_bar_height"]), fill=(0, 0, 0))
    draw.text((10, 10), text, fill=(255, 255, 255), font=label_font(style["title_font_size"]))
    return output


def label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font with a portable fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_detail_row(
    panels: list[Image.Image],
    setting: str,
    cell_w: int,
    cell_h: int,
    style: dict,
    output_path: Path,
) -> None:
    """Save one candidate detail image as a readable 3x2 grid."""
    width = cell_w * style["detail_grid_columns"]
    height = cell_h * style["detail_grid_rows"] + style["detail_header_height"]
    output = Image.new("RGB", (width, height), (18, 18, 18))
    draw_detail_header(output, output_path.stem, setting, style)
    for col_index, panel in enumerate(panels):
        x_offset = (col_index % style["detail_grid_columns"]) * cell_w
        y_offset = (col_index // style["detail_grid_columns"]) * cell_h + style["detail_header_height"]
        output.paste(panel.resize((cell_w, cell_h)), (x_offset, y_offset))
    output.save(output_path)


def draw_detail_header(image: Image.Image, candidate: str, setting: str, style: dict) -> None:
    """Draw a large candidate label above a detail grid."""
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, style["detail_header_height"]), fill=(0, 0, 0))
    draw.text(
        (14, 10),
        f"candidate: {candidate}",
        fill=(255, 255, 255),
        font=label_font(style["detail_header_font_size"]),
    )
    draw.text(
        (14, 48),
        f"setting: {setting}",
        fill=(255, 255, 255),
        font=label_font(style["detail_meta_font_size"]),
    )
    draw.text(
        (14, 82),
        "color: green=overlap | red=mesh-only FP | blue=GT-only FN",
        fill=(255, 255, 255),
        font=label_font(style["detail_meta_font_size"]),
    )


def sanitize_filename(name: str) -> str:
    """Return a conservative filename for a candidate name."""
    allowed = [char if char.isalnum() or char in {"-", "_"} else "_" for char in name]
    return "".join(allowed)
