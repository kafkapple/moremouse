"""Render markerless/MAMMAL data sanity grids and a short video."""

from pathlib import Path
import json
import pickle
import shutil

from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from moremouse.data.video_frames import encode_video, extract_frame
from moremouse.visualization.overlay import draw_keypoints, load_rgb, overlay_mask


def main() -> None:
    """Create RGB/mask/keypoint grids for canonical markerless frames."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    root = Path(cfg.root)
    output_root = Path(cfg.outputs.sanity_dir)
    frame_ids = [0, 2000, 6000, 12000, 17980]
    views = [int(view) for view in cfg.views]
    frame_dir = output_root / "frames"
    grid_dir = output_root / "grids"
    if output_root.exists():
        shutil.rmtree(output_root)
    frame_dir.mkdir(parents=True)
    grid_dir.mkdir(parents=True)

    keypoints_by_view = _load_keypoints(root, views)
    grid_paths = []
    for grid_index, frame_id in enumerate(frame_ids):
        cells = []
        for view in views:
            rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
            mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
            extract_frame(root / cfg.source.rgb_videos / f"{view}.mp4", frame_id, rgb_path)
            extract_frame(root / cfg.source.segmentation_masks / f"{view}.mp4", frame_id, mask_path)
            image = overlay_mask(load_rgb(rgb_path), Image.open(mask_path))
            image = draw_keypoints(image, keypoints_by_view[view][frame_id])
            cells.append(_label(image.resize((320, 320)), f"view {view} frame {frame_id}"))
        grid_path = grid_dir / f"grid_{grid_index:03d}_frame_{frame_id:06d}.png"
        _save_grid(cells, grid_path)
        grid_paths.append(grid_path)

    video_path = output_root / "markerless_sanity.mp4"
    encode_video(str(grid_dir / "grid_*.png"), video_path, fps=2)
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "frames": frame_ids,
                "views": views,
                "grids": [str(path) for path in grid_paths],
                "video": str(video_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote sanity report to {}", report_path)


def _load_keypoints(root: Path, views: list[int]) -> dict[int, object]:
    keypoints = {}
    for view in views:
        path = root / "keypoints2d_undist" / f"result_view_{view}.pkl"
        with path.open("rb") as file:
            keypoints[view] = pickle.load(file)
    return keypoints


def _label(image: Image.Image, label: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 24), fill=(0, 0, 0))
    draw.text((8, 5), label, fill=(255, 255, 255))
    return output


def _save_grid(cells: list[Image.Image], output_path: Path) -> None:
    width = max(cell.width for cell in cells)
    height = max(cell.height for cell in cells)
    output = Image.new("RGB", (width * 3, height * 2), (20, 20, 20))
    for index, cell in enumerate(cells):
        output.paste(cell, ((index % 3) * width, (index // 3) * height))
    output.save(output_path)


if __name__ == "__main__":
    main()
