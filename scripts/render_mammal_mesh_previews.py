"""Render previews for selected MAMMAL fitted meshes."""

from pathlib import Path
import json
import shutil

from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from moremouse.geometry.obj import load_obj_mesh
from moremouse.visualization.mesh_preview import render_mesh_triplet


def main() -> None:
    """Render mesh triplets and write mesh statistics."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    output_root = Path(cfg.outputs.mesh_preview_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    previews_dir = output_root / "previews"
    previews_dir.mkdir(parents=True)
    manifest = json.loads(Path(cfg.manifest).read_text(encoding="utf-8"))
    assets = {int(asset["frame_id"]): asset for asset in manifest["assets"]}
    frame_ids = [0, 2000, 6000, 12000, 17980]
    stats = []
    preview_paths = []
    for frame_id in frame_ids:
        asset = assets[frame_id]
        mesh = load_obj_mesh(Path(asset["obj_path"]))
        preview_path = previews_dir / f"mesh_frame_{frame_id:06d}.png"
        render_mesh_triplet(mesh, preview_path)
        preview_paths.append(preview_path)
        bounds_min, bounds_max = mesh.bounds
        stats.append(
            {
                "frame_id": frame_id,
                "vertices": int(mesh.vertices.shape[0]),
                "faces": int(mesh.faces.shape[0]),
                "obj_path": asset["obj_path"],
                "source": asset["source"],
                "bounds_min": bounds_min.tolist(),
                "bounds_max": bounds_max.tolist(),
            }
        )
    grid_path = output_root / "mammal_mesh_preview_grid.png"
    _save_grid(preview_paths, grid_path)
    report = {"frames": frame_ids, "stats": stats, "grid": str(grid_path)}
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote mesh preview report to {}", output_root / "report.json")


def _save_grid(preview_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in preview_paths]
    width = images[0].width
    height = images[0].height
    output = Image.new("RGB", (width, height * len(images)), (18, 18, 18))
    draw = ImageDraw.Draw(output)
    for row, image in enumerate(images):
        output.paste(image, (0, row * height))
        draw.text((8, row * height + 28), preview_paths[row].stem, fill=(255, 255, 255))
    output.save(output_path)


if __name__ == "__main__":
    main()
