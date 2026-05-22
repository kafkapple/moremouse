"""Shared helpers for AGAM training and visualization."""

from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw

from moremouse.data.agam import avatar_to_visualization_world, target_avatar_to_render_avatar
from moremouse.data.video_frames import extract_frame
from moremouse.geometry.obj import load_obj_mesh
from moremouse.models.agam import AgamAvatarModel
from moremouse.rendering.gaussian_avatar import render_gaussian_avatar
from moremouse.rendering.virtual_cameras import spherical_virtual_cameras
from moremouse.visualization.grid import save_pil_grid


def build_batch(dataset_cfg: dict, exp: dict, output_dir: Path) -> dict[str, object]:
    """Load the canonical mesh, train-frame meshes, and input images."""
    report = json.loads(Path(exp.mesh_source_report).read_text(encoding="utf-8"))
    best_rows = report["best_by_frame"]
    canonical_row = best_rows[str(int(exp.canonical_frame_id))]
    canonical_mesh = load_obj_mesh(Path(canonical_row["obj_path"]))
    frame_meshes = []
    frame_images = []
    frame_ids = [int(frame) for frame in exp.train_frames]
    for frame_id in frame_ids:
        row = best_rows[str(frame_id)]
        frame_meshes.append(load_obj_mesh(Path(row["obj_path"])))
        frame_images.append(load_input(dataset_cfg, exp, frame_id, output_dir / "frames"))
    return {
        "canonical_mesh": canonical_mesh,
        "frame_meshes": frame_meshes,
        "images": np.stack(frame_images),
        "frame_ids": frame_ids,
        "best_rows": best_rows,
    }


def load_input(dataset_cfg: dict, exp: dict, frame_id: int, frame_dir: Path) -> np.ndarray:
    """Extract and normalize one monocular RGB frame."""
    rgb_path = frame_dir / f"input_v{int(exp.input_view)}_f{frame_id:06d}.png"
    video = Path(dataset_cfg.root) / dataset_cfg.source.rgb_videos / f"{int(exp.input_view)}.mp4"
    extract_frame(video, frame_id, rgb_path)
    image = Image.open(rgb_path).convert("RGB").resize(tuple(int(v) for v in exp.image_size))
    return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.iinfo(np.uint8).max


def render_eval(
    batch: dict[str, object],
    exp: dict,
    model: AgamAvatarModel,
    target_batch,
    device: torch.device,
    output_dir: Path,
    dataset_cfg: dict,
) -> list[dict]:
    """Render input/target/predicted comparison grids for configured eval frames."""
    cameras = pickle.load((Path(dataset_cfg.root) / dataset_cfg.source.cameras.primary).open("rb"))
    frame_to_index = {frame_id: index for index, frame_id in enumerate(batch["frame_ids"])}
    rows: list[dict] = []
    for frame_id in [int(frame) for frame in exp.eval_frames]:
        index = frame_to_index[frame_id]
        image = torch.from_numpy(batch["images"][index:index + 1]).to(device)
        with torch.no_grad():
            prediction = model(image)
        target_avatar = target_avatar_to_render_avatar(single_avatar(target_batch, index))
        predicted_avatar = target_avatar_to_render_avatar(single_avatar(prediction.avatar, 0))
        input_rgb = Image.open(output_dir / "frames" / f"input_v{int(exp.input_view)}_f{frame_id:06d}.png").convert("RGB")
        camera = cameras[int(exp.input_view)]
        target_render = render_gaussian_avatar(target_avatar, camera, input_rgb.size)
        predicted_render = render_gaussian_avatar(predicted_avatar, camera, input_rgb.size)
        diff = ImageChops.difference(target_render, predicted_render)
        grid_path = output_dir / "grids" / f"agam_eval_f{frame_id:06d}.png"
        save_pil_grid(
            [
                label(input_rgb, f"input f{frame_id:06d}"),
                label(target_render, f"target f{frame_id:06d}"),
                label(predicted_render, f"pred f{frame_id:06d}"),
                label(diff, f"diff f{frame_id:06d}"),
            ],
            2,
            grid_path,
            (18, 18, 18),
        )
        rows.append({
            "frame_id": frame_id,
            "grid_path": str(grid_path),
            "mean_abs_diff": float(np.asarray(diff).mean()),
        })
    return rows


def render_dense_preview(
    batch: dict[str, object],
    exp: dict,
    model: AgamAvatarModel,
    target_batch,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Render a dense target/prediction preview on orbit cameras."""
    frame_to_index = {frame_id: index for index, frame_id in enumerate(batch["frame_ids"])}
    preview_frame_id = int(exp.eval_frames[-1]) if exp.eval_frames else int(batch["frame_ids"][0])
    preview_index = frame_to_index.get(preview_frame_id, 0)
    cameras = spherical_virtual_cameras(
        int(exp.dense_view_count),
        float(exp.camera_radius),
        tuple(int(v) for v in exp.dense_image_size),
        float(exp.fov_degrees),
    )
    with torch.no_grad():
        prediction = model(torch.from_numpy(batch["images"][preview_index:preview_index + 1]).to(device))
    target_avatar = avatar_to_visualization_world(target_avatar_to_render_avatar(single_avatar(target_batch, preview_index)))
    predicted_avatar = avatar_to_visualization_world(target_avatar_to_render_avatar(single_avatar(prediction.avatar, 0)))
    target_tiles: list[Image.Image] = []
    predicted_tiles: list[Image.Image] = []
    for view, camera in enumerate(cameras[: int(exp.preview_top_k)]):
        target_tiles.append(label(render_gaussian_avatar(target_avatar, camera, tuple(int(v) for v in exp.dense_image_size)),
                                  f"target view {view:02d}"))
        predicted_tiles.append(label(render_gaussian_avatar(predicted_avatar, camera, tuple(int(v) for v in exp.dense_image_size)),
                                     f"pred view {view:02d}"))
    preview_path = output_dir / "grids" / "agam_dense_preview.png"
    save_pil_grid(target_tiles + predicted_tiles, int(exp.preview_top_k), preview_path, (0, 0, 0))
    return {
        "preview_path": str(preview_path),
        "dense_view_count": int(exp.dense_view_count),
        "preview_top_k": int(exp.preview_top_k),
        "preview_frame_id": preview_frame_id,
    }


def single_avatar(avatar, index: int):
    """Extract one avatar instance from a batch."""
    return type(avatar)(
        centers=avatar.centers[index].detach().cpu(),
        colors=avatar.colors[index].detach().cpu(),
        opacities=avatar.opacities[index].detach().cpu(),
        scales=avatar.scales[index].detach().cpu(),
        rotations=avatar.rotations[index].detach().cpu(),
    )


def label(image: Image.Image, text: str) -> Image.Image:
    """Annotate one image tile."""
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 24), fill=(0, 0, 0))
    draw.text((6, 4), text, fill=(255, 255, 255))
    return output


def scope_note() -> str:
    """Return a short description of the AGAM stage."""
    return "Anchor-based AGAM proxy training on canonical fitted MAMMAL meshes."
