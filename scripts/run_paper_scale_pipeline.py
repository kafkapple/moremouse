"""Run paper-scale dense-view and DMTet artifact generation."""

from pathlib import Path
import json
import shutil

import numpy as np
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from moremouse.geometry.dmtet import cube_tetrahedra_grid, marching_tetrahedra
from moremouse.geometry.geodesic_surface import farthest_point_anchors, geodesic_anchor_distances, geodesic_rgb_embedding
from moremouse.geometry.obj import load_obj_mesh
from moremouse.rendering.gaussian_avatar import build_surface_gaussians, render_gaussian_avatar
from moremouse.rendering.mesh_raster import face_normals, rasterize_face_colors, vertex_to_face_colors
from moremouse.rendering.virtual_cameras import spherical_virtual_cameras
from moremouse.visualization.grid import save_pil_grid


def main() -> None:
    """Generate dense 64-view AGAM-proxy renders and DMTet extraction artifacts."""
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset.paper_scale
    output_dir = Path(cfg.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "dense_views").mkdir(parents=True)
    (output_dir / "debug").mkdir(parents=True)
    mesh = load_selected_mesh(cfg)
    vertices = normalize_vertices(mesh.vertices)
    colors = geodesic_rgb_embedding(geodesic_anchor_distances(vertices, mesh.faces, farthest_point_anchors(vertices, 8)))
    cameras = spherical_virtual_cameras(int(cfg.dense_view_count), float(cfg.camera_radius),
                                        tuple(int(v) for v in cfg.dense_image_size), float(cfg.fov_degrees))
    rows = render_dense_views(vertices, mesh.faces, colors, cameras, output_dir)
    dmtet = extract_dmtet_proxy(int(cfg.dmtet_resolution), output_dir)
    report = {"frame_id": int(cfg.frame_id), "dense_view_count": len(rows), "dense_views": rows,
              "dmtet": dmtet, "scope": scope_note()}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote paper-scale report to {}", output_dir / "report.json")


def load_selected_mesh(cfg: dict):
    """Load the frame-selected best-source mesh."""
    report = json.loads(Path(cfg.mesh_source_report).read_text(encoding="utf-8"))
    row = report["best_by_frame"][str(int(cfg.frame_id))]
    return load_obj_mesh(Path(row["obj_path"]))


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """Normalize mesh vertices to the paper's unit-scale camera sphere setting."""
    center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    shifted = vertices - center
    scale = np.linalg.norm(shifted, axis=1).max()
    return (shifted / max(float(scale), np.finfo(np.float32).eps)).astype(np.float32)


def render_dense_views(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray,
                       cameras: list[dict], output_dir: Path) -> list[dict]:
    """Render dense Gaussian, geodesic, and normal supervision views."""
    avatar = build_surface_gaussians(vertices, faces, colors)
    geodesic_faces = vertex_to_face_colors(colors, faces)
    normal_faces = face_normals(vertices, faces)
    rows, previews = [], []
    for index, camera in enumerate(cameras):
        size = (camera["mapx"].shape[1], camera["mapx"].shape[0])
        gaussian = render_gaussian_avatar(avatar, camera, size)
        uv = project_centers(vertices, camera)
        geodesic = rasterize_face_colors(uv, faces, geodesic_faces, size)
        normal = rasterize_face_colors(uv, faces, normal_faces, size)
        paths = save_triplet(index, gaussian, geodesic, normal, output_dir / "dense_views")
        if index < 8:
            previews.append(label(gaussian.copy(), f"dense view {index:02d}"))
        rows.append({"view": index, **paths})
    save_pil_grid(previews, 4, output_dir / "debug" / "dense_64_preview.png", (0, 0, 0))
    return rows


def project_centers(vertices: np.ndarray, camera: dict) -> np.ndarray:
    """Project vertices using the same helper as Gaussian rendering."""
    from moremouse.geometry.projection import project_vertices
    uv, _ = project_vertices(vertices, camera)
    return uv


def save_triplet(index: int, gaussian: Image.Image, geodesic: Image.Image,
                 normal: Image.Image, output_dir: Path) -> dict[str, str]:
    """Save one dense-view supervision triplet."""
    paths = {
        "rgb": output_dir / f"view_{index:02d}_gaussian.png",
        "geodesic": output_dir / f"view_{index:02d}_geodesic.png",
        "normal": output_dir / f"view_{index:02d}_normal.png",
    }
    gaussian.save(paths["rgb"])
    geodesic.save(paths["geodesic"])
    normal.save(paths["normal"])
    return {key: str(value) for key, value in paths.items()}


def extract_dmtet_proxy(resolution: int, output_dir: Path) -> dict[str, object]:
    """Extract a DMTet sphere proxy to validate the DMTet stage."""
    vertices, tetrahedra = cube_tetrahedra_grid(resolution)
    sdf = np.linalg.norm(vertices, axis=1) - 0.72
    mesh_vertices, mesh_faces = marching_tetrahedra(vertices, tetrahedra, sdf)
    path = output_dir / "debug" / "dmtet_proxy.obj"
    write_obj(path, mesh_vertices, mesh_faces)
    return {"resolution": resolution, "vertices": int(mesh_vertices.shape[0]),
            "faces": int(mesh_faces.shape[0]), "obj_path": str(path)}


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangle OBJ mesh."""
    with path.open("w", encoding="utf-8") as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            file.write(f"f {int(face[0]) + 1} {int(face[1]) + 1} {int(face[2]) + 1}\n")


def label(image: Image.Image, text: str) -> Image.Image:
    """Label a dense-view preview tile."""
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 26), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return image


def scope_note() -> str:
    """Describe paper-scale generated artifacts."""
    return "64-view AGAM-proxy Gaussian, geodesic, normal supervision plus DMTet extraction proxy."


if __name__ == "__main__":
    main()
