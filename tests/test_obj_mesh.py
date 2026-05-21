"""Tests for OBJ mesh loading."""

from pathlib import Path

import numpy as np

from moremouse.geometry.obj import load_obj_mesh
from moremouse.visualization.mesh_preview import render_mesh_triplet


def test_load_obj_mesh_parses_vertices_and_faces(tmp_path: Path) -> None:
    """Verify OBJ vertices and one-based faces are parsed."""
    obj_path = tmp_path / "mesh.obj"
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )

    mesh = load_obj_mesh(obj_path)

    assert mesh.vertices.shape == (3, 3)
    assert np.array_equal(mesh.faces[0], np.array([0, 1, 2], dtype=np.int32))


def test_render_mesh_triplet_writes_image(tmp_path: Path) -> None:
    """Verify mesh preview rendering writes an image."""
    obj_path = tmp_path / "mesh.obj"
    image_path = tmp_path / "preview.png"
    obj_path.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n",
        encoding="utf-8",
    )

    render_mesh_triplet(load_obj_mesh(obj_path), image_path, size=64)

    assert image_path.exists()
