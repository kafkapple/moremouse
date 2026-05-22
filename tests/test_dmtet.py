import numpy as np

from moremouse.geometry import cube_tetrahedra_grid, marching_tetrahedra


def test_marching_tetrahedra_extracts_sphere() -> None:
    """Extract an isosurface from a sphere SDF on a tetrahedral cube grid."""
    vertices, tetrahedra = cube_tetrahedra_grid(5)
    sdf = np.linalg.norm(vertices, axis=1) - 0.75

    out_vertices, out_faces = marching_tetrahedra(vertices, tetrahedra, sdf)

    assert out_vertices.shape[1] == 3
    assert out_faces.shape[1] == 3
    assert len(out_vertices) > 0
