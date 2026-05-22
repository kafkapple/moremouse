"""Small DMTet-style marching tetrahedra extraction."""

import numpy as np

TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def marching_tetrahedra(vertices: np.ndarray, tetrahedra: np.ndarray, sdf: np.ndarray,
                        level: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Extract a triangle mesh from tetrahedral SDF samples."""
    out_vertices: list[np.ndarray] = []
    out_faces: list[list[int]] = []
    edge_to_vertex: dict[tuple[int, int], int] = {}
    for tet in tetrahedra:
        values = sdf[tet] - level
        inside = values < 0
        if inside.all() or not inside.any():
            continue
        polygon = []
        for left_local, right_local in TET_EDGES:
            left, right = int(tet[left_local]), int(tet[right_local])
            left_value, right_value = values[left_local], values[right_local]
            if (left_value < 0) == (right_value < 0):
                continue
            key = tuple(sorted((left, right)))
            if key not in edge_to_vertex:
                alpha = float(left_value / (left_value - right_value))
                point = vertices[left] * (1.0 - alpha) + vertices[right] * alpha
                edge_to_vertex[key] = len(out_vertices)
                out_vertices.append(point.astype(np.float32))
            polygon.append(edge_to_vertex[key])
        if len(polygon) == 3:
            out_faces.append(polygon)
        elif len(polygon) == 4:
            out_faces.append([polygon[0], polygon[1], polygon[2]])
            out_faces.append([polygon[0], polygon[2], polygon[3]])
    if not out_vertices:
        raise ValueError("marching tetrahedra produced no surface")
    return np.stack(out_vertices).astype(np.float32), np.asarray(out_faces, dtype=np.int32)


def cube_tetrahedra_grid(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a regular cube tetrahedral grid in [-1, 1]."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    vertices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    tetrahedra = []
    for x_coord in range(resolution - 1):
        for y_coord in range(resolution - 1):
            for z_coord in range(resolution - 1):
                cube = [_grid_index(x_coord + dx, y_coord + dy, z_coord + dz, resolution)
                        for dx, dy, dz in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))]
                tetrahedra.extend([[cube[0], cube[1], cube[3], cube[4]], [cube[1], cube[2], cube[3], cube[6]],
                                   [cube[1], cube[3], cube[4], cube[6]], [cube[1], cube[4], cube[5], cube[6]],
                                   [cube[3], cube[4], cube[6], cube[7]]])
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


def _grid_index(x_coord: int, y_coord: int, z_coord: int, resolution: int) -> int:
    """Flatten a regular grid coordinate."""
    return (x_coord * resolution + y_coord) * resolution + z_coord
