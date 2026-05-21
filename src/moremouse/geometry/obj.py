"""Minimal OBJ mesh loading for MAMMAL fitting outputs."""

from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, field_validator


class ObjMesh(BaseModel, frozen=True):
    """Triangle or polygon mesh parsed from an OBJ file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    vertices: Float[np.ndarray, "vertices 3"]
    faces: Int[np.ndarray, "faces corners"]

    @field_validator("vertices")
    @classmethod
    def validate_vertices(cls, value: np.ndarray) -> np.ndarray:
        """Validate vertex array shape and finite values."""
        if value.ndim != 2 or value.shape[1] != 3:
            raise ValueError("vertices must have shape [N, 3]")
        if not np.isfinite(value).all():
            raise ValueError("vertices contain non-finite values")
        return value

    @field_validator("faces")
    @classmethod
    def validate_faces(cls, value: np.ndarray) -> np.ndarray:
        """Validate face array shape."""
        if value.ndim != 2 or value.shape[1] < 3:
            raise ValueError("faces must have shape [F, >=3]")
        if np.any(value < 0):
            raise ValueError("faces must be zero-based non-negative indices")
        return value

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return axis-aligned min and max bounds."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def load_obj_mesh(path: Path) -> ObjMesh:
    """Load vertex and face records from an OBJ file."""
    if not path.exists():
        raise FileNotFoundError(f"OBJ file does not exist: {path}")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([_parse_face_index(token) for token in line.split()[1:]])
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    if not faces:
        raise ValueError(f"OBJ contains no faces: {path}")
    return ObjMesh(vertices=np.asarray(vertices, dtype=np.float32), faces=_pad_faces(faces))


def _parse_face_index(token: str) -> int:
    """Parse a one-based OBJ face vertex token to zero-based index."""
    raw_index = token.split("/")[0]
    index = int(raw_index)
    if index <= 0:
        raise ValueError("Only positive OBJ face indices are supported")
    return index - 1


def _pad_faces(faces: list[list[int]]) -> np.ndarray:
    """Pad polygon faces to a rectangular integer array."""
    max_corners = max(len(face) for face in faces)
    padded = np.full((len(faces), max_corners), -1, dtype=np.int32)
    for row, face in enumerate(faces):
        padded[row, : len(face)] = face
        padded[row, len(face) :] = face[-1]
    return padded
