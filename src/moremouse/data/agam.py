"""AGAM template and target-avatar builders."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from moremouse.geometry.geodesic_surface import (
    farthest_point_anchors,
    geodesic_anchor_distances,
    geodesic_rgb_embedding,
)
from moremouse.geometry.obj import ObjMesh
from moremouse.rendering.gaussian_avatar import GaussianAvatar


@dataclass(frozen=True)
class AgamTemplate:
    """Canonical anchor template used by the local AGAM proxy."""

    anchor_indices: np.ndarray
    geodesic_anchor_indices: np.ndarray
    centers: np.ndarray
    colors: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray


@dataclass(frozen=True)
class TorchGaussianAvatar:
    """Torch-native avatar container for AGAM regression."""

    centers: torch.Tensor
    colors: torch.Tensor
    opacities: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor


def build_agam_template(mesh: ObjMesh, anchor_count: int, geodesic_anchor_count: int = 8) -> AgamTemplate:
    """Build a canonical anchor template from one fitted mesh."""
    if anchor_count < 1:
        raise ValueError("anchor_count must be positive")
    anchors = farthest_point_anchors(mesh.vertices, min(anchor_count, mesh.vertices.shape[0]))
    geodesic_anchors = farthest_point_anchors(mesh.vertices, min(geodesic_anchor_count, mesh.vertices.shape[0]))
    colors = _anchor_colors(mesh.vertices, mesh.faces, anchors, geodesic_anchors)
    scales = _anchor_scales(mesh.vertices, mesh.faces, anchors)
    opacities = np.full(anchors.shape[0], 0.95, dtype=np.float32)
    return AgamTemplate(
        anchor_indices=anchors.astype(np.int32),
        geodesic_anchor_indices=geodesic_anchors.astype(np.int32),
        centers=mesh.vertices[anchors].astype(np.float32),
        colors=colors,
        scales=scales,
        opacities=opacities,
    )


def build_target_avatar(mesh: ObjMesh, template: AgamTemplate) -> TorchGaussianAvatar:
    """Build a frame-specific AGAM target avatar from a fitted mesh."""
    centers = torch.from_numpy(mesh.vertices[template.anchor_indices].astype(np.float32))
    colors = torch.from_numpy(_anchor_colors(mesh.vertices, mesh.faces, template.anchor_indices, template.geodesic_anchor_indices))
    scales = torch.from_numpy(_anchor_scales(mesh.vertices, mesh.faces, template.anchor_indices))
    opacities = torch.full((template.anchor_indices.shape[0],), 0.95, dtype=torch.float32)
    rotations = torch.zeros((template.anchor_indices.shape[0], 4), dtype=torch.float32)
    rotations[:, 0] = 1.0
    return TorchGaussianAvatar(
        centers=centers,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )


def target_avatar_to_render_avatar(avatar: TorchGaussianAvatar) -> GaussianAvatar:
    """Convert a target avatar to the rasterizer-friendly GaussianAvatar contract."""
    return GaussianAvatar(
        centers=avatar.centers.detach().cpu().numpy().astype(np.float32),
        colors=avatar.colors.detach().cpu().numpy().astype(np.float32),
        opacities=avatar.opacities.detach().cpu().numpy().astype(np.float32),
        scales=avatar.scales.detach().cpu().numpy().astype(np.float32),
    )


def stack_torch_avatars(avatars: list[TorchGaussianAvatar]) -> TorchGaussianAvatar:
    """Stack a list of avatar targets into one batch."""
    if not avatars:
        raise ValueError("avatars must not be empty")
    centers = torch.stack([avatar.centers for avatar in avatars], dim=0)
    colors = torch.stack([avatar.colors for avatar in avatars], dim=0)
    opacities = torch.stack([avatar.opacities for avatar in avatars], dim=0)
    scales = torch.stack([avatar.scales for avatar in avatars], dim=0)
    rotations = torch.stack([avatar.rotations for avatar in avatars], dim=0)
    return TorchGaussianAvatar(
        centers=centers,
        colors=colors,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
    )


def move_torch_avatar(avatar: TorchGaussianAvatar, device: torch.device) -> TorchGaussianAvatar:
    """Move all avatar tensors to one device."""
    return TorchGaussianAvatar(
        centers=avatar.centers.to(device),
        colors=avatar.colors.to(device),
        opacities=avatar.opacities.to(device),
        scales=avatar.scales.to(device),
        rotations=avatar.rotations.to(device),
    )


def _anchor_colors(vertices: np.ndarray, faces: np.ndarray, anchors: np.ndarray, geodesic_anchors: np.ndarray) -> np.ndarray:
    """Create stable correspondence colors for anchor vertices."""
    distances = geodesic_anchor_distances(vertices, faces, geodesic_anchors)
    colors = geodesic_rgb_embedding(distances)
    return colors[anchors].astype(np.float32)


def _anchor_scales(vertices: np.ndarray, faces: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Estimate a stable isotropic radius for each anchor."""
    neighbors = _vertex_neighbors(faces, vertices.shape[0])
    global_scale = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) / max(float(vertices.shape[0]) ** (1.0 / 3.0), 1.0)
    scales = np.empty(anchors.shape[0], dtype=np.float32)
    for row, anchor in enumerate(anchors):
        attached = neighbors[int(anchor)]
        if not attached:
            scales[row] = max(global_scale * 0.05, 1e-3)
            continue
        lengths = np.linalg.norm(vertices[np.asarray(attached)] - vertices[int(anchor)], axis=1)
        scales[row] = max(float(lengths.mean()) * 0.6, 1e-3)
    return scales


def _vertex_neighbors(faces: np.ndarray, vertex_count: int) -> list[list[int]]:
    """Build a vertex adjacency list from triangle faces."""
    neighbors: list[set[int]] = [set() for _ in range(vertex_count)]
    for face in faces[:, :3]:
        a, b, c = [int(index) for index in face]
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [sorted(list(item)) for item in neighbors]
