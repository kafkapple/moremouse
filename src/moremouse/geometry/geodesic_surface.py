"""Mesh geodesic correspondence embeddings."""

from __future__ import annotations

from heapq import heappop, heappush

import numpy as np
from jaxtyping import Float, Int


def build_edge_graph(vertices: Float[np.ndarray, "vertices 3"],
                     faces: Int[np.ndarray, "faces corners"]) -> list[list[tuple[int, float]]]:
    """Build a weighted undirected graph from mesh triangle edges."""
    graph: list[list[tuple[int, float]]] = [[] for _ in range(vertices.shape[0])]
    edges: set[tuple[int, int]] = set()
    for face in faces[:, :3]:
        ids = [int(index) for index in face]
        for start, end in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
            left, right = sorted((start, end))
            if (left, right) in edges:
                continue
            edges.add((left, right))
            length = float(np.linalg.norm(vertices[left] - vertices[right]))
            graph[left].append((right, length))
            graph[right].append((left, length))
    return graph


def farthest_point_anchors(vertices: Float[np.ndarray, "vertices 3"], count: int) -> Int[np.ndarray, "anchors"]:
    """Select deterministic Euclidean farthest-point anchors."""
    if count < 1:
        raise ValueError("anchor count must be positive")
    anchors = [int(np.argmin(vertices[:, 0]))]
    distances = np.linalg.norm(vertices - vertices[anchors[0]], axis=1)
    for _ in range(1, min(count, vertices.shape[0])):
        index = int(np.argmax(distances))
        anchors.append(index)
        distances = np.minimum(distances, np.linalg.norm(vertices - vertices[index], axis=1))
    return np.asarray(anchors, dtype=np.int32)


def geodesic_anchor_distances(vertices: Float[np.ndarray, "vertices 3"],
                              faces: Int[np.ndarray, "faces corners"],
                              anchors: Int[np.ndarray, "anchors"]) -> Float[np.ndarray, "vertices anchors"]:
    """Compute shortest-path geodesic distances from selected anchors."""
    graph = build_edge_graph(vertices, faces)
    columns = [_finite_distances(_dijkstra(graph, int(anchor)), vertices, int(anchor)) for anchor in anchors]
    distances = np.stack(columns, axis=1).astype(np.float32)
    if not np.isfinite(distances).all():
        raise ValueError("geodesic distances contain non-finite values")
    return distances


def geodesic_rgb_embedding(distances: Float[np.ndarray, "vertices anchors"]) -> Float[np.ndarray, "vertices 3"]:
    """Map geodesic anchor distances to stable RGB-like correspondence colors."""
    centered = distances - distances.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    channels = centered @ vt[:3].T
    minimum = channels.min(axis=0, keepdims=True)
    maximum = channels.max(axis=0, keepdims=True)
    scale = np.maximum(maximum - minimum, np.finfo(np.float32).eps)
    return ((channels - minimum) / scale).astype(np.float32)


def _dijkstra(graph: list[list[tuple[int, float]]], source: int) -> Float[np.ndarray, "vertices"]:
    """Run Dijkstra from one source vertex."""
    distances = np.full(len(graph), np.inf, dtype=np.float64)
    distances[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        distance, node = heappop(heap)
        if distance > distances[node]:
            continue
        for neighbor, weight in graph[node]:
            candidate = distance + weight
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                heappush(heap, (candidate, neighbor))
    return distances.astype(np.float32)


def _finite_distances(distances: np.ndarray, vertices: np.ndarray, anchor: int) -> np.ndarray:
    """Replace disconnected graph distances with Euclidean fallback distances."""
    finite = np.isfinite(distances)
    if finite.all():
        return distances
    distances = distances.copy()
    distances[~finite] = np.linalg.norm(vertices[~finite] - vertices[anchor], axis=1)
    return distances
