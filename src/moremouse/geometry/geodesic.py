"""Geodesic correspondence embedding utilities."""

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def validate_embedding(embedding: FloatArray) -> None:
    """Validate a vertex embedding array.

    Parameters
    ----------
    embedding:
        Array with shape ``(num_vertices, channels)``.

    Raises
    ------
    ValueError
        If shape or finite-value constraints are violated.
    """
    if embedding.ndim != 2:
        raise ValueError(f"Embedding must be rank 2, got shape {embedding.shape}")
    if embedding.shape[0] < 2 or embedding.shape[1] < 1:
        raise ValueError(f"Embedding shape is too small: {embedding.shape}")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or Inf")


def compute_embedding_distances(embedding: FloatArray) -> FloatArray:
    """Compute pairwise Euclidean distances in embedding space.

    Parameters
    ----------
    embedding:
        Float array with shape ``(num_vertices, channels)``.

    Returns
    -------
    FloatArray
        Pairwise distance matrix with shape ``(num_vertices, num_vertices)``.
    """
    validate_embedding(embedding)
    delta = embedding[:, None, :] - embedding[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    if not np.isfinite(distances).all():
        raise ValueError("Computed embedding distances contain NaN or Inf")
    return distances

