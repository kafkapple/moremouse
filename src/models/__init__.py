"""
MoReMouse Models

Contains:
- MouseBodyModel: MAMMAL-based articulated mouse mesh model
- GaussianAvatar: Gaussian mouse avatar (AGAM) for synthetic data generation
- MoReMouse: Main reconstruction network
- TriplaneDecoder: Triplane-based 3D representation decoder
- GeodesicEmbedding: Surface correspondence embeddings
"""

from .mouse_body import MouseBodyModel, load_mouse_model
from .gaussian_avatar import GaussianAvatar, GaussianAvatarTrainer
from .moremouse_net import MoReMouse, DINOv2Encoder
from .triplane import TriplaneDecoder, TriplaneGenerator, MultiHeadMLP
from .geodesic_embedding import GeodesicEmbedding, create_geodesic_embedding

__all__ = [
    "MouseBodyModel",
    "load_mouse_model",
    "GaussianAvatar",
    "GaussianAvatarTrainer",
    "MoReMouse",
    "DINOv2Encoder",
    "TriplaneDecoder",
    "TriplaneGenerator",
    "MultiHeadMLP",
    "GeodesicEmbedding",
    "create_geodesic_embedding",
]
