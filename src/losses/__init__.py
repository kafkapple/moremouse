"""
MoReMouse Loss Functions

Contains all loss functions used in training:
- Reconstruction losses (MSE, L1, SSIM, LPIPS)
- Mask loss
- Depth loss
- Geodesic embedding loss
"""

from .reconstruction import (
    MSELoss,
    L1Loss,
    SSIMLoss,
    LPIPSLoss,
    SmoothL1Loss,
)
from .mask import MaskLoss
from .depth import DepthLoss
from .geodesic import GeodesicLoss
from .combined import MoReMouseLoss

__all__ = [
    "MSELoss",
    "L1Loss",
    "SSIMLoss",
    "LPIPSLoss",
    "SmoothL1Loss",
    "MaskLoss",
    "DepthLoss",
    "GeodesicLoss",
    "MoReMouseLoss",
]
