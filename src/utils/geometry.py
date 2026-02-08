"""
Shared Geometry Utilities

Keypoint extraction, skeleton definitions, and drawing functions
used across multiple scripts and modules.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# MAMMAL Mouse Keypoint Definitions
# =============================================================================

# Mapping from 22-keypoint index to MAMMAL 140-joint indices.
# Keypoints 0-2, 4 are vertex-based (overridden below).
KEYPOINT22_JOINT_MAP: Dict[int, List[int]] = {
    # Spine/body
    3: [64, 65],   # Spine mid
    5: [48, 51],   # Hip
    6: [54, 55],   # Back
    7: [61],       # Tail start
    # Right Front Leg
    8: [79],       # Shoulder
    9: [74],       # Elbow
    10: [73],      # Wrist
    11: [70],      # Paw
    # Left Front Leg
    12: [104],     # Shoulder
    13: [99],      # Elbow
    14: [98],      # Wrist
    15: [95],      # Paw
    # Right Hind Leg
    16: [15],      # Hip
    17: [5],       # Knee
    18: [4],       # Ankle
    # Left Hind Leg
    19: [38],      # Hip
    20: [28],      # Knee
    21: [27],      # Ankle
}

# Skeleton connectivity for 22-keypoint visualization
BONES: List[List[int]] = [
    [0, 2], [1, 2],                            # Head
    [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],   # Spine
    [8, 9], [9, 10], [10, 11], [11, 3],        # Right front leg
    [12, 13], [13, 14], [14, 15], [15, 3],     # Left front leg
    [16, 17], [17, 18], [18, 5],               # Right hind leg
    [19, 20], [20, 21], [21, 5],               # Left hind leg
]

# Keypoint names for labeling
KEYPOINT22_NAMES: List[str] = [
    "Nose", "L_Ear", "R_Ear", "Spine_Mid", "Tail_Base",
    "Hip", "Back", "Tail_Start",
    "RF_Shoulder", "RF_Elbow", "RF_Wrist", "RF_Paw",
    "LF_Shoulder", "LF_Elbow", "LF_Wrist", "LF_Paw",
    "RH_Hip", "RH_Knee", "RH_Ankle",
    "LH_Hip", "LH_Knee", "LH_Ankle",
]


# =============================================================================
# Keypoint Extraction
# =============================================================================

def extract_keypoints22(
    joints_3d: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract 22 semantic keypoints from MAMMAL 140-joint skeleton.

    Args:
        joints_3d: [B, 140, 3] joint positions
        device: Target device for output tensor

    Returns:
        [B, 22, 3] keypoint positions
    """
    B = joints_3d.shape[0]
    keypoints = torch.zeros(B, 22, 3, device=device)

    for kp_idx, joint_ids in KEYPOINT22_JOINT_MAP.items():
        if kp_idx < 22:
            joint_positions = joints_3d[:, joint_ids, :]
            keypoints[:, kp_idx] = joint_positions.mean(dim=1)

    # Vertex-based keypoints (approximated using nearby joints)
    keypoints[:, 0] = joints_3d[:, 64, :]   # Nose
    keypoints[:, 1] = joints_3d[:, 66, :]   # Left ear
    keypoints[:, 2] = joints_3d[:, 67, :]   # Right ear
    keypoints[:, 4] = joints_3d[:, 0, :]    # Tail base

    return keypoints


# =============================================================================
# Drawing Functions
# =============================================================================

def draw_skeleton(
    img: np.ndarray,
    kps_2d: np.ndarray,
    color: Tuple[int, int, int],
    bones: Optional[List[List[int]]] = None,
    radius: int = 4,
) -> None:
    """Draw skeleton on image (in-place).

    Args:
        img: [H, W, 3] BGR image (modified in-place)
        kps_2d: [K, 2] 2D keypoint coordinates
        color: BGR color tuple
        bones: Skeleton connectivity (default: BONES)
        radius: Keypoint circle radius
    """
    try:
        import cv2
    except ImportError:
        return

    if bones is None:
        bones = BONES

    h, w = img.shape[:2]

    for i, j in bones:
        if i < len(kps_2d) and j < len(kps_2d):
            pt1 = tuple(kps_2d[i].astype(int))
            pt2 = tuple(kps_2d[j].astype(int))
            if all(0 <= c < max(h, w) for c in pt1 + pt2):
                cv2.line(img, pt1, pt2, color, 2)

    for k in range(len(kps_2d)):
        pt = tuple(kps_2d[k].astype(int))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(img, pt, radius, color, -1)
