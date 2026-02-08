#!/usr/bin/env python
"""Debug rendering to identify white image issue."""

import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
from src.models.gaussian_avatar import GaussianAvatarTrainer
from src.models.mouse_body import load_mouse_model

parser = argparse.ArgumentParser(description="Debug rendering")
parser.add_argument('--mouse-model', type=str,
                    default=os.environ.get('MOUSE_MODEL_DIR'),
                    help='Path to mouse model (env: MOUSE_MODEL_DIR)')
args = parser.parse_args()

if args.mouse_model is None:
    parser.error("--mouse-model is required (or set MOUSE_MODEL_DIR env var)")

device = torch.device("cuda:0")
body_model = load_mouse_model(args.mouse_model, device=device)
trainer, iteration = GaussianAvatarTrainer.from_checkpoint(
    "checkpoints/avatar/avatar_final.pt", body_model, device=device
)

avatar = trainer.avatar
num_joints = body_model.num_joints
pose = torch.zeros(1, num_joints * 3, device=device)

with torch.no_grad():
    params = avatar(pose)

    print(f"params keys: {list(params.keys())}")
    for k in params.keys():
        print(f"  {k}: shape={params[k].shape}")

    # Check trainer's _world_scale
    world_scale = getattr(trainer, '_world_scale', 160.0)
    print(f"\nworld_scale from trainer: {world_scale}")

    # Apply world_scale like _save_visualization does
    if world_scale != 1.0:
        params["means"] = params["means"] * world_scale
        params["scales"] = params["scales"] * world_scale
        print(f"After scaling - means range: ({params['means'].min():.2f}, {params['means'].max():.2f})")

    # Apply Y-up to Z-up coordinate transform
    import math
    means = params["means"]
    x, y, z = means[..., 0], means[..., 1], means[..., 2]
    params["means"] = torch.stack([x, z, -y], dim=-1)
    print(f"After Y-up to Z-up - means: ({params['means'].min():.2f}, {params['means'].max():.2f})")

    # Transform quaternions
    quats = params["rotations"]
    base_quat = torch.tensor([math.cos(-math.pi/4), math.sin(-math.pi/4), 0, 0],
                              dtype=quats.dtype, device=device)
    base_quat = base_quat.view(1, 1, 4).expand(quats.shape[0], quats.shape[1], 4)
    # quaternion multiply
    w1, x1, y1, z1 = base_quat[..., 0], base_quat[..., 1], base_quat[..., 2], base_quat[..., 3]
    w2, x2, y2, z2 = quats[..., 0], quats[..., 1], quats[..., 2], quats[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
    qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
    qz = w1*z2 + x1*y2 - y1*x2 + z1*w2
    params["rotations"] = torch.stack([w, qx, qy, qz], dim=-1)

    # Model center after transform
    mean_center = params["means"].mean(dim=1)
    print(f"Model center: {mean_center[0].cpu().numpy()}")

    # Camera matching training: at z=500 (for 378x378 image, f~600)
    # After world_scale, model is ~80mm scale. Camera at ~500mm gives good view
    viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    viewmat[0, 2, 3] = 500.0  # Camera at Z=500

    # Intrinsics for 512x512 image with focal matching training (~1.5*width)
    f = 512 * 1.2  # ~614
    K = torch.tensor([[f, 0, 256], [0, f, 256], [0, 0, 1]], dtype=torch.float32, device=device).unsqueeze(0)

    print(f"\nCamera: viewmat[2,3]={viewmat[0,2,3].item()}, f={f}")

    try:
        rgb, alpha = avatar.render(params, viewmat, K, 512, 512)
        print(f"\nRender SUCCESS")
        print(f"rgb shape: {rgb.shape}, range: ({rgb.min():.4f}, {rgb.max():.4f})")
        print(f"alpha shape: {alpha.shape}, range: ({alpha.min():.4f}, {alpha.max():.4f})")
        print(f"alpha non-zero: {(alpha > 0.01).sum().item()}")

        rgb_np = (rgb[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite("results/avatar_renders/debug_raw.png", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
        print(f"\nSaved: results/avatar_renders/debug_raw.png")

    except Exception as e:
        print(f"\nRender FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
