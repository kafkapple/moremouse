
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.mammal_loader import MAMMALMultiviewDataset
import cv2
import numpy as np
import torch
import os

def test_loader():
    parser = argparse.ArgumentParser(description="Test canonical loader")
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get('NERF_DATA_DIR'),
                        help='Path to NeRF capture data (env: NERF_DATA_DIR)')
    args = parser.parse_args()

    if args.data_dir is None:
        parser.error("--data-dir is required (or set NERF_DATA_DIR env var)")

    data_dir = args.data_dir
    output_dir = "outputs/test_canonical_loader"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Initializing dataset from {data_dir}...")
    dataset = MAMMALMultiviewDataset(
        data_dir=data_dir,
        num_frames=10,
        canonical_mode=True,
        crop_scale=1.5,
        world_scale=160.0
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    sample_idx = 0
    print(f"Loading sample {sample_idx}...")
    sample = dataset[sample_idx]
    
    # Check keys
    print("Keys in sample:", sample.keys())
    
    # Visualize
    images = sample['images'] # [V, 3, H, W]
    viewmats = sample['viewmats'] # [V, 4, 4]
    Ks = sample['Ks'] # [V, 3, 3]
    
    num_views = images.shape[0]
    print(f"Loaded {num_views} views")
    
    for v in range(num_views):
        img_tensor = images[v]
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Save crop
        cv2.imwrite(f"{output_dir}/view_{v:02d}_crop.png", img_bgr)
        
        # Verify camera lookat
        # Canonical camera should look roughly at origin (0,0,0)
        viewmat = viewmats[v].numpy()
        # Camera pos in world (canonical world)
        R = viewmat[:3, :3]
        t = viewmat[:3, 3]
        pos = -R.T @ t
        
        # Look vector (inverse of Z axis of camera)
        # Cam Z is backward, so -Z is forward.
        # cam_z_world = R.T @ [0,0,1]
        forward = - (R.T @ np.array([0,0,1]))
        
        # Check angle between forward and vector to origin
        to_origin = -pos
        disp = np.linalg.norm(to_origin)
        to_origin_norm = to_origin / (disp + 1e-8)
        
        dot = np.dot(forward, to_origin_norm)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        
        print(f"View {v}: Pos={pos}, Dist={disp:.3f}, LookAt angle error={angle:.2f} deg")
        
    print("Done!")

if __name__ == "__main__":
    test_loader()
