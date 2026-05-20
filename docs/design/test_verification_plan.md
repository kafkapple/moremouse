# Test and Verification Plan

## Unit Tests

- Config loader freezes OmegaConf and resolves paths.
- Dataset manifest rejects missing paths and duplicate frame/view keys.
- Geodesic embedding rejects non-finite tensors and wrong shapes.
- Camera sampler returns normalized camera centers at configured radius.
- Result path resolver always stays under configured result root.

## Smoke Tests

- Load one dataset manifest.
- Build a tiny synthetic camera set.
- Run geodesic embedding on a toy mesh graph.
- Save a 2x2 visualization grid.
- Build the markerless/MAMMAL fitting asset manifest on gpu03:

```bash
cd /home/joon/dev/MoReMouse
PYTHONPATH=src /home/joon/anaconda3/bin/conda run -n moremouse python scripts/build_mammal_asset_manifest.py
```

## GPU03 Preflight

Before any long job:

```bash
ssh gpu03 'df -h / /home /node_data 2>/dev/null; nvidia-smi'
```

Required:

- at least 100GB free for smoke rendering
- at least one GPU with enough free memory for the selected stage
- conda env `moremouse`, not base

## Training Gates

### Gate A: Single-batch Sanity

- one batch loads
- tensor shapes match model contract
- no NaN/Inf in inputs or losses
- config snapshot is written

### Gate B: Tiny Overfit

- overfit a tiny subset
- PSNR improves from initial baseline
- image grid and video generated

### Gate C: Held-out Views

- compute PSNR, SSIM, LPIPS, mask IoU where available
- save novel-view grid
- save turntable video
