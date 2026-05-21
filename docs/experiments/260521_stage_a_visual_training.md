# 260521 Stage A Visual and Training Smoke

## Goal

Validate the markerless/MAMMAL data path before implementing the full MoReMouse dense 3D model:

- RGB videos, segmentation masks, and 2D keypoints are frame-aligned.
- Results are written under the MoReMouse result root.
- gpu03 conda env can run PyTorch CUDA training.
- Visual grid and mp4 outputs are produced for review.

## Commands

```bash
cd /home/joon/dev/MoReMouse
PYTHONPATH=src /home/joon/anaconda3/bin/conda run -n moremouse python scripts/render_markerless_sanity.py
PYTHONPATH=src /home/joon/anaconda3/bin/conda run -n moremouse python scripts/train_tiny_mask_baseline.py
```

## Outputs

Sanity visualization:

- remote: `/home/joon/results/MoReMouse/figures/sanity_260521`
- local mirror: `/Users/joon/results/MoReMouse/figures/sanity_260521`
- frames: `[0, 2000, 6000, 12000, 17980]`
- views: `[0, 1, 2, 3, 4, 5]`
- grid image count: 5
- video: `markerless_sanity.mp4`

Tiny mask baseline:

- remote: `/home/joon/results/MoReMouse/experiments/tiny_mask_baseline_260521`
- local mirror: `/Users/joon/results/MoReMouse/experiments/tiny_mask_baseline_260521`
- device: `cuda:0`
- torch: `2.7.1+cu128`
- frames: `[0, 20, 40, 60, 80, 100]`
- views: `[0, 1, 2, 3, 4, 5]`
- initial loss: `0.6196621159712473`
- final loss: `0.20300880322853723`
- grid: `figures/tiny_mask_predictions.png`
- video: `figures/tiny_mask_predictions.mp4`

MAMMAL mesh preview:

- remote: `/home/joon/results/MoReMouse/figures/mammal_mesh_preview_260521`
- local mirror: `/Users/joon/results/MoReMouse/figures/mammal_mesh_preview_260521`
- frames: `[0, 2000, 6000, 12000, 17980]`
- per-frame mesh size: 14,522 vertices and 28,800 faces
- frame 6000 correctly resolves to the `refit_accurate_23` override asset
- grid: `mammal_mesh_preview_grid.png`

## Verification

- local tests: `16 passed in 0.30s`
- coding principles: `PASS`
- gpu03 tests after pull: `16 passed in 0.21s`
- sanity script completed and wrote `report.json`
- tiny baseline script completed and wrote `report.json`
- after mesh diagnostics were added, local tests: `18 passed in 0.28s`
- after mesh diagnostics were pulled on gpu03, tests: `18 passed in 0.22s`
- mesh preview script completed and wrote `report.json`

## Notes

This is not the final MoReMouse method. It is Gate A validation that the selected markerless mouse data, masks, keypoints, result paths, ffmpeg extraction, and CUDA training loop are coherent enough to proceed.

## Next Gate

Implement the first geometry-aware training dataset:

- sample MAMMAL fitted keyframes from the 900-frame asset manifest
- load OBJ vertices/faces and fitting parameter paths
- attach RGB/mask/keypoint observations by view and frame
- then add the first dense-view/3D representation training objective
