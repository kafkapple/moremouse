# Author-Level MoReMouse Reproduction Plan

Date: 2026-05-22

## Source Status

Public search found the arXiv paper, project page, and AAAI PDF, but no public author GitHub repository. This plan therefore treats the paper as the implementation SSOT and aims to reproduce the author-level system by matching modules, data scale, losses, schedules, and artifacts.

Primary references:

- arXiv: <https://arxiv.org/abs/2507.04258>
- Project page: <https://zyyw-eric.github.io/MoreMouse-webpage/>
- AAAI PDF: <https://ojs.aaai.org/index.php/AAAI/article/download/38360/42322>

## Current State

- Implemented and tested local full-stack path:
  - DINOv2 ViT-B/14 tokenizer wrapper with cached frozen tokens.
  - Transformer image-to-triplane decoder.
  - Triplane field module with density/color/deformation heads.
  - Geodesic correspondence embedding.
  - 64-view Gaussian/geodesic/normal artifact generation.
  - DMTet-style marching tetrahedra extraction proxy.
  - 6-view reconstruction report on markerless mouse frames.
- Verified:
  - `pytest`: 28 passed.
  - coding principles: PASS.
  - gpu03 run: DINOv2 cache used, best coeff MSE `2.46e-12`.

## Dense-View Visualization Meaning

The current dense-view preview is not a real six-camera video frame and not a final photorealistic AGAM render. It is a synthetic supervision diagnostic:

- A fitted MAMMAL mesh frame is normalized to unit scale.
- 64 virtual cameras are sampled on an orbit sphere.
- Each virtual camera renders three supervision channels:
  - Gaussian proxy image from surface triangle centers.
  - Geodesic correspondence color image.
  - Normal map image.
- The preview shows the first 8 of 64 views.
- The strong artificial colors are expected because the current Gaussian colors use semantic geodesic colors, not learned fur appearance.

## Gap to Author-Level Implementation

### AGAM

Current: deterministic surface Gaussian proxy.

Target:

- UV-valid texel to Gaussian mapping.
- Pose-conditioned UV position map.
- StyleUNet-like predictors for:
  - position offset
  - RGB
  - opacity
  - anisotropic rotation
  - anisotropic scale
- Losses:
  - L1 RGB
  - SSIM
  - LPIPS
  - TV smoothness
- Data:
  - about 800 uniformly sampled frames from first 8,000 frames.
  - sparse real training views.

### Dense Synthetic Dataset

Current: one frame, 64 views, 256px diagnostic triplets.

Target:

- Train pose range: first 6,000 frames.
- Two independent 64-view sets per train pose.
- About 12,000 multi-view scenes.
- Resolution: 800x800.
- Camera sphere radius: 2.22.
- FoV: 29.86 degrees.
- Supervision:
  - RGB
  - alpha/mask
  - normal
  - geodesic correspondence feature
  - camera metadata
  - depth if available from renderer

### MoReMouse Network

Current:

- DINOv2 token wrapper.
- Compact transformer-triplane decoder.
- Mesh-PCA proxy output for tractable evaluation.

Target:

- DINOv2-base, 378x378 input, 768 channels.
- 64x64 triplane tokens, 512 channels.
- Transformer backbone: 12 layers, 16 heads, head dimension 64.
- Triplane upsampler: 3 x 80 x 128 x 128.
- Multi-head MLP:
  - density
  - RGB
  - feature embedding
  - deformation
- Volumetric renderer:
  - 128 samples per ray.
  - mask/opacity BCE.
  - RGB/feature MSE.
  - depth consistency in object mask.
  - Smooth L1 RGB discrepancy.
  - LPIPS.

### DMTet

Current: standalone marching tetrahedra proxy.

Target:

- 256^3 tetrahedral grid.
- DMTet fine-tuning for 100 epochs.
- Geometry, texture, feature, normal, mask losses.
- Export:
  - final mesh
  - multiview renders
  - evaluation metrics

## Multi-Day Training Plan

### Phase 0: Resource Gate

Before any multi-day job:

- Check gpu03 GPUs, free VRAM, free disk.
- Do not start if `/home/joon` or selected result disk has less than 500 GB free.
- Use only conda env `moremouse`.
- Save all long-run artifacts under:
  - gpu03: `/home/joon/results/MoReMouse`
  - local mirror: `/Users/joon/results/MoReMouse`

### Phase 1: AGAM Author-Scale Pretraining

Goal: replace deterministic Gaussian proxy with learned AGAM.

Expected duration: 1-3 days depending on resolution and LPIPS.

Deliverables:

- `src/moremouse/models/agam.py`
- `scripts/train_agam.py`
- `configs/experiments/agam_markerless_mouse_1.yaml`
- checkpoint snapshots every N steps.
- HTML grids:
  - input sparse view
  - held-out real view render
  - Gaussian splat image
  - mask/normal/geodesic overlays

### Phase 2: Dense Dataset Rendering

Goal: render author-scale synthetic dataset.

Expected duration: hours to 1 day depending on final resolution.

Deliverables:

- `scripts/render_dense_dataset.py`
- manifest with frame, pose id, camera id, channels, paths.
- 64-view previews per selected frame.
- storage audit report.

### Phase 3: Triplane NeRF Training

Goal: train DINOv2 + transformer-triplane renderer on dense synthetic data.

Expected duration: 2-5 days.

Deliverables:

- `src/moremouse/rendering/volumetric.py`
- `scripts/train_triplane_nerf.py`
- checkpoint metrics:
  - PSNR
  - SSIM
  - LPIPS
  - mask IoU
  - feature MSE
- 6-view and 64-view HTML grids.

### Phase 4: DMTet Fine-Tuning

Goal: convert trained triplane field into high-resolution mesh.

Expected duration: 1-2 days.

Deliverables:

- `scripts/train_dmtet.py`
- final OBJ/PLY/GLB exports.
- 6-view render comparison.
- visual-hull IoU if visual hull generation is available.

### Phase 5: Evaluation and Report

Goal: match paper tables as far as local data allows.

Deliverables:

- PSNR/SSIM/LPIPS tables.
- visual-hull IoU table.
- qualitative comparisons with:
  - current MAMMAL best-source mesh
  - local full-stack MVP
  - DMTet output
- final embedded HTML report.

## Immediate Next Actions

1. Implement `AGAM` model skeleton with explicit Gaussian parameter heads.
2. Add a sparse-view training dataset loader from markerless videos and MAMMAL fits.
3. Add SSIM and LPIPS dependencies to gpu03 env only after checking install compatibility.
4. Run a 20-frame AGAM pilot for 2-4 hours.
5. Inspect held-out view renders before scaling to 800 frames.

## Stop Conditions

Stop and ask before launching a multi-day run if:

- gpu03 has insufficient free disk or GPU availability.
- LPIPS/xformers/diff-gaussian-rasterization install requires system CUDA changes.
- AGAM pilot held-out render is visibly worse than current mesh projection baseline.
- MAMMAL pose/mesh source changes are needed.
