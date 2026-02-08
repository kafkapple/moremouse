# MoReMouse 논문 vs 구현 감사 보고서

**Date**: 2026-02-08
**Paper**: [MoReMouse (arXiv:2507.04258v2)](https://arxiv.org/html/2507.04258v2)
**Related**: [Refactoring Report](../refactoring/260208_refactoring_report.md) | [Pipeline Overview](../pipeline_overview.md) | [Coordinate Analysis](251213_moremouse_coordinate_alignment_analysis.md)

---

## Executive Summary

Implementation completeness: **~93%**. Core pipeline (DINOv2 + Triplane + NeRF) fully matches paper. All loss weights match Table A3. DMTet Stage 2 is the only unimplemented module.

---

## 1. Architecture Match

| Component | Paper (Table A3) | Implementation | Match |
|-----------|-----------------|----------------|-------|
| **DINOv2 Encoder** | ViT-B/14, 378×378, 768-dim | `moremouse_net.py:40` | 100% |
| **Triplane Decoder** | 12 layers, 16 heads, 64→128, 80ch | `triplane.py:48-54` | 100% |
| **Multi-head MLP** | 64 neurons, 10 shared layers | `triplane.py:362-363` | 100% |
| **NeRF Rendering** | 128 samples, r=0.87, trunc_exp, bias=-1.0 | `moremouse_net.py:207-210` | 100% |
| **MAMMAL Body** | 140 joints, 13059 verts, axis-angle | `mouse_body.py:48-50` | 100% |
| **Canonical Space** | 1/180 scale, r=2.22, FOV=29.86° | `transforms.py:42-45` | 100% |
| **AGAM Avatar** | StyleUNet + UV Gaussian | Simplified (no StyleUNet) | 90% |
| **DMTet Stage 2** | 100 epochs, SDF bias=-4.0 | Not implemented (skip) | 0% |

## 2. Loss Weight Match

| Loss | Paper | Implementation | File | Match |
|------|-------|----------------|------|-------|
| MSE | 1.0 | 1.0 | `combined.py:44` | O |
| LPIPS | 1.0 | 1.0 | `combined.py:45` | O |
| Mask | 0.3 | 0.3 | `combined.py:46` | O |
| SmoothL1 | 0.2 | 0.2 | `combined.py:47` | O |
| Depth | 0.2 | 0.2 | `combined.py:48` | O |
| Geodesic | 0.1 | 0.1 | `combined.py:49` | O |
| **Avatar** SSIM | 0.2 | 0.2 | `gaussian_avatar.py:415` | O |
| **Avatar** LPIPS | 0.1 | 0.1 | `gaussian_avatar.py:416` | O |
| **Avatar** TV | 0.01 | 0.01 | `gaussian_avatar.py:417` | O |

---

## 3. HIGH Priority Issues

### H1. AGAM: StyleUNet Not Implemented

- **Paper**: StyleUNet CNN predicts pose-conditioned UV offset map `Δμ_Ψl`
- **Implementation**: `gaussian_avatar.py` uses learnable `position_offsets` directly (no CNN)
- **Impact**: Avatar quality may be lower; functional for training pipeline
- **Assessment**: Intentional simplification. StyleUNet needed only for publication-quality avatars

### H2. TV Loss — Mesh Ordering Assumption

- **Location**: `gaussian_avatar.py:494-524`
- **Issue**: `params[i] - params[i+1]` assumes array index = spatial adjacency
- **Risk**: If UV-to-vertex mapping doesn't preserve spatial locality, TV regularization is meaningless
- **Action**: Verify `mouse_body.py` UV mapping order

### H3. Yaw Angle Negation (Undocumented)

- **Location**: `gaussian_avatar.py:847` → `yaw_angle = -angle`
- **Issue**: MAMMAL angle sign is flipped without documentation
- **Risk**: Dataset-specific calibration hardcoded; breaks on different datasets
- **Action**: Add documentation comment explaining calibration origin

### H4. 2D Keypoint Resolution Hardcoded

- **Location**: `mammal_loader.py:1215, 1527, 1577` → `1152×1024`
- **Issue**: MAMMAL video resolution hardcoded instead of read from metadata
- **Risk**: Keypoint scaling fails on different-resolution videos
- **Action**: Read resolution from video file or config

---

## 4. MEDIUM Priority Issues

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| M1 | SSIM window padding | `gaussian_avatar.py` | `window_size=11` may cause padding issues on small crops |
| M2 | Canonical K double-handling | `mammal_loader.py` CanonicalSpaceDataset | Camera intrinsics adjusted twice after crop |
| M3 | Frame ratio interpolation | `mammal_loader.py:444-449` | Nearest-neighbor for frame↔transform mismatch |
| M4 | Triplane bounds unchecked | `moremouse_net.py` | Sample coords outside [-1,1] silently clamped |

---

## 5. DMTet Stage 2 Analysis

### Current State

| Item | Status |
|------|--------|
| Config (`sdf_bias`, `grid_resolution`) | Defined in YAML |
| Training loop | SKIPPED with warning (`train.py:432-440`) |
| SDF Network | Not implemented |
| Marching Tetrahedra | Not implemented (requires Kaolin) |
| Mesh Renderer | Not implemented (requires nvdiffrast) |
| Kaolin dependency | Not installed |

### Impact Without DMTet

| Capability | NeRF Only (Current) | NeRF + DMTet (Paper) |
|------------|---------------------|----------------------|
| Image rendering | Yes | Yes |
| Pose estimation | Yes | Yes |
| 3D mesh output | No | Yes |
| Surface detail | Medium (volumetric) | High (surface) |
| Render speed | Slow (ray marching) | Fast (rasterization) |

### Implementation Scope (if needed)

```
1. DMTet Renderer (~300 lines)
   - SDF MLP: Linear(80,128) → ReLU → Linear(128,1) + sdf_bias=-4.0
   - Deform MLP: Linear(80,128) → ReLU → Linear(128,3)
   - Tetrahedral grid init (kaolin)
   - marching_tetrahedra → vertices, faces

2. Mesh Rendering (~200 lines)
   - nvdiffrast differentiable rasterizer
   - RGB + Silhouette + Normal rendering

3. Training Loop (~100 lines)
   - 100 epochs, AdamW lr=1e-5, CosineAnnealingLR
   - Loss: L_rgb + 0.3×L_mask + 0.1×L_normal + 0.01×L_laplacian
```

**Dependencies**: `kaolin>=0.15.0`, `nvdiffrast`
**Estimated effort**: ~600 lines of code

---

## 6. Simplification Opportunities

| Target | Current | Possible | Savings |
|--------|---------|----------|---------|
| `mammal_loader.py` | 1710 lines | Split MAMMALDataset / CanonicalSpaceDataset | ~500 lines/file |
| `mouse_body.py` Euler code | 46 lines | Remove (axis-angle only used) | ~46 lines |
| DINOv2 timm fallback | 25 lines | Remove if torch.hub reliable | ~25 lines |
| Magic numbers | Scattered | Consolidate to config.yaml | Maintainability |

**Assessment**: Low priority. Phase 1-3 refactoring already eliminated major duplication. Remaining simplifications carry functional risk → defer until after training validation.

---

## 7. Recommended Next Steps

| Priority | Item | Reason |
|----------|------|--------|
| P1 | H2: Verify TV loss mesh ordering | Affects training quality |
| P1 | H4: Remove hardcoded 1152×1024 | Required for other datasets |
| P2 | H3: Document yaw negation | Maintainability (comment only) |
| P3 | DMTet Stage 2 | Full paper reproduction |
| P3 | StyleUNet for AGAM | Avatar quality improvement |

---

*MoReMouse Paper vs Implementation Audit | 2026-02-08*
