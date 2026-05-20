# 260520 MoReMouse Paper Notes

Source: MoReMouse: Monocular Reconstruction of Laboratory Mouse, arXiv:2507.04258v2, revised 2025-11-23.

Project page: https://zyyw-eric.github.io/MoreMouse-webpage/

## One-line Summary

MoReMouse trains a mouse-specific single-image 3D reconstruction model by first creating an animatable Gaussian mouse avatar, rendering dense synthetic multi-view supervision, and then training a DINOv2 plus transformer triplane model with geodesic surface embeddings.

## Problem

Existing mouse methods usually provide sparse 3D keypoints or optimize a mesh per sequence. The paper targets feedforward dense surface reconstruction of C57BL/6 mice from one image, with stronger temporal/semantic surface consistency than generic single-view 3D models.

## Main Contributions

1. AGAM: an animatable Gaussian avatar of mouse built from sparse-view real videos and an articulated mouse model.
2. Dense-view synthetic dataset: 64-view rendered supervision from AGAM.
3. MoReMouse architecture: DINOv2 image encoder, transformer image-to-triplane decoder, triplane NeRF pretraining, DMTet fine-tuning.
4. Geodesic correspondence embeddings: color-coded semantic channels from surface geodesic distances to stabilize limbs, tail, and other weak-texture regions.

## Method Modules

### 1. Mouse Parametric Model

- Refined articulated mouse mesh from Bolanos et al. / MAMMAL lineage.
- Reported size: 140 joints, 13,059 vertices.
- Parameters:
  - local: joint rotations and bone deformation
  - global: rotation, translation, scale
- LBS drives canonical mouse poses.

### 2. AGAM

AGAM maps valid UV texels to Gaussian points. A pose-conditioned UV position map is used as input to StyleUNet-like networks that predict:

- position offsets
- RGB color
- opacity
- anisotropic Gaussian rotation
- anisotropic Gaussian scale

Training loss combines L1, SSIM, LPIPS, and TV smoothness. The paper reports training on 800 uniformly sampled frames from the first 8,000 frames of markerless mouse 1, then using all poses to render dense views.

### 3. Dense-view Dataset

- Source sequence: markerless mouse 1.
- Train pose range: first 6,000 frames.
- Evaluation pose range: last 6,000 frames.
- Per train frame: two independent 64-view sets, yielding 12,000 multi-view scenes.
- Render settings:
  - camera sphere radius 2.22
  - model scale factor 1/180
  - image resolution 800x800
  - FoV 29.86 degrees
- Synthetic evaluation uses four orthogonal views.

### 4. MoReMouse Network

- Image tokenizer: DINOv2-base, 378x378 input, 768 channels.
- Triplane tokenizer: 64x64 plane tokens, 512 channels.
- Transformer backbone: 12 layers, 16 heads, head dimension 64, cross-attention dimension 768.
- Triplane upsampler output: 3 x 80 x 128 x 128.
- Multi-head MLP: shared hidden layers plus heads for density, feature, and deformation.
- Renderer stage 1: triplane NeRF, 128 samples per ray.
- Renderer stage 2: DMTet, 256^3 isosurface resolution.

### 5. Geodesic Embedding

The paper optimizes a 3D embedding per mesh vertex so Euclidean distances in embedding space approximate surface geodesic distances. It then converts the embedding into HSV-style colors by using PCA components for hue and saturation and fixed value. These channels are rendered and supervised together with RGB.

Implementation note: full all-pairs geodesic on 13k vertices is feasible but memory-heavy. Initial implementation should support landmark or batched approximations, then validate against a small mesh exactly.

## Training Objective

The paper reports a two-stage objective:

- MSE for RGB and feature embeddings
- depth consistency within object mask
- Smooth L1 for RGB discrepancy
- binary cross entropy for opacity/mask
- LPIPS perceptual loss

Training schedule:

- NeRF stage: 60 epochs
- DMTet stage: 100 epochs
- AdamW, learning rate 1e-5
- cosine annealing, 3,000 warm-up steps

## Evaluation

Primary metrics:

- PSNR
- SSIM
- LPIPS
- visual-hull IoU in supplemental tables

Baselines include TripoSR, Triplane-GS, LGM, InstantMesh, 3D Fauna, and TRELLIS.

## Known Failure Modes

- rare upright poses not represented in training
- tail/head self-occlusion
- small regions such as feet and snout distorted by limited resolution

## Local Adaptation Questions

Stop and ask before committing a full training run if any of these are unresolved:

- Which exact MAMMAL mesh/fitting version should be the canonical mesh?
- Whether markerless_mouse_1_nerf on gpu03 is approved as the first source sequence.
- Whether FaceLift M5/M5t2 camera and image data should be treated as real evaluation or adaptation data.
- Where gpu03 should store long-running generated dense-view data if `/home/joon` is not enough.

