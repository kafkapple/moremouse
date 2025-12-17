# MoReMouse Implementation Complete Report

**Date**: 2025-12-10
**Version**: 1.0
**Reference**: arXiv:2507.04258v2
**Status**: ✅ All components implemented

---

## 1. Implementation Summary

MoReMouse 프로젝트의 모든 핵심 컴포넌트가 **100% 구현 완료**되었습니다.

### Data Source (Paper)
| Item | Specification |
|------|---------------|
| **Dataset** | markerless_mouse_1 (Dunn et al., 2021) |
| **Camera System** | 6-view synchronized cameras |
| **Avatar Training** | 800 frames (from first 8000 frames) |
| **Species** | C57BL/6 laboratory mice |
| **Note** | "C57BL/6 mice usually show similar texture"

| Category | Status | Components |
|----------|--------|------------|
| **Models** | ✅ 100% | GaussianAvatar, MouseBodyModel, TriplaneGenerator, MoReMouse |
| **Data Loaders** | ✅ 100% | SyntheticDataset, MAMMALMultiviewDataset, VideoReader |
| **Loss Functions** | ✅ 100% | MSE, LPIPS, SSIM, Mask, Depth, Geodesic |
| **Scripts** | ✅ 100% | train, inference, evaluate, generate_synthetic_data, run_pipeline |
| **Configs** | ✅ 100% | model, data, train, avatar (Hydra-based) |

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MoReMouse Full Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage 1: AGAM Training                Stage 2: Synthetic Data          │
│  ┌──────────────────────┐              ┌──────────────────────┐         │
│  │  Multi-view Videos   │───────────▶  │  Gaussian Avatar     │         │
│  │  (6-8 cameras)       │  400K iter   │  + Random Poses      │         │
│  └──────────────────────┘              └──────────┬───────────┘         │
│                                                   │                      │
│                                                   ▼                      │
│                                        ┌──────────────────────┐         │
│                                        │  12K scenes × 64     │         │
│                                        │  views = 768K images │         │
│                                        └──────────┬───────────┘         │
│                                                   │                      │
│                                                   ▼                      │
│  Stage 3: MoReMouse Training           Stage 4: Inference               │
│  ┌──────────────────────┐              ┌──────────────────────┐         │
│  │  NeRF (60 epochs)    │───────────▶  │  Single Image        │         │
│  │  DMTet (100 epochs)  │              │  → Novel Views + 3D  │         │
│  └──────────────────────┘              └──────────────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Implemented Modules

### 3.1 Models (`src/models/`)

#### GaussianAvatar (`gaussian_avatar.py`)
UV 기반 Gaussian Splatting으로 합성 데이터 생성

```python
class GaussianAvatar(nn.Module):
    """
    Parameters per Gaussian (13,059 total):
    - position_offsets: [V, 3] 메쉬 정점 기준 오프셋
    - colors: [V, 3] RGB 색상
    - opacities: [V, 1] 불투명도
    - scales: [V, 3] Gaussian 스케일
    - rotations: [V, 4] Quaternion 회전

    Methods:
    - forward(pose, bone_lengths, ...) → Gaussian parameters
    - render(viewmat, K, ...) → Rendered image
    """

class GaussianAvatarTrainer:
    """
    Features:
    - L1 + SSIM + LPIPS loss
    - Auto-checkpoint detection & resume
    - 400K iteration training (paper spec)
    """
```

#### MouseBodyModel (`mouse_body.py`)
MAMMAL 기반 140 관절 articulated body model

```python
class MouseBodyModel(nn.Module):
    """
    Specifications:
    - 140 joints with hierarchical structure
    - 13,059 vertices
    - Linear Blend Skinning (LBS) deformation
    - ZYX Euler angle rotation convention

    Methods:
    - forward(euler_angles, bone_lengths, translation, scale)
    - get_keypoints_22() → 22 semantic keypoints
    """
```

#### TriplaneGenerator (`triplane.py`)
Transformer 기반 3D feature encoding

```python
class TriplaneGenerator(nn.Module):
    """
    Architecture (Paper Table A3):
    - Input: [B, N, 768] DINOv2 patch features
    - 12-layer transformer decoder with Flash Attention
    - Learnable queries: 64×64 (memory efficient)
    - Upsampler: 64×64 → 128×128
    - Output channels: 80
    """

class TriplaneDecoder(nn.Module):
    """
    - Bilinear interpolation grid sampling
    - 10 shared hidden layers (64 neurons)
    - Multi-head output: RGB, density, geodesic embedding
    """
```

#### MoReMouse Network (`moremouse_net.py`)
End-to-end reconstruction network

```python
class MoReMouse(nn.Module):
    """
    Components:
    1. DINOv2 encoder (frozen, 768-dim)
    2. TriplaneGenerator (transformer-based)
    3. TriplaneDecoder (multi-head MLP)
    4. NeRF renderer (128 samples/ray)

    Paper-compliant settings:
    - num_samples: 128
    - radius: 0.87
    - near: 0.1, far: 4.0
    - Chunked rendering: 4096 rays/chunk
    """
```

### 3.2 Data Loaders (`src/data/`)

#### MAMMALMultiviewDataset (`mammal_loader.py`)
Multi-view 데이터 로딩 (Video + Image 형식 자동 감지)

```python
class VideoReader:
    """LRU cache (100 frames) for efficient video frame access"""

class MAMMALMultiviewDataset(Dataset):
    """
    Supported formats:
    1. Image-based: cam0/frame_XXXXXX.png
    2. Video-based: videos_undist/0.mp4 (MAMMAL nerf format)

    Features:
    - Auto-detection of data format
    - Flexible camera selection
    - Robust naming convention handling
    - Default calibration generation (8 cameras on sphere)
    """
```

#### SyntheticDataset (`dataset.py`)
합성 데이터 학습용 Dataset

```python
class SyntheticDataset(Dataset):
    """
    Output dict:
    - input_image: [3, H, W]
    - target_images: [V, 3, H, W]
    - viewmats: [V, 4, 4]
    - Ks: [V, 3, 3]
    - pose: [J*3]
    - embedding: [V, 3] (geodesic)
    """
```

### 3.3 Loss Functions (`src/losses/`)

| Loss | File | Formula | Weight (λ) |
|------|------|---------|------------|
| MSE | `reconstruction.py` | `||I_pred - I_gt||²` | 1.0 |
| L1 | `reconstruction.py` | `||I_pred - I_gt||₁` | - |
| SSIM | `reconstruction.py` | `1 - SSIM(I_pred, I_gt)` | - |
| LPIPS | `reconstruction.py` | `VGG perceptual` | 1.0 |
| Mask | `mask.py` | `BCE(α_pred, α_gt)` | 0.3 |
| Depth | `depth.py` | `Scale-invariant L1` | 0.2 |
| Geodesic | `geodesic.py` | `||E_pred - E_gt||²` | 0.1 |

**Combined Loss** (`combined.py`):
```python
L_total = λ_mse·L_mse + λ_lpips·L_lpips + λ_mask·L_mask +
          λ_smooth·L_smooth + λ_depth·L_depth + λ_geo·L_geo
```

### 3.4 Utilities (`src/utils/`)

| Module | Functions |
|--------|-----------|
| `logging.py` | setup_logging(), get_logger() |
| `metrics.py` | compute_psnr(), compute_ssim(), compute_lpips() |
| `visualization.py` | Multi-view grid, mesh rendering, colormaps |

---

## 4. Scripts

| Script | Purpose | Key Arguments |
|--------|---------|---------------|
| `train.py` | Two-stage training | Hydra config overrides |
| `inference.py` | Single image/video inference | `--image`, `--video`, `--checkpoint` |
| `evaluate.py` | Metrics evaluation | `--checkpoint`, `--dataset` |
| `generate_synthetic_data.py` | Avatar-based data generation | `--num-frames`, `--num-views` |
| `run_pipeline.py` | End-to-end pipeline | `--stage {avatar,synthetic,train,evaluate,visualize,all}` |

---

## 5. Configuration System (Hydra)

### Directory Structure
```
configs/
├── config.yaml          # Main config (paths, experiment, logging)
├── model/
│   └── moremouse.yaml   # Model architecture (Paper Table A3)
├── data/
│   ├── synthetic.yaml   # Synthetic dataset config
│   └── real.yaml        # Real dataset config
├── train/
│   └── default.yaml     # Training hyperparameters
└── avatar/
    └── default.yaml     # Avatar training config
```

### Paper-Compliant Settings (`model/moremouse.yaml`)

```yaml
encoder:
  name: dinov2_vitb14
  image_size: 378
  feature_dim: 768
  frozen: true

triplane:
  resolution: 128
  channels: 512
  output_channels: 80
  num_layers: 12
  num_heads: 16
  head_dim: 64

decoder:
  hidden_layers: 10
  hidden_dim: 64

nerf:
  num_samples: 128
  radius: 0.87
  near: 0.1
  far: 4.0

avatar:
  num_gaussians_per_vertex: 1  # Total: 13,059
```

---

## 6. Training Pipeline

### Stage 1: NeRF Training (60 epochs)
- Volumetric rendering for smooth gradient propagation
- Loss: MSE + LPIPS + Mask + Depth
- Optimizer: AdamW (lr=1e-5, weight_decay=0.01)
- Scheduler: Cosine annealing with 3000 warmup steps

### Stage 2: DMTet Training (100 epochs)
- Explicit surface extraction for geometric detail
- Additional losses: Normal smoothness, Laplacian regularization
- Fine-tune decoder while freezing encoder

---

## 7. Quick Start Commands

```bash
# 1. Environment Setup
conda activate moremouse
export CUDA_VISIBLE_DEVICES=1

# 2. Generate Synthetic Data (Quick test)
python scripts/generate_synthetic_data.py \
    --output data/synthetic \
    --num-frames 50 \
    --num-views 8 \
    --device cuda:0

# 3. Training (Quick test)
python scripts/train.py \
    experiment.name=quick_test \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=5 \
    train.stages.dmtet.epochs=0 \
    logging.use_wandb=false

# 4. Inference
python scripts/inference.py \
    --image input.png \
    --checkpoint checkpoints/latest.pt \
    --num-views 8 \
    --device cuda:0

# 5. Evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset synthetic \
    --device cuda:0
```

---

## 8. Full Pipeline (Paper Settings)

```bash
# Stage 1: Gaussian Avatar (skip if no multi-view data)
python scripts/run_pipeline.py --stage avatar \
    --data-dir ../MAMMAL_mouse/data/markerless_mouse_1 \
    --avatar-iterations 400000

# Stage 2: Synthetic Data Generation
python scripts/run_pipeline.py --stage synthetic \
    --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
    --num-frames 6000 \
    --num-views 64

# Stage 3: MoReMouse Training
python scripts/run_pipeline.py --stage train \
    --nerf-epochs 60 \
    --dmtet-epochs 100

# Stage 4: Evaluation
python scripts/run_pipeline.py --stage evaluate \
    --checkpoint checkpoints/best.pt

# Stage 5: Visualization
python scripts/run_pipeline.py --stage visualize \
    --checkpoint checkpoints/best.pt \
    --image test_input.png
```

---

## 9. Output Structure

```
moremouse/
├── data/
│   └── synthetic/            # Generated synthetic data
│       ├── train/            # Training samples
│       ├── eval/             # Evaluation samples
│       └── metadata.json
├── checkpoints/
│   ├── latest.pt             # Latest checkpoint
│   ├── best.pt               # Best PSNR checkpoint
│   └── avatar/               # Avatar checkpoints
├── outputs/                  # Hydra outputs
│   └── {experiment_name}/
│       ├── config.yaml
│       ├── train.log
│       └── tensorboard/
├── results/
│   ├── inference/            # Inference outputs
│   │   ├── view_XX.png       # Novel views
│   │   ├── rotation.mp4      # 360° video
│   │   └── visualization.png # Comparison grid
│   └── evaluation.json       # Metrics
└── logs/                     # Training logs
```

---

## 10. Pose Data & Avatar Rendering

### 10.1 MAMMAL Fitting Results Location

MAMMAL 포즈 추정 결과는 다음 위치에 저장됩니다:

```
/home/joon/MAMMAL_mouse/results/fitting/<experiment_name>/params/
├── step_1_frame_000000.pkl
├── step_1_frame_000001.pkl
└── ... (프레임별 포즈 파라미터)
```

**포즈 파일 형식 (.pkl)**:
| Key | Shape | Description |
|-----|-------|-------------|
| `thetas` | `[1, 140, 3]` | Joint rotations (axis-angle) |
| `trans` | `[1, 3]` | Global translation |
| `rotation` | `[1, 3]` | Global rotation |
| `bone_lengths` | `[1, 20]` | Bone lengths |
| `scale` | `[1, 1]` | Scale factor |

### 10.2 Avatar Rendering Commands

```bash
# Rest pose (default)
python scripts/render_avatar.py \
    --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --output-dir results/avatar_renders \
    --radius 800

# Single pose from MAMMAL fitting result
python scripts/render_avatar.py \
    --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --pose-file /home/joon/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/params/step_1_frame_000000.pkl \
    --output-dir results/avatar_renders

# Animation sequence from pose directory
python scripts/render_avatar.py \
    --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --pose-dir /home/joon/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20251206_165254/params/ \
    --output-dir results/avatar_animation \
    --video

# 360° rotation video
python scripts/render_avatar.py \
    --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --num-views 72 \
    --output-dir results/avatar_renders \
    --video
```

### 10.3 Coordinate System Notes

- **Body model**: Y-up coordinate system (MAMMAL convention)
- **Rendering**: Z-up coordinate system (converted via -90° X rotation)
- **world_scale**: 160.0 (model coords ~0.5m → world coords ~80mm)

---

## 11. Key Technical Features

### 11.1 Memory Optimization
- **Flash Attention**: O(n) memory for triplane generation via `F.scaled_dot_product_attention`
- **Chunked Rendering**: 4096 rays per chunk to fit in GPU memory
- **Video Reader LRU Cache**: Efficient frame caching (100 frames default)

### 10.2 Robustness Features
- **Auto-checkpoint Detection**: Automatically resume from latest checkpoint
- **Fallback Systems**:
  - Default camera calibration if missing
  - Dijkstra fallback if potpourri3d unavailable
- **Flexible Data Format Detection**: Video vs image-based datasets

### 10.3 Paper Compliance
- ✅ All Table A3 architecture specifications
- ✅ Loss weights (λ_mse=1.0, λ_lpips=1.0, λ_mask=0.3, λ_depth=0.2, λ_geo=0.1)
- ✅ Training schedule (NeRF 60 → DMTet 100 epochs)
- ✅ NeRF parameters (128 samples, radius=0.87)

---

## 11. Dependencies

### Core
- PyTorch 2.0.1 + CUDA 11.8
- Python 3.10+

### Models
- timm (DINOv2)
- transformers (alternative DINOv2 loading)
- gsplat (Gaussian splatting)
- kaolin (DMTet, optional)

### Data Processing
- OpenCV
- numpy, scipy
- trimesh, networkx

### Training
- hydra-core, omegaconf
- einops
- wandb (optional)
- lpips

### Optional
- potpourri3d (geodesic distance computation)

---

## 12. Expected Performance

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|--------|--------|---------|
| Synthetic | 22.03 | 0.966 | 0.053 |
| Real | 18.42 | - | - |

---

*Generated: 2025-12-10*
*MoReMouse Implementation v1.0 - Complete*
