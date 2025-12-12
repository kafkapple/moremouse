---
date: 2025-12-10
context_name: "2_Research"
tags: [ai-assisted, moremouse, 3d-reconstruction, gaussian-splatting, nerf, mouse-reconstruction]
project: moremouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# MoReMouse 2단계 파이프라인 설계 및 구현 연구

## 기본 정보

| 항목 | 내용 |
|------|------|
| **날짜** | 2025-12-10 |
| **연구 주제** | 생쥐 6-view 영상 데이터를 활용한 단안(monocular) 3D 재구성 파이프라인 |
| **핵심 목표** | MoReMouse 논문의 2단계 파이프라인 설계 및 완전 구현 |
| **참조 논문** | [MoReMouse: Monocular Reconstruction of Laboratory Mouse](https://arxiv.org/abs/2507.04258) |

---

## 1. 배경 및 동기 (Background & Motivation)

### 1.1 문제 정의

단안 이미지(monocular image)에서 3D 재구성은 본질적으로 **ill-posed problem**이다:
- 깊이 정보의 모호성 (depth ambiguity)
- 가려진 영역 (occlusion) 처리 어려움
- 특히 생쥐는 텍스처가 균일하고(C57BL/6 mice), 비강체 변형(non-rigid deformation)이 심함

### 1.2 선행 연구와의 관계

| 기술 | 역할 |
|------|------|
| **MAMMAL** (Dunn et al., 2021) | 140 관절 생쥐 body model, 13,059 vertices |
| **3D Gaussian Splatting** (SIGGRAPH 2023) | 실시간 렌더링 가능한 3D 표현 |
| **DINOv2** | Self-supervised vision features (768-dim) |
| **DMTet** (NeurIPS 2021) | Differentiable mesh extraction |

### 1.3 핵심 아이디어

**문제**: 단안 이미지 → 3D 직접 학습은 GT 데이터 부족으로 어려움

**해결책**: 2단계 파이프라인
1. Multi-view 데이터로 photorealistic avatar 학습
2. Avatar로 대규모 합성 데이터 생성 → 단안→3D 네트워크 학습

---

## 2. 방법론 (Methodology)

### 2.1 전체 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MoReMouse 2-Stage Pipeline                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        STAGE 1: Data Engine                          │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  Multi-view Videos      Gaussian Avatar       Synthetic Dataset      │   │
│  │  (6 cameras)            (AGAM)                (12K scenes)           │   │
│  │  ┌─────────┐           ┌─────────┐           ┌─────────────┐         │   │
│  │  │ 🐭 🐭 🐭│  ───────▶ │ ○○○○○○  │  ───────▶ │ 768K images │         │   │
│  │  │   ×6    │  400K iter │ 13,059  │  random   │ + poses     │         │   │
│  │  │ views   │           │ Gaussians│  poses    │ + cameras   │         │   │
│  │  └─────────┘           └─────────┘           └─────────────┘         │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       STAGE 2: Model Training                        │   │
│  ├──────────────────────────────────────────────────────────────────────┤   │
│  │                                                                       │   │
│  │  Input Image        Triplane         NeRF            DMTet           │   │
│  │  [378×378]          Features         Rendering       Mesh            │   │
│  │  ┌─────────┐       ┌─────────┐      ┌─────────┐     ┌─────────┐      │   │
│  │  │         │ DINOv2│ XY XZ YZ│ MLP  │ Volume  │ SDF │  Final  │      │   │
│  │  │   🐭    │──────▶│ ┌┐┌┐┌┐  │─────▶│ Render  │────▶│  Mesh   │      │   │
│  │  │         │ 768-d │ └┘└┘└┘  │      │ 128 pts │     │  + RGB  │      │   │
│  │  └─────────┘       └─────────┘      └─────────┘     └─────────┘      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Stage 1: Gaussian Mouse Avatar (AGAM)

#### 목적
Multi-view 비디오에서 photorealistic 생쥐 아바타 학습

#### 입력 데이터
- **데이터셋**: markerless_mouse_1 (Dunn et al., 2021)
- **카메라**: 6-view 동기화 비디오
- **프레임**: 800 frames (8000 프레임에서 균등 샘플링)
- **이미지 크기**: 800×800

#### 아키텍처
```python
class GaussianAvatar:
    """UV 기반 Gaussian Splatting"""

    # 정점당 1개 Gaussian → 총 13,059개
    Parameters per Gaussian:
    - position_offset: [V, 3]  # 메쉬 정점 기준 오프셋
    - color: [V, 3]            # RGB
    - opacity: [V, 1]          # 불투명도
    - scale: [V, 3]            # Gaussian 크기
    - rotation: [V, 4]         # Quaternion
```

#### Linear Blend Skinning (LBS)
포즈에 따라 Gaussian 위치 변형:
```
v' = Σ w_j · T_j · v
```
- `w_j`: skinning weight (정점이 j번째 관절에 영향받는 정도)
- `T_j`: j번째 관절의 4×4 변환 행렬

#### 렌더링 (gsplat)
```python
rendered = rasterization(
    means=positions,      # [13059, 3]
    quats=rotations,      # [13059, 4]
    scales=scales,        # [13059, 3]
    opacities=opacities,  # [13059, 1]
    colors=colors,        # [13059, 3]
    viewmats=camera,      # [4, 4]
    Ks=intrinsics,        # [3, 3]
)
```

#### 학습 설정
| 파라미터 | 값 |
|----------|-----|
| Iterations | 400,000 |
| Learning Rate | 1e-3 |
| Loss | L1 + SSIM + LPIPS |
| Auto-resume | ✅ 지원 |

### 2.3 Stage 2: MoReMouse Network

#### 2.3.1 DINOv2 Encoder (Frozen)

```python
encoder = dinov2_vitb14  # ViT-B/14
# Input: [B, 3, 378, 378]
# Output: [B, 768, 27, 27] patch features
```

#### 2.3.2 Triplane Generator (Paper Table A3)

```python
class TriplaneGenerator:
    # 12-layer Transformer decoder
    # Flash Attention: O(n) memory

    Input:  [B, 729, 768]      # DINOv2 features
    Queries: [64×64, 512]      # Learnable
    Output: [B, 3, 80, 128, 128]  # XY, XZ, YZ planes
```

#### 2.3.3 NeRF Decoder

```python
class TriplaneDecoder:
    # 10 shared hidden layers (64 neurons)

    For each 3D point (x, y, z):
        f_xy = bilinear_sample(planes[0], x, y)
        f_xz = bilinear_sample(planes[1], x, z)
        f_yz = bilinear_sample(planes[2], y, z)
        f = f_xy + f_xz + f_yz

        density, color, embedding = MLP(f)
```

#### 2.3.4 Volume Rendering

```python
def volume_render(density, color, z_vals):
    # 128 samples per ray
    alpha = 1 - exp(-density * delta_t)
    T = cumprod(1 - alpha)
    weights = T * alpha

    rgb = sum(weights * color)
    depth = sum(weights * z_vals)
    return rgb, depth
```

### 2.4 2-Stage Training

| Stage | Epochs | 목적 |
|-------|--------|------|
| **NeRF** | 60 | Volumetric rendering으로 부드러운 gradient 전파 |
| **DMTet** | 100 | Explicit surface 추출로 geometric detail 향상 |

### 2.5 손실 함수

```python
L_total = λ_mse·L_mse + λ_lpips·L_lpips + λ_mask·L_mask
        + λ_smooth·L_smooth + λ_depth·L_depth + λ_geo·L_geo
```

| Loss | Weight (λ) | 역할 |
|------|------------|------|
| MSE | 1.0 | Pixel-wise RGB reconstruction |
| LPIPS | 1.0 | Perceptual similarity (VGG) |
| Mask | 0.3 | Binary cross-entropy (opacity) |
| Smooth L1 | 0.2 | Large discrepancy penalty |
| Depth | 0.2 | Scale-invariant depth |
| Geodesic | 0.1 | Embedding consistency |

---

## 3. 주요 결과 (Key Findings/Results)

### 3.1 구현 완료 상태

| Category | Status | Components |
|----------|--------|------------|
| **Models** | ✅ 100% | GaussianAvatar, MouseBodyModel, TriplaneGenerator, MoReMouse |
| **Data Loaders** | ✅ 100% | SyntheticDataset, MAMMALMultiviewDataset, VideoReader |
| **Loss Functions** | ✅ 100% | MSE, LPIPS, SSIM, Mask, Depth, Geodesic |
| **Scripts** | ✅ 100% | train, inference, evaluate, generate_synthetic_data, run_pipeline |
| **Configs** | ✅ 100% | model, data, train, avatar (Hydra-based) |

### 3.2 핵심 구현 모듈

```
src/
├── models/
│   ├── mouse_body.py        # MAMMAL 140-joint body model
│   ├── gaussian_avatar.py   # AGAM + Trainer (400K iter)
│   ├── geodesic_embedding.py # Heat method geodesic
│   ├── triplane.py          # Transformer + Decoder + Upsampler
│   └── moremouse_net.py     # DINOv2 + Triplane + NeRF
├── data/
│   ├── dataset.py           # SyntheticDataset, RealDataset
│   ├── mammal_loader.py     # Multi-view loader (video/image)
│   └── transforms.py        # Data augmentation
├── losses/
│   ├── reconstruction.py    # MSE, L1, SSIM, LPIPS
│   ├── mask.py, depth.py    # Mask/Depth losses
│   └── combined.py          # MoReMouseLoss
└── utils/
    ├── logging.py           # Console/file logging
    ├── metrics.py           # PSNR, SSIM, LPIPS
    └── visualization.py     # Multi-view grid, mesh
```

### 3.3 기술적 특징

| Feature | Implementation |
|---------|----------------|
| **Flash Attention** | `F.scaled_dot_product_attention` → O(n) memory |
| **Chunked Rendering** | 4096 rays/chunk → GPU memory efficient |
| **Video Reader** | LRU cache (100 frames) → 효율적 video access |
| **Auto-resume** | Checkpoint 자동 감지 및 resume 지원 |
| **Data Format** | Video (mp4) + Image 형식 자동 감지 |

### 3.4 예상 성능 (Paper 기준)

| Dataset | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---------|--------|--------|---------|
| Synthetic | 22.03 | 0.966 | 0.053 |
| Real | 18.42 | - | - |

---

## 4. 분석 및 논의 (Analysis & Discussion)

### 4.1 파이프라인 설계 근거

**왜 2단계 파이프라인인가?**

1. **데이터 문제 해결**: 실제 GT 3D mesh 데이터 획득 어려움 → Avatar로 합성 데이터 생성
2. **도메인 갭 최소화**: Multi-view로 학습된 avatar는 실제 생쥐 외관을 잘 복원
3. **확장성**: 다양한 포즈/뷰포인트 조합 생성 가능 (12K scenes × 64 views = 768K images)

### 4.2 구현 과정의 주요 결정

| 결정 사항 | 선택 | 이유 |
|-----------|------|------|
| Gaussian per vertex | 1개 | 13,059개로 충분한 표현력 |
| Triplane resolution | 128×128 | Memory vs quality 균형 |
| Query resolution | 64×64 → Upsample | Flash Attention 메모리 효율 |
| NeRF samples | 128 per ray | Paper specification |

### 4.3 시사점

1. **Avatar 품질이 최종 성능 결정**: Stage 1의 avatar 품질이 합성 데이터 품질 결정
2. **Flash Attention 필수**: 128×128 triplane은 기존 attention으로 OOM 발생
3. **Video format 지원 중요**: MAMMAL nerf format (mp4)와 image format 모두 지원 필요

---

## 5. 미결 과제 (Open Questions)

### 5.1 현재 한계

| 한계 | 설명 | 우선순위 |
|------|------|----------|
| **Avatar 학습 미완료** | 400K iteration 학습 필요 (약 24-48시간) | High |
| **합성 데이터 미생성** | 6000 frames × 64 views 생성 필요 | High |
| **DMTet 검증 미완료** | Kaolin 의존성으로 별도 환경 필요 | Medium |
| **Real data 평가** | 실제 데이터셋 evaluation 미진행 | Medium |

### 5.2 추가 탐색 필요

1. **Avatar 학습 하이퍼파라미터 튜닝**
   - Learning rate schedule
   - Loss weight 조정 (L1 vs SSIM vs LPIPS 비율)

2. **합성 데이터 다양성**
   - Pose sampling 전략 (uniform vs importance sampling)
   - Camera placement 전략

3. **일반화 성능**
   - 다른 생쥐 개체에 대한 성능
   - Out-of-distribution pose에 대한 robustness

---

## 6. 다음 단계 (Next Steps)

### 즉시 실행 가능
```bash
# 1. Avatar 학습 시작 (gpu05)
CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline.py \
    --stage avatar \
    --data-dir /home/joon/data/markerless_mouse_1_nerf \
    --avatar-iterations 400000

# 2. 합성 데이터 생성
python scripts/run_pipeline.py --stage synthetic \
    --avatar-checkpoint checkpoints/avatar/avatar_final.pt \
    --num-frames 6000 --num-views 64

# 3. MoReMouse 학습
python scripts/run_pipeline.py --stage train \
    --nerf-epochs 60 --dmtet-epochs 100
```

### 추후 작업
- [ ] Avatar 학습 완료 및 품질 검증
- [ ] 합성 데이터 생성 및 품질 확인
- [ ] Full training 실행 (NeRF 60 + DMTet 100 epochs)
- [ ] Real data evaluation
- [ ] 논문 수치 재현 확인

---

## 참고 문헌

1. MoReMouse: Monocular Reconstruction of Laboratory Mouse (arXiv:2507.04258, 2025)
2. 3D Gaussian Splatting for Real-Time Radiance Field Rendering (SIGGRAPH 2023)
3. Deep Marching Tetrahedra (NeurIPS 2021)
4. MAMMAL: Multi-Animal Articulated Model (2021)
5. DINOv2: Learning Robust Visual Features (CVPR 2024)

---

## Git Commits (Recent)

```
add9ad0 feat(avatar): add resume training and auto-checkpoint detection
2708646 fix(data): handle None pose in MAMMAL dataloader
d065fe5 feat(data): add video format support for MAMMAL multi-view data
1b5d033 fix: gsplat render fail
759319e feat: modules baseline
```

---

*Generated: 2025-12-10*
*MoReMouse 2-Stage Pipeline Research Note*
