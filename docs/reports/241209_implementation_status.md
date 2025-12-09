# MoReMouse Implementation Status Report

**Date**: 2024-12-09
**Environment**: gpu05 (SSH)
**Reference**: arXiv:2507.04258v2

---

## 1. Implementation Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Training (train.py)** | ✅ 구현됨 | NeRF/DMTet 2단계, AMP 지원 |
| **Inference (inference.py)** | ✅ 구현됨 | Single image/video, Novel view synthesis |
| **Synthetic Data Gen** | ✅ 구현됨 | Gaussian Avatar 기반 |
| **Evaluation (evaluate.py)** | ✅ 구현됨 | PSNR, SSIM, LPIPS, IoU |
| **MoReMouse Network** | ✅ 구현됨 | DINOv2 + Triplane + NeRF |
| **Triplane Generator** | ✅ 구현됨 | Flash Attention, Upsampler 포함 |
| **Gaussian Avatar (AGAM)** | ⚠️ 부분 구현 | gsplat 의존성 필요 |
| **Mouse Body Model** | ✅ 구현됨 | LBS, 140 joints |
| **Loss Functions** | ✅ 구현됨 | MSE, LPIPS, Mask, Depth, Geodesic |
| **Dataset Classes** | ✅ 구현됨 | Synthetic + Real |
| **DMTet Renderer** | ❌ 미구현 | NeRF만 완전 구현 |
| **Visualization Tools** | ⚠️ 부분 구현 | inference.py에 기본 기능 |

---

## 2. Dataset Paths

### 필요 데이터

```
# 기본 경로 (config.yaml 기준)
${project_root}/../MAMMAL_mouse/
├── data/
│   └── markerless_mouse_1/   # 원본 비디오 데이터
└── mouse_model/
    ├── mouse.pkl             # 메인 모델 파일 (필수)
    ├── mouse_txt/
    │   ├── bone_length_mapper.txt
    │   └── reduced_face_7200.txt
    └── keypoint22_mapper.json
```

### 환경 변수 설정 (옵션)

```bash
export MAMMAL_DATA_PATH=/path/to/MAMMAL_mouse/data
export MOUSE_MODEL_PATH=/path/to/MAMMAL_mouse/mouse_model
```

### gpu05에서 예상 경로

```bash
# 기본적으로 다음 경로 사용 시도:
~/moremouse/../MAMMAL_mouse/data
~/moremouse/../MAMMAL_mouse/mouse_model

# 또는 심볼릭 링크 생성:
ln -s /path/to/MAMMAL_mouse ~/MAMMAL_mouse
```

---

## 3. Execution Commands

### Step 1: 합성 데이터 생성

```bash
ssh gpu05
cd ~/moremouse
conda activate splatter

# 합성 데이터 생성 (Gaussian Avatar 사용)
python scripts/generate_synthetic_data.py \
    --output data/synthetic \
    --mouse-model ../MAMMAL_mouse/mouse_model \
    --num-frames 100 \
    --num-views 8 \
    --image-size 378 \
    --device cuda:0
```

### Step 2: 학습 (2단계)

```bash
# Stage 1: NeRF (60 epochs) + Stage 2: DMTet (100 epochs)
python scripts/train.py \
    experiment.name=moremouse_exp1 \
    experiment.device=cuda:0

# 또는 특정 설정 오버라이드
python scripts/train.py \
    data.dataloader.batch_size=2 \
    train.stages.nerf.epochs=30 \
    logging.use_wandb=true
```

### Step 3: 평가

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --dataset synthetic \
    --data-dir data \
    --device cuda:0
```

### Step 4: 추론 (Novel View Synthesis)

```bash
# 단일 이미지
python scripts/inference.py \
    --image path/to/input.png \
    --checkpoint checkpoints/best.pt \
    --output results/inference \
    --num-views 8 \
    --device cuda:0

# 비디오
python scripts/inference.py \
    --video path/to/input.mp4 \
    --checkpoint checkpoints/best.pt \
    --output results/video_inference \
    --num-views 4 \
    --device cuda:0
```

---

## 4. Visualization Outputs

### inference.py 출력물

```
results/inference/
├── input.png           # 입력 이미지
├── view_00.png         # Novel view 0
├── view_01.png         # Novel view 1
├── ...
├── view_07.png         # Novel view 7
└── rotation.mp4        # 회전 비디오
```

### 학습 중 체크포인트

```
checkpoints/
├── latest.pt           # 최신 체크포인트
├── best.pt             # 최고 성능 체크포인트
└── checkpoint_XXXXXXXX.pt  # 주기적 저장
```

---

## 5. 미구현/보완 필요 사항

### 5.1 DMTet Renderer (미구현)

현재 NeRF 렌더링만 구현됨. DMTet 2단계 학습은 코드 구조는 있으나 실제 렌더러 미구현.

**필요 작업**:
- `src/renderers/dmtet.py` 추가
- Marching Cubes mesh extraction
- Differentiable surface rendering

### 5.2 시각화 스크립트 (부분 구현)

**현재 있는 것**:
- `inference.py`: Novel view rendering, 회전 비디오 생성

**추가 필요**:
- 학습 진행 시각화 (loss curves, learning rate)
- Triplane 시각화
- Geodesic embedding 시각화
- Mesh export (OBJ/PLY)

### 5.3 Dependencies 확인 필요

```bash
# 필수 패키지
pip install gsplat        # Gaussian splatting
pip install lpips         # Perceptual loss
pip install einops        # Tensor operations
pip install hydra-core    # Config management

# DINOv2 (자동 다운로드)
# torch.hub에서 자동 로드, 또는:
pip install transformers  # 대안
```

---

## 6. Quick Start (gpu05)

```bash
# 1. SSH 접속
ssh gpu05

# 2. 환경 활성화
cd ~/moremouse
conda activate splatter

# 3. 데이터 경로 확인
ls ../MAMMAL_mouse/mouse_model/mouse.pkl

# 4. 테스트 학습 (소규모)
python scripts/train.py \
    experiment.name=test_run \
    train.stages.nerf.epochs=5 \
    train.stages.dmtet.epochs=0 \
    data.dataloader.batch_size=2

# 5. 추론 테스트
python scripts/inference.py \
    --image test_input.png \
    --checkpoint checkpoints/latest.pt \
    --device cuda:0
```

---

## 7. 논문 파이프라인 vs 현재 구현

| 논문 단계 | 구현 상태 | 비고 |
|----------|----------|------|
| 1. AGAM 학습 (800 frames, 400K iter) | ⚠️ | gsplat 필요 |
| 2. 합성 데이터 생성 (12K scenes) | ✅ | generate_synthetic_data.py |
| 3. NeRF 학습 (60 epochs) | ✅ | train.py Stage 1 |
| 4. DMTet 학습 (100 epochs) | ❌ | 렌더러 미구현 |
| 5. Novel View Synthesis | ✅ | inference.py |
| 6. Mesh Export | ❌ | 미구현 |
