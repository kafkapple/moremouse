# MoReMouse Test Pipeline Execution Report

**Date**: 2025-12-14
**Version**: Test v1
**Status**: 🔄 In Progress

---

## 1. 실행 개요

MoReMouse 전체 파이프라인 테스트 실행 (10시간 이내 완료 목표)

### 실행 환경
| Item | Value |
|------|-------|
| **Server** | gpu05 (RTX 3090) |
| **CUDA** | 11.8 |
| **Python** | 3.10 |
| **GPU** | CUDA_VISIBLE_DEVICES=1 |

---

## 2. 파이프라인 단계별 진행 상황

### Stage 1: Gaussian Avatar Training ✅ 완료

**설정**:
```bash
python scripts/run_pipeline.py --stage avatar \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --avatar-iterations 2000 \
    --vis-freq 500 \
    --save-freq 500 \
    --device cuda:0
```

**결과**:
- 학습 프레임: 55 frames (18000 중 유효 pose)
- 카메라: 6 views
- 체크포인트: `checkpoints/avatar/avatar_final.pt` (12MB)
- 시각화: `outputs/avatar_vis/iter_*.png`

**체크포인트 목록**:
```
checkpoints/avatar/
├── avatar_iter_001000.pt
├── avatar_iter_001500.pt
├── avatar_iter_002000.pt
└── avatar_final.pt
```

**멀티뷰 렌더링 테스트**: 8개 뷰 렌더링 성공
- 출력: `outputs/multiview_test/grid.png`

---

### Stage 2: MoReMouse Network Training 📋 대기중

**설정** (테스트용 축소, OOM 방지 batch_size=2):
```bash
python scripts/train.py \
    experiment.name=moremouse_test_v1 \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=10 \
    train.stages.dmtet.epochs=20 \
    data.dataloader.batch_size=2 \
    logging.use_wandb=false
```

**OOM 발생 시 추가 축소**:
```bash
data.dataloader.batch_size=1
```

**설정 비교** (Paper vs Test):

| Parameter | Paper | Test |
|-----------|-------|------|
| NeRF epochs | 60 | 10 |
| DMTet epochs | 100 | 20 |
| Batch size | 8 | 4 |
| WandB | true | false |

**예상 시간**: ~3-4시간

**출력 예정**:
- 체크포인트: `checkpoints/moremouse_test_v1/`
- 로그: `outputs/moremouse_test_v1/`

---

### Stage 3: Evaluation 📋 대기중

**명령어**:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/moremouse_test_v1/best.pt \
    --device cuda:0 \
    --output outputs/eval_moremouse_test
```

**평가 메트릭**:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity)
- LPIPS (Learned Perceptual Image Patch Similarity)

---

### Stage 4: 3D Visualization 📋 대기중

**명령어**:
```bash
python scripts/visualize_3d.py \
    --checkpoint checkpoints/moremouse_test_v1/best.pt \
    --output outputs/vis_moremouse_test \
    --num-views 8
```

**출력 예정**:
- Novel view renders
- 360° rotation video
- Depth/normal maps

---

### Stage 5: Final Report 📋 대기중

**명령어**:
```bash
python scripts/generate_final_report.py \
    --checkpoint checkpoints/moremouse_test_v1/best.pt \
    --output outputs/reports/moremouse_test_v1
```

---

## 3. 파일 위치 요약

### 입력 데이터
```
/home/joon/data/markerless_mouse_1_nerf/
├── videos_undist/           # 6개 카메라 비디오
├── videos_undist_mask/      # 마스크 비디오
├── camera_params/           # 카메라 캘리브레이션
└── global_transform.pkl     # Global transform
```

### 포즈 데이터
```
/home/joon/MAMMAL_mouse/results/monocular/mouse_batch_20251125_132606_mouse_1/
└── *.pkl                    # 프레임별 포즈 파라미터
```

### 출력
```
/home/joon/moremouse/
├── checkpoints/
│   ├── avatar/              # Stage 1 체크포인트
│   └── moremouse_test_v1/   # Stage 2 체크포인트
├── outputs/
│   ├── avatar_vis/          # Avatar 시각화
│   ├── multiview_test/      # 멀티뷰 렌더링
│   ├── eval_moremouse_test/ # 평가 결과
│   └── vis_moremouse_test/  # 3D 시각화
└── docs/reports/            # 문서
```

---

## 4. 실행 로그

### 2025-12-14 22:30 KST
- Stage 1 완료: Avatar 2000 iter 학습 완료
- 멀티뷰 렌더링 테스트 완료

### 2025-12-14 23:48 KST
- Stage 2 시도: CUDA OOM 에러 발생
- 원인: 다른 프로세스들이 GPU 메모리 ~34GB 사용 중
- 해결: batch_size=4 → batch_size=2로 축소

### 2025-12-15 00:15 KST
- 사용자가 직접 실행하기로 결정
- 단계별 명령어 정리 완료

---

## 5. 전체 파이프라인 자동화 스크립트

향후 재현을 위한 단일 스크립트:

```bash
#!/bin/bash
# MoReMouse Full Pipeline Test Script
# Usage: bash run_full_test.sh

set -e

# Environment setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate moremouse
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_VISIBLE_DEVICES=1
cd /home/joon/moremouse

echo "============================================"
echo "MoReMouse Test Pipeline"
echo "============================================"

# Stage 1: Avatar (skip if already done)
if [ ! -f "checkpoints/avatar/avatar_final.pt" ]; then
    echo "[Stage 1] Training Avatar..."
    python scripts/run_pipeline.py --stage avatar \
        --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
        --avatar-iterations 2000 \
        --vis-freq 500 \
        --save-freq 500 \
        --device cuda:0
fi

# Stage 2: MoReMouse Training (batch_size=2 for OOM prevention)
echo "[Stage 2] Training MoReMouse..."
python scripts/train.py \
    experiment.name=moremouse_test_v1 \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=10 \
    train.stages.dmtet.epochs=20 \
    data.dataloader.batch_size=2 \
    logging.use_wandb=false

# Stage 3: Evaluation
echo "[Stage 3] Running Evaluation..."
python scripts/evaluate.py \
    --checkpoint checkpoints/moremouse_test_v1/best.pt \
    --device cuda:0 \
    --output outputs/eval_moremouse_test

# Stage 4: Visualization
echo "[Stage 4] Generating Visualizations..."
python scripts/visualize_3d.py \
    --checkpoint checkpoints/moremouse_test_v1/best.pt \
    --output outputs/vis_moremouse_test \
    --num-views 8

# Stage 5: Final Report
echo "[Stage 5] Generating Final Report..."
python scripts/generate_final_report.py \
    --checkpoint checkpoints/moremouse_test_v1/best.pt \
    --output outputs/reports/moremouse_test_v1

echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
```

---

## 6. 다음 단계 (Full Training)

테스트 완료 후, 논문 설정으로 전체 학습:

```bash
# Paper settings
python scripts/train.py \
    experiment.name=moremouse_full \
    experiment.device=cuda:0 \
    train.stages.nerf.epochs=60 \
    train.stages.dmtet.epochs=100 \
    data.dataloader.batch_size=8 \
    logging.use_wandb=true
```

---

*Last Updated: 2025-12-14 23:48 KST*
