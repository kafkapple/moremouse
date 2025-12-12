# MAMMAL Pose Estimation Pipeline 가이드

**날짜**: 2024-12-12
**목적**: 새로운 비디오 프레임에 대한 MAMMAL pose 추정 파이프라인 사용법

---

## 1. 개요

MAMMAL pose estimation은 단일 RGB 이미지와 마스크로부터 140-joint mouse body model의 포즈를 추정합니다.

### 주요 구성요소
- **fit_monocular.py**: 메인 피팅 스크립트
- **ArticulationTorch**: MAMMAL 관절 모델
- **SuperAnimal Detector**: 키포인트 추정 (선택적)

---

## 2. 환경 설정

### gpu05 서버 환경
```bash
# SSH 접속
ssh gpu05

# 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate moremouse

# MAMMAL 디렉토리로 이동
cd /home/joon/MAMMAL_mouse
```

### 환경 변수 (자동 설정됨)
```bash
# env_config.sh에서 자동 설정
export CUDA_VISIBLE_DEVICES=1   # gpu05에서 GPU 1 사용
export EGL_DEVICE_ID=1
export PYOPENGL_PLATFORM=egl
```

---

## 3. 입력 데이터 준비

### 필요한 파일 형식
```
input_dir/
├── frame_0000_rgb.png     # RGB 이미지
├── frame_0000_mask.png    # 바이너리 마스크 (마우스=255, 배경=0)
├── frame_0001_rgb.png
├── frame_0001_mask.png
└── ...
```

### 지원하는 파일명 패턴
- `*_rgb.png` + `*_mask.png` (표준)
- `*_cropped.png` + `*_mask.png` (4x 업샘플링 데이터)
- `*.png` + `*_mask.png` (일반)

### 마스크 생성 (필요시)
```bash
# SAM (Segment Anything Model) 사용 예시
python preprocessing_utils/generate_masks.py \
    --input_dir data/frames/ \
    --output_dir data/frames_masked/
```

---

## 4. Pose 추정 실행

### 기본 명령어
```bash
cd /home/joon/MAMMAL_mouse

# 쉘 스크립트 사용 (권장)
./run_mesh_fitting_monocular.sh <input_dir> <output_dir> [max_images]

# 예시: 5 프레임 테스트
./run_mesh_fitting_monocular.sh data/test_frames/ results/test_output/ 5
```

### Python 직접 실행
```bash
python fit_monocular.py \
    --input_dir data/test_frames/ \
    --output_dir results/test_output/ \
    --max_images 5 \
    --device cuda:0 \
    --keypoints all
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|-----|------|-------|
| `--input_dir` | 입력 RGB+Mask 디렉토리 | (필수) |
| `--output_dir` | 출력 저장 디렉토리 | (필수) |
| `--max_images` | 처리할 최대 이미지 수 | 전체 |
| `--device` | GPU 선택 | cuda |
| `--keypoints` | 키포인트 선택 (`all`, `head`, `spine`, `none`) | all |
| `--detector` | 키포인트 검출기 (`geometric`, `superanimal`) | geometric |

---

## 5. 출력 파일 구조

### 결과 디렉토리
```
output_dir/
├── frame_0000_params.pkl    # MAMMAL 파라미터 (중요!)
├── frame_0000_mesh.obj      # 3D 메시 (Blender 호환)
├── frame_0000_overlay.png   # 시각화: RGB + 메시 + 마스크
├── frame_0000_rendered.png  # 렌더링된 메시
├── frame_0000_comparison.png # 원본 vs 피팅 비교
└── frame_0000_mask.png      # 처리된 마스크
```

### params.pkl 구조
```python
{
    'thetas': (1, 140, 3),     # 관절 회전 (axis-angle)
    'bone_lengths': (140,),    # 뼈 길이
    'R': (1, 3),               # 글로벌 회전 (axis-angle)
    'T': (1, 3),               # 글로벌 이동
    's': (1, 1),               # 스케일
    'keypoints_2d': (22, 3),   # 검출된 키포인트
    'keypoints_filtered': (22, 3),  # 필터링된 키포인트
    'selected_indices': [...]  # 사용된 키포인트 인덱스
}
```

---

## 6. MoReMouse 학습에 연결

### Pose 데이터 경로 설정
MAMMAL 결과 디렉토리를 MoReMouse 데이터 로더가 자동 검색:

```python
# mammal_loader.py가 자동으로 검색하는 경로들
pose_search_dirs = [
    data_dir / "poses",
    data_dir.parent / "results" / "monocular",
    Path("/home/joon/MAMMAL_mouse/results/monocular"),
]
```

### 수동 경로 지정
```python
dataset = MAMMALMultiviewDataset(
    data_dir="/home/joon/MAMMAL_mouse/nerf_data/mouse_1",
    pose_dir="/home/joon/MAMMAL_mouse/results/monocular/mouse_batch_xxx"
)
```

### 학습 실행
```bash
# gpu05에서
cd /home/joon/moremouse
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME=/usr/local/cuda-11.1

python scripts/run_pipeline.py \
    --stage avatar \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --avatar-iterations 100000 \
    --device cuda:0
```

---

## 7. 추가 프레임 Pose 추정 워크플로우

### Step 1: 비디오에서 프레임 추출
```bash
# 비디오에서 특정 프레임 추출
ffmpeg -i video.mp4 -vf "select='between(n\,100\,199)'" -vsync 0 frames/frame_%04d.png
```

### Step 2: 마스크 생성
```bash
# SAM 또는 기존 세그멘테이션 도구 사용
# (별도 스크립트 필요)
```

### Step 3: Pose 추정
```bash
cd /home/joon/MAMMAL_mouse
./run_mesh_fitting_monocular.sh frames/ results/new_poses/
```

### Step 4: 결과 확인
```bash
# overlay 이미지로 시각적 확인
ls results/new_poses/*_overlay.png
```

---

## 8. 문제 해결

### CUDA 메모리 오류
```bash
# 배치 크기 줄이기 또는 GPU 메모리 확인
nvidia-smi
# 다른 프로세스 종료 후 재시도
```

### 마스크 불량 경고
```
Warning: Mask not found for frame_0010.png, skipping
```
→ 마스크 파일명이 `frame_0010_mask.png`인지 확인

### 키포인트 검출 실패
```bash
# keypoints none 옵션으로 실행 (기하학적 피팅만 사용)
./run_mesh_fitting_monocular.sh input/ output/ - -- --keypoints none
```

### EGL 렌더링 오류
```bash
# EGL 장치 설정 확인
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=1
```

---

## 9. 빠른 참조 명령어

### 테스트 실행 (5프레임)
```bash
ssh gpu05
cd /home/joon/MAMMAL_mouse
./run_mesh_fitting_monocular.sh data/test/ results/test/ 5
```

### 전체 프레임 실행 (백그라운드)
```bash
nohup ./run_mesh_fitting_monocular.sh data/full/ results/full/ > fitting.log 2>&1 &
tail -f fitting.log
```

### 결과 복사 (로컬로)
```bash
# 로컬 머신에서
rsync -avz gpu05:/home/joon/MAMMAL_mouse/results/monocular/xxx/ ./results/
```

---

## 10. 관련 문서

- [gsplat CUDA 설정](../troubleshooting/251212_gsplat_cuda_gpu1_training.md)
- [Gaussian Avatar 구현 비교](../reports/251212_gaussian_avatar_paper_comparison.md)
- MAMMAL 원 저장소: `/home/joon/MAMMAL_mouse/`
