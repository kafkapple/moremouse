# gsplat CUDA 컴파일 문제 해결 및 GPU 1 학습 가이드

**날짜**: 2024-12-12
**문제**: gsplat JIT 컴파일 실패, GPU 선택

---

## 1. 문제 원인

### gsplat CUDA 컴파일 실패
- **증상**: `crt/host_defines.h: No such file or directory`
- **원인**: conda의 `cuda-nvcc` 버전(12.9)과 PyTorch CUDA 버전(11.8) 불일치
- **해결**: 시스템 CUDA 11.1 사용

### backgrounds shape 에러
- **증상**: `AssertionError: torch.Size([3])`
- **원인**: gsplat 1.0.0은 `backgrounds`를 `(C, channels)` 형태로 기대
- **해결**: `torch.ones(3)` → `torch.ones(1, 3)` 수정

---

## 2. 환경 설정

### gpu05 CUDA 환경
```bash
# 시스템 CUDA 버전
/usr/local/cuda-11.1  # 설치됨
/usr/local/cuda-12.1  # 설치됨

# PyTorch CUDA 버전
PyTorch: 2.0.1
CUDA: 11.8

# gsplat 버전
gsplat: 1.0.0
```

---

## 3. 프로세스 종료

```bash
# gpu05에 SSH 접속
ssh gpu05

# 실행 중인 학습 프로세스 확인
ps aux | grep run_pipeline

# 프로세스 종료
pkill -f run_pipeline
```

---

## 4. GPU 1로 학습 실행

### 기본 명령어 (디버그 모드)

```bash
# gpu05에서 실행
source ~/anaconda3/etc/profile.d/conda.sh
conda activate moremouse

# CUDA 11.1 사용 설정 (gsplat 컴파일을 위해 필수!)
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=/usr/local/cuda-11.1/bin:$PATH

# GPU 1만 사용하도록 설정 (물리적 GPU 1 → 논리적 cuda:0)
export CUDA_VISIBLE_DEVICES=1

# GPU 1 번으로 학습 (1000 iter 테스트)
cd /home/joon/moremouse
python scripts/run_pipeline.py \
    --stage avatar \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --avatar-iterations 1000 \
    --vis-freq 100 \
    --save-freq 500 \
    --device cuda:0
```

**주의**: `CUDA_VISIBLE_DEVICES=1`을 설정하면 물리적 GPU 1이 논리적 `cuda:0`이 됩니다.
따라서 `--device cuda:0`으로 지정해야 합니다.

### 백그라운드 실행 (장시간 학습용)

```bash
export CUDA_VISIBLE_DEVICES=1
nohup python scripts/run_pipeline.py \
    --stage avatar \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --avatar-iterations 100000 \
    --vis-freq 1000 \
    --save-freq 10000 \
    --device cuda:0 \
    > training_log.txt 2>&1 &

# 로그 확인
tail -f training_log.txt
```

---

## 5. 주요 파라미터 설명

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--stage avatar` | Avatar 학습 단계 실행 | - |
| `--mouse-model` | MAMMAL mouse model 경로 | - |
| `--avatar-iterations` | 총 학습 iteration 수 | 100000 |
| `--vis-freq` | 시각화 저장 주기 | 1000 |
| `--save-freq` | 체크포인트 저장 주기 | 10000 |
| `--device` | GPU 선택 (cuda:0, cuda:1) | cuda:0 |

---

## 6. CUDA_HOME 설정이 필요한 이유

gsplat은 runtime에 CUDA 커널을 JIT(Just-In-Time) 컴파일합니다:

1. **PyTorch 빌드**: CUDA 11.8로 빌드됨
2. **시스템 CUDA**: 11.1, 12.1만 설치됨
3. **conda cuda-nvcc**: 12.9 (호환 불가)

**해결 방법**: 시스템의 CUDA 11.1을 사용
- CUDA 11.1은 11.8과 minor version만 다르므로 호환됨
- `CUDA_HOME` 환경변수로 nvcc 경로 지정

---

## 7. 관련 파일 수정 내역

### src/models/gaussian_avatar.py
- Line 375: `backgrounds` shape 수정
```python
# 기존
backgrounds=torch.ones(3, device=device),

# 수정
backgrounds=torch.ones(1, 3, device=device),  # (C, channels) for gsplat 1.0
```

---

## 8. 데이터 로더 수정 (MAMMAL 형식 지원)

### 해결한 경고
- `Warning: Calibration file not found` → `new_cam.pkl` 로드 지원 추가
- `Warning: Pose directory not found` → `center_rotation.npz` 글로벌 변환 로드 지원 추가

### src/data/mammal_loader.py 변경사항

1. **`new_cam.pkl` 지원** (카메라 캘리브레이션)
   - 형식: `List[Dict]` with `K`, `R`, `T` per camera
   - 변환: R, T → 4x4 viewmat

2. **`center_rotation.npz` 지원** (글로벌 변환)
   - `centers`: (N, 3) - 마우스 중심 위치
   - `angles`: (N,) - 회전 각도
   - 프레임 매핑: 18000 video frames → 3600 entries (ratio 5.0)

### 확인된 데이터 로드 결과
```
Loaded MAMMAL calibration: 6 cameras
Loaded global transform: 3600 entries (ratio: 5.0)
MAMMAL Dataset (video mode): 800 frames, 6 cameras
```

---

## 9. 문제 해결 체크리스트

- [ ] conda의 `cuda-nvcc` 제거됨 확인: `conda list | grep cuda`
- [ ] CUDA_HOME 설정: `echo $CUDA_HOME`
- [ ] nvcc 버전 확인: `which nvcc && nvcc --version`
- [ ] GPU 사용 현황 확인: `nvidia-smi`
- [ ] CUDA_VISIBLE_DEVICES 설정: `export CUDA_VISIBLE_DEVICES=1`
- [ ] 올바른 GPU 선택: `--device cuda:0` (CUDA_VISIBLE_DEVICES 설정 시)
- [ ] 캘리브레이션 로드 확인: `Loaded MAMMAL calibration: N cameras`
- [ ] 글로벌 변환 로드 확인: `Loaded global transform: N entries`
