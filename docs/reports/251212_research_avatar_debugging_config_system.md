---
date: 2025-12-12
context_name: "2_Research"
tags: [ai-assisted, moremouse, gaussian-avatar, config, debugging]
project: moremouse
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 251212 Research: Gaussian Avatar 디버깅 및 환경 설정 시스템 구축

## 📋 핵심 내용

### 1. NoneType Collate 에러 수정
DataLoader에서 batch collate 시 NoneType 에러 발생 문제 해결

**문제점**:
- `mammal_global`, `global_transform`이 None일 때 collate 실패
- `has_pose`, `frame_idx`가 Python primitive type이라 collate 불일치

**해결**:
```python
# Before: None 반환 가능
mammal_global = None

# After: placeholder tensor dict 반환
mammal_global_out = {
    'R': torch.zeros(3, dtype=torch.float32),
    'T': torch.zeros(3, dtype=torch.float32),
    's': torch.tensor(1.0, dtype=torch.float32),  # default scale
    'valid': torch.tensor(False),  # validity flag
}
```

**관련 파일**: `src/data/mammal_loader.py:714-747`

### 2. 환경별 Configuration 시스템 구축
gpu05 서버와 로컬(bori) 환경 간 경로 차이를 YAML config로 관리

**구조**:
```
configs/
├── default.yaml      # 기본 training 파라미터
├── gpu05.yaml        # gpu05 서버 경로
└── local.yaml        # 로컬 개발 환경 경로
```

**자동 환경 감지**:
- hostname 기반 자동 감지 (`detect_environment()`)
- gpu05, dlbox → gpu05 환경
- joon-dell, dell, bori → local 환경

**사용법**:
```python
from src.utils.config import load_config, get_paths

config = load_config()  # 자동 환경 감지
paths = get_paths()     # 경로만 가져오기
```

**관련 파일**: `src/utils/config.py`, `scripts/run_pipeline.py:286-314`

### 3. 이전 세션 작업 (참고)
- Axis-angle 회전 변환 수정 (MAMMAL → Rodrigues formula)
- 키포인트 시각화 추가
- world_scale 자동 계산

## 💡 교훈 및 인사이트

### DataLoader Collate 관련
1. **모든 반환값은 collate 가능해야 함**: None, bool, int → Tensor로 변환
2. **Validity flag 패턴**: None 대신 `{'value': tensor, 'valid': bool_tensor}` 구조 사용
3. **기본값 설정**: scale=0.0보다 scale=1.0이 더 안전 (곱셈 연산에서)

### 환경 설정 관리
1. **YAML config 장점**: 코드 변경 없이 환경별 설정 가능
2. **Hostname 기반 감지**: 명시적 환경 변수보다 자동 감지가 편리
3. **경로 확장**: `os.path.expanduser()`로 `~` 경로 처리

### 디버깅 전략
1. **단계별 확인**: DataLoader → Model → Renderer 순서로 문제 격리
2. **시각화 중요**: 숫자보다 이미지로 확인하면 문제 빠르게 파악
3. **Placeholder 데이터**: 문제 위치 파악 위해 의도적으로 단순한 데이터 사용

## 🎯 Action Items

- [ ] gpu05에서 수정된 코드로 avatar 학습 재실행 및 결과 확인
- [ ] 로컬 환경에서 sshfs로 gpu05 데이터 마운트 설정
- [ ] 키포인트 시각화 결과로 pose alignment 상태 검증
- [ ] gsplat CUDA 컴파일 에러 해결 (CUDA 버전 호환성 확인)

## 🔗 관련 파일

### 수정된 파일
- `src/data/mammal_loader.py` - NoneType collate 수정
- `src/utils/config.py` - 환경별 config 로더 (신규)
- `scripts/run_pipeline.py` - config 시스템 통합
- `configs/default.yaml` - 기본 설정 (신규)
- `configs/gpu05.yaml` - gpu05 환경 설정 (신규)
- `configs/local.yaml` - 로컬 환경 설정 (신규)

### 이전 세션 관련 파일
- `src/models/mouse_body.py` - axis_angle_to_rotation_matrix
- `src/models/gaussian_avatar.py` - keypoint visualization, world_scale

### Git Commits
```
c67f1d8 fix(config): add bori hostname to local environment detection
792d5f3 feat(config): add environment-aware configuration system
ac31669 feat(vis): add keypoint and debug info visualization
```

## 📝 로컬 실행 가이드

### 데이터 마운트 (sshfs)
```bash
mkdir -p ~/mnt/gpu05_data
sshfs gpu05:/home/joon/data ~/mnt/gpu05_data
sshfs gpu05:/home/joon/MAMMAL_mouse ~/mnt/gpu05_data/MAMMAL_mouse
```

### 실행
```bash
# 환경 자동 감지되어 local config 사용
python scripts/run_pipeline.py --stage avatar --avatar-iterations 100 --vis-freq 10
```
