# MoReMouse 환경 동기화 가이드

## 개요

MoReMouse는 환경별 config 시스템을 통해 여러 서버/로컬 환경에서 동일한 코드로 실행 가능합니다.

## 동기화 구조

```
┌─────────────────┐         ┌─────────────────┐
│   Local (bori)  │         │     gpu05       │
├─────────────────┤         ├─────────────────┤
│ configs/        │◄──git──►│ configs/        │
│   local.yaml    │         │   gpu05.yaml    │
│ src/            │◄──git──►│ src/            │
│ scripts/        │◄──git──►│ scripts/        │
├─────────────────┤         ├─────────────────┤
│ ~/mnt/gpu05_data│◄─sshfs─►│ /home/joon/data │
│   (마운트)       │         │   (원본 데이터)  │
└─────────────────┘         └─────────────────┘
```

## 1. 코드 동기화 (Git)

```bash
# 로컬에서 작업 후
git add -A && git commit -m "message" && git push origin main

# gpu05에서 pull
ssh gpu05 "cd /home/joon/moremouse && git pull origin main"
```

## 2. Config 시스템

### 자동 환경 감지
`src/utils/config.py`가 hostname으로 환경을 자동 감지합니다:

| Hostname 패턴 | 환경 | Config 파일 |
|--------------|------|-------------|
| gpu05, dlbox | gpu05 | configs/gpu05.yaml |
| joon-dell, bori | local | configs/local.yaml |
| 기타 | default | configs/default.yaml |

### Config 파일 구조
```yaml
# configs/gpu05.yaml
paths:
  data_dir: /home/joon/data/markerless_mouse_1_nerf
  mouse_model: /home/joon/MAMMAL_mouse/mouse_model
  pose_dir: /home/joon/MAMMAL_mouse/results/monocular/mouse_batch_*

# configs/local.yaml
paths:
  data_dir: ~/mnt/gpu05_data/data/markerless_mouse_1_nerf
  mouse_model: ~/mnt/gpu05_data/MAMMAL_mouse/mouse_model
  pose_dir: ~/mnt/gpu05_data/MAMMAL_mouse/results/monocular/mouse_batch_*
```

### 사용법
```python
from src.utils.config import load_config, get_paths

config = load_config()  # 자동 환경 감지
paths = get_paths()     # 경로만 가져오기

# 또는 명시적 환경 지정
config = load_config(environment='gpu05')
```

## 3. 데이터 동기화 (sshfs)

로컬에서 gpu05 데이터에 접근하려면 sshfs 마운트 필요:

```bash
# sshfs 설치
sudo apt-get install -y sshfs

# 마운트 디렉토리 생성
mkdir -p ~/mnt/gpu05_data

# 마운트 (reconnect 옵션으로 연결 끊김 시 자동 재연결)
sshfs gpu05:/home/joon ~/mnt/gpu05_data -o reconnect,ServerAliveInterval=15

# 마운트 해제
fusermount -u ~/mnt/gpu05_data
```

## 4. 새 서버 추가 시

1. `configs/` 디렉토리에 새 config 파일 생성:
```yaml
# configs/new_server.yaml
environment: new_server
hostname_patterns:
  - new-server-hostname

paths:
  data_dir: /path/to/data
  mouse_model: /path/to/mouse_model
  pose_dir: /path/to/poses

device: cuda:0
```

2. `src/utils/config.py`의 `detect_environment()` 함수에 hostname 패턴 추가:
```python
elif 'new-server' in hostname:
    return 'new_server'
```

## 5. 실행

양쪽 환경에서 동일한 명령으로 실행:
```bash
python scripts/run_pipeline.py --stage avatar --avatar-iterations 100

# 환경이 자동 감지되어 적절한 경로 사용
# [Config] Detected environment: gpu05  (또는 local)
```

## 주의사항

- 데이터는 git에 포함되지 않음 (용량 문제)
- 체크포인트/출력 파일도 환경별로 분리됨
- 서버 이전 시 config 파일만 새로 작성하면 됨
