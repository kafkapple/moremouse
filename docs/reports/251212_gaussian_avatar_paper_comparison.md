# Gaussian Avatar 구현 개선 보고서

**날짜**: 2024-12-12
**목적**: MoReMouse 논문 수준의 Gaussian Avatar 구현

---

## 1. 개요

기존 구현의 품질 문제를 해결하기 위해 원 논문(MoReMouse) 방식을 분석하고 핵심 기능들을 구현했습니다.

### 주요 변경 사항

| 항목 | 기존 구현 | 개선된 구현 | 논문 |
|------|-----------|-------------|------|
| **역전파** | detach() 차단 | 완전한 gradient flow | O |
| **UV 매개변수화** | vertex index만 사용 | UV 좌표 기반 anchor | O |
| **로컬 좌표계** | 없음 | tangent frame 사용 | O |
| **손실함수** | L1+SSIM+LPIPS | +TV loss, +Surface loss | O |
| **초기화** | 모두 0 | 작은 랜덤 값 | O |

---

## 2. 상세 변경 내용

### 2.1 역전파 복구 (Step 1)

**파일**: `src/models/gaussian_avatar.py:276-297`

```python
# 기존 (detach로 차단)
means = gaussian_params["means"][b].detach()
quats = gaussian_params["rotations"][b].detach()

# 개선 (gradient 유지)
means = gaussian_params["means"][b]
quats = gaussian_params["rotations"][b]
```

**효과**: gsplat 렌더러를 통한 end-to-end 학습 가능

---

### 2.2 UV 좌표 로드 및 활용 (Step 2)

**파일**: `src/models/mouse_body.py:134-158`

```python
# UV 좌표 로드
uv_coords = np.loadtxt(uv_coords_path)  # [15399, 2]
faces_tex = np.loadtxt(faces_tex_path)   # [28800, 3]

# UV→Vertex 매핑 계산
uv_to_vert = _compute_uv_to_vert_mapping(faces_vert, faces_tex, len(uv_coords))
```

**데이터 소스**: `MAMMAL_mouse/mouse_model/mouse_txt/textures.txt`

**효과**:
- Gaussian 수: 13,059 (vertex) → 15,399 (UV)
- 메시 표면의 UV 매개변수화 활용

---

### 2.3 로컬 좌표계 (Tangent Frame) 구현 (Step 3)

**파일**: `src/models/mouse_body.py:209-338`

```python
def compute_vertex_normals(vertices):
    """Face normals → Vertex normals"""

def compute_tangent_frames(vertices, normals=None):
    """UV 기반 tangent frame 계산
    Returns: (normals, tangents, bitangents)
    """
```

**파일**: `src/models/gaussian_avatar.py:271-303`

```python
# Forward pass에서 로컬 좌표계 적용
normals, tangents, bitangents = body_model.compute_tangent_frames(V)

# Position offset을 로컬→월드 변환
offset_world = (
    offsets[..., 0:1] * T_gaussians +  # Tangent 방향
    offsets[..., 1:2] * B_gaussians +  # Bitangent 방향
    offsets[..., 2:3] * N_gaussians    # Normal 방향
)

# Rotation도 로컬→월드 변환
frame_quats = rotation_matrix_to_quaternion(frame_matrix)
rotations = quaternion_multiply(frame_quats, local_quats)
```

**효과**: 포즈 변화에 따른 일관된 Gaussian 방향 유지

---

### 2.4 손실함수 강화 (Step 4)

**파일**: `src/models/gaussian_avatar.py:471-524`

```python
# 기존
total_loss = l1_loss + λ_ssim * ssim_loss + λ_lpips * lpips_loss

# 개선
total_loss = (
    l1_loss +
    λ_ssim * ssim_loss +      # 0.2 (논문)
    λ_lpips * lpips_loss +    # 0.1 (논문)
    λ_tv * tv_loss +          # 0.01 (신규)
    λ_surface * surface_loss  # 0.1 (신규)
)
```

#### TV Loss (Total Variation)
- 인접 Gaussian 간 파라미터 부드러움 강제
- Position, Color, Scale에 적용

#### Surface Loss
- Position offset 크기 제한 (5mm threshold)
- Huber-like penalty로 표면에서 이탈 방지

---

### 2.5 초기화 전략 개선 (Step 5)

**파일**: `src/models/gaussian_avatar.py:152-194`

| 파라미터 | 기존 | 개선 |
|---------|------|------|
| position_offsets | zeros | randn * 0.001 |
| colors_raw | zeros | randn * 0.1 |
| log_scales | log(0.01) | log(0.005) + randn * 0.1 |
| quaternions | identity | identity + randn * 0.01 |
| opacity | 0.8 | 0.9 |

**효과**:
- 초기 다양성 확보
- 더 작은 Gaussian으로 세밀한 디테일

---

### 2.6 Optimizer 개선

**파일**: `src/models/gaussian_avatar.py:422-430`

```python
# Per-parameter learning rates
param_groups = [
    {'params': [position_offsets], 'lr': lr * 0.1},   # 위치: 느리게
    {'params': [colors_raw], 'lr': lr},               # 색상: 표준
    {'params': [opacity_raw], 'lr': lr * 0.5},        # 불투명도: 중간
    {'params': [log_scales], 'lr': lr * 0.5},         # 스케일: 중간
    {'params': [quaternions], 'lr': lr * 0.1},        # 회전: 느리게
]
```

---

## 3. 논문과의 비교

### 3.1 구현 완료 항목

| 논문 기능 | 구현 상태 | 비고 |
|----------|----------|------|
| UV 매개변수화 | ✅ 완료 | textures.txt 활용 |
| LBS 변형 | ✅ 완료 | 기존 유지 |
| Local tangent frame | ✅ 완료 | UV 기반 계산 |
| L1 + SSIM + LPIPS loss | ✅ 완료 | λ값 논문과 동일 |
| TV loss | ✅ 완료 | 추가 구현 |
| 역전파 지원 | ✅ 완료 | detach 제거 |

### 3.2 구현 차이점

| 논문 | 현재 구현 | 차이 이유 |
|------|-----------|-----------|
| StyleUNet 네트워크 | 직접 파라미터 | 단순화, 추후 확장 가능 |
| Triplane 64×64 | Per-Gaussian 파라미터 | 메모리 효율성 |
| 측지선 임베딩 | 미구현 | 추가 구현 필요 시 |

### 3.3 추후 개선 가능 항목

1. **StyleUNet 네트워크**: UV 맵을 입력으로 Gaussian 속성 예측
2. **측지선 거리 임베딩**: 표면 일관성 향상
3. **Multi-resolution**: 다양한 해상도의 Gaussian

---

## 4. 사용법

### 학습 재시작 (기존 체크포인트 무시)

```bash
# 기존 체크포인트 삭제 후 재학습
rm -rf checkpoints/avatar/*
python scripts/run_pipeline.py --stage avatar
```

### 새로운 손실함수 가중치 조정

```python
trainer = GaussianAvatarTrainer(
    avatar,
    lambda_ssim=0.2,      # 구조 유사성
    lambda_lpips=0.1,     # 지각적 유사성
    lambda_tv=0.01,       # 공간 부드러움
    lambda_surface=0.1,   # 표면 제약
)
```

---

## 5. 파일 변경 요약

| 파일 | 변경 내용 |
|------|----------|
| `src/models/mouse_body.py` | UV 좌표 로드, tangent frame 계산 |
| `src/models/gaussian_avatar.py` | UV 기반 매개변수화, 로컬 좌표계, 손실함수 |

---

## 6. 참고 자료

- **논문**: MoReMouse - Gaussian Avatar for Laboratory Mouse
- **UV 데이터**: MAMMAL_mouse/mouse_model/mouse_txt/
- **원본 코드 참고**: MAMMAL_mouse/uvmap/uv_renderer.py
