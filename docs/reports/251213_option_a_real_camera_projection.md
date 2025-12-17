# Option A 구현: Real Camera Projection for Keypoint Visualization

- **날짜**: 2024-12-13
- **주제**: MoReMouse 원본 방식 (Option A) 키포인트 시각화 구현
- **목적**: Model 키포인트와 GT 키포인트를 올바르게 비교하기 위한 투영 방식 구현

---

## 배경

### 문제 상황
기존 시각화에서 Model 키포인트(X 마커)와 GT 키포인트(원형)가 서로 다른 좌표계를 사용:
- **GT 키포인트**: Cropped 실제 이미지 좌표 (MAMMAL에서 제공)
- **Model 키포인트**: Synthetic 카메라로 투영된 좌표

### 두 가지 접근 방식

| Option | 방법 | 장점 | 단점 |
|--------|------|------|------|
| **A (MoReMouse 원본)** | Real camera → crop transform | 실제 pose estimation error 표시 | Canonical space에서는 오차가 커 보임 |
| **B (Procrustes)** | Procrustes alignment | 깔끔한 시각화 | 실제 error를 숨김 |

**선택**: Option A (MoReMouse 원본 방식)

---

## 구현 내용

### 1. Real Camera Parameters 반환 (`mammal_loader.py`)

```python
def _get_real_camera_params(self, cam_idx: int) -> Dict[str, torch.Tensor]:
    """Get real MAMMAL camera parameters for Option A projection."""
    cam_params = self._base_dataset.get_camera_params(cam_idx)
    K_original = np.array(cam_params['K'])

    # Undo resize scaling to get original image K
    orig_width, orig_height = 1152, 1024
    base_image_size = self._base_dataset.image_size
    scale_x = orig_width / base_image_size
    scale_y = orig_height / base_image_size
    K_original[0, 0] *= scale_x  # fx
    K_original[0, 2] *= scale_x  # cx
    K_original[1, 1] *= scale_y  # fy
    K_original[1, 2] *= scale_y  # cy

    viewmat = np.array(cam_params['viewmat'])

    return {
        'K': torch.from_numpy(K_original).float(),
        'viewmat': torch.from_numpy(viewmat).float(),
        'orig_width': torch.tensor(orig_width, dtype=torch.float32),
        'orig_height': torch.tensor(orig_height, dtype=torch.float32),
    }
```

**핵심**: MAMMAL의 K는 resize된 이미지 기준이므로, 원본 이미지 좌표로 복원 필요.

### 2. 3D → Cropped 2D 투영 (`gaussian_avatar.py`)

```python
def _project_joints_3d_to_cropped(
    self,
    joints_3d: np.ndarray,  # [140, 3] 3D joints in world space
    real_camera: Dict,       # K, viewmat from MAMMAL calibration
    crop_info: Dict,         # x1, y1, scale from cropping
) -> np.ndarray:
    """Project 3D joints to cropped image coordinates (Option A)."""

    # Step 1: 3D → Camera space
    R = viewmat[:3, :3]
    T = viewmat[:3, 3]
    joints_cam = (R @ joints_3d.T).T + T

    # Step 2: Camera space → Original image 2D
    joints_proj = (K @ joints_cam.T).T
    z = np.maximum(joints_proj[:, 2:3], 1e-6)
    joints_2d_orig = joints_proj[:, :2] / z

    # Step 3: Original image → Cropped image (crop transform)
    joints_2d_cropped[:, 0] = (joints_2d_orig[:, 0] - x1) * scale
    joints_2d_cropped[:, 1] = (joints_2d_orig[:, 1] - y1) * scale

    return joints_2d_cropped
```

### 3. 투영 파이프라인 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    Option A Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Model 3D Joints (World Space)                                  │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────┐                                           │
│  │ Real viewmat    │  R @ joints + T                           │
│  │ (MAMMAL calib)  │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  Camera Space Joints                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Real K          │  K @ joints / z                           │
│  │ (Original size) │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  Original Image 2D (1152 x 1024)                               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Crop Transform  │  (x - x1) * scale                         │
│  │ (x1, y1, scale) │  (y - y1) * scale                         │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  Cropped Image 2D (378 x 378)  ←── GT Keypoints도 이 좌표계    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 결과 분석

### 시각화 파일 위치
- **gpu05**: `/home/joon/moremouse/outputs/avatar_vis/iter_000050.png`
- **로컬 복사**: `/tmp/option_a_vis_iter50.png`

### 현재 결과 해석
X 마커(Model)가 원형(GT)과 많이 떨어져 있음:

**이유**: Canonical Space 모드에서 학습 중
- `Ψ_g = 0` (global transform 없음)
- 메쉬가 원점에 scale 1/180으로 위치
- Real camera로 투영하면 당연히 맞지 않음

**이것이 정상**: Option A는 실제 pose estimation error를 보여주는 것이 목적.
Global pose가 학습되면 X와 ○가 일치하기 시작함.

---

## 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/data/mammal_loader.py` | `_get_real_camera_params()` 추가, `__getitem__`에 `real_camera` 반환 |
| `src/models/gaussian_avatar.py` | `_project_joints_3d_to_cropped()` 추가, `_save_visualization()` 수정 |

---

## 다음 단계

1. **Global Pose 학습 활성화**: `Ψ_g` 파라미터 학습으로 Model이 실제 위치로 이동
2. **Keypoint Loss 활성화**: 투영된 Model 키포인트와 GT 키포인트 간 loss 계산
3. **수렴 확인**: X와 ○가 점점 가까워지는지 시각화로 확인

---

## 핵심 학습 포인트

1. **좌표계 일관성**: GT와 Model을 비교하려면 같은 좌표계 사용 필수
2. **Crop Transform**: MAMMAL은 원본 이미지를 crop하므로, 투영 후 crop 변환 필요
3. **K 행렬 스케일링**: resize된 이미지의 K를 원본 이미지 K로 복원해야 함
