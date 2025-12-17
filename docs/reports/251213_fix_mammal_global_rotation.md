# MAMMAL Global Rotation Fix Report

**Date**: 2025-12-13
**Issue**: Rendered view alignment mismatch with GT image
**Status**: Fixed

---

## 1. Problem Description

### Symptom
학습 시 좌측 렌더링 뷰가 top-view처럼 보이고, 우측 GT 이미지는 side-view로 보임.
두 시점이 제대로 정렬되지 않음.

### Root Cause
`train_step`과 `_save_visualization`에서 MAMMAL fitting의 global rotation (mammal_global.R)이 적용되지 않고 있었음.

```python
# 이전 코드 (주석 처리되어 있었음)
# NOTE: Removed base rotation and yaw rotation
# Testing hypothesis: MAMMAL pose already includes global orientation
# Only apply scale + translation
```

MAMMAL body model은 Z-up 좌표계를 사용하고, 카메라 월드 좌표계와 정렬하려면 `mammal_global["R"]` (axis-angle [3])를 적용해야 함.

---

## 2. Solution

### 변환 순서
1. **Scale**: `world_scale` 적용 (body model → world coords)
2. **Rotate**: `mammal_global_R` 적용 (body coords → world coords)
3. **Translate**: `world_trans` 적용 (world 좌표계 내 위치)

### 구현 내용

#### 2.1 `train_step()` 수정

```python
# Apply MAMMAL global rotation (body coords -> world coords)
# Order: scale -> rotate -> translate
if mammal_global_R is not None:
    # Convert axis-angle [B, 3] to rotation matrix [B, 3, 3]
    R = self.avatar.body_model.axis_angle_to_rotation_matrix(
        mammal_global_R.to(self.device)
    )  # [B, 3, 3]

    # Rotate gaussian means: [B, N, 3] @ [B, 3, 3].T -> [B, N, 3]
    means = gaussian_params["means"]  # [B, N, 3]
    rotated_means = torch.einsum('bni,bji->bnj', means, R)
    gaussian_params["means"] = rotated_means

    # Rotate gaussian quaternions
    # (quaternion multiplication for orientation alignment)
    ...
```

#### 2.2 `_save_visualization()` 수정

동일한 회전을 시각화에도 적용하여 렌더링과 GT 이미지가 같은 시점에서 비교되도록 함.
Joint positions도 함께 회전하여 keypoint projection이 올바르게 표시됨.

#### 2.3 Train loop 수정

```python
# Extract global rotation from MAMMAL fitting (axis-angle)
mammal_global_R = None
mammal_valid = mammal_global.get("valid", torch.tensor(False)) if mammal_global is not None else torch.tensor(False)
...
if mammal_valid and mammal_global.get("R") is not None:
    R = mammal_global["R"]  # [3] axis-angle
    mammal_global_R = R.unsqueeze(0) if R.dim() == 1 else R  # [1, 3]

# Training step with world_scale, world_trans, and mammal_global_R
losses = self.train_step(
    pose, target_images, viewmat, K, W, H,
    trans=None, scale=None, world_scale=self._world_scale,
    world_trans=world_trans, mammal_global_R=mammal_global_R
)
```

---

## 3. Helper Function Added

### `_quaternion_multiply()`

Gaussian orientation 회전을 위한 Hamilton product 구현:

```python
def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (Hamilton product).
    Args:
        q1, q2: [N, 4] quaternions in (w, x, y, z) format
    Returns:
        [N, 4] product quaternion q1 * q2
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)
```

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `src/models/gaussian_avatar.py` | `train_step` signature, rotation application, `_quaternion_multiply`, `_save_visualization` |

---

## 5. Data Flow

```
mammal_global (from params.pkl)
├── R: [3] axis-angle rotation
├── T: [3] translation (in pixel coords, not used for 3D)
└── s: scalar scale

world_trans (from center_rotation.npz)
└── center: [3] world position in meters → cm conversion → platform offset

Transformation Order:
1. gaussian_params["means"] *= world_scale
2. gaussian_params["means"] = R @ means (rotation)
3. gaussian_params["means"] += world_trans (translation)
```

---

## 6. Expected Results

수정 후:
- 좌측 렌더링 뷰와 우측 GT 이미지가 같은 시점에서 비교됨
- Keypoint projection이 GT 위치와 정렬됨
- Photometric loss가 올바른 영역에서 계산됨

---

## 7. Testing

```bash
# gpu05에서 테스트 실행
ssh gpu05 "cd /home/joon/moremouse && \
  source ~/anaconda3/etc/profile.d/conda.sh && \
  conda activate moremouse && \
  python scripts/run_pipeline.py --stage avatar \
    --mouse-model /home/joon/MAMMAL_mouse/mouse_model \
    --avatar-iterations 100 --vis-freq 25 --save-freq 50 \
    --device cuda:0"
```

시각화 결과 확인: `/home/joon/moremouse/outputs/avatar_vis/iter_*.png`
