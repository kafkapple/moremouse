# MoReMouse 좌표계 정렬 분석 보고서

**날짜**: 2025-12-13
**주제**: MoReMouse 논문의 AGAM 학습 방식 분석 및 우리 구현과의 차이점

---

## 1. 핵심 발견: MoReMouse는 좌표 정렬 문제를 "회피"했다

### 논문의 핵심 전략

> "To facilitate optimization, we set **Ψ_g = 0** and drive the mesh model using only Ψ_l, resulting in canonical vertex coordinates V_Ψl."

**해석**: Global parameters (translation, rotation, scale)를 **완전히 비활성화**하고, local pose parameters (관절 각도, 뼈 길이)만 사용.

### 이미지 전처리로 문제 해결

> "In preprocessing...apply **translation and scaling transformations to the input images**, ensuring that the mouse centroid is aligned with the image center and normalized to a consistent scale."

**핵심**:
- 메시를 카메라 좌표계로 변환하는 것이 아님
- **이미지를 잘라서** 마우스가 항상 중앙에 오도록 정규화
- 가상의 "마우스 중심 카메라"를 생성하는 효과

---

## 2. MoReMouse vs 우리 구현 비교

| 항목 | MoReMouse | 우리 구현 |
|------|-----------|-----------|
| **Global transform** | Ψ_g = 0 (비활성화) | center, angle 사용 |
| **이미지 처리** | 마우스 중심으로 crop & normalize | 원본 이미지 사용 |
| **좌표 정렬** | 불필요 (canonical space) | world_scale, PLATFORM_OFFSET 필요 |
| **카메라 포즈** | 가상 카메라 (구 위에 배치) | 실제 카메라 포즈 사용 |
| **학습 복잡도** | 단순 | 복잡 (좌표 변환 필요) |

---

## 3. 논문의 한계점 (저자 인정)

> "The current system assumes that the mouse is always **centered in a canonical scene space**. This design **limits our ability to track full 3D poses in a real-world coordinate system**."

**해석**:
- 실제 3D 좌표계에서의 추적은 지원 안 함
- 마우스의 절대 위치/방향 복원 불가
- 상대적 포즈만 학습 가능

---

## 4. 왜 MoReMouse 방식이 작동하는가?

### AGAM 학습 시:
```
Input: cropped image (mouse centered) + local pose Ψ_l
Output: Gaussian parameters on UV space

Camera: synthetic, looking at origin
Mesh: always at origin (no global transform)
```

### 우리 방식의 문제:
```
Input: full image + local pose Ψ_l + global transform (center, angle)
Output: Gaussian parameters

Camera: real camera pose
Mesh: transformed by (scale, rotation, translation)
→ 3개의 좌표계를 정렬해야 함!
```

---

## 5. 해결책 제안

### Option A: MoReMouse 방식 따르기 (권장)
1. 각 프레임에서 마우스 바운딩 박스 계산 (2D keypoints 기반)
2. 이미지를 마우스 중심으로 crop
3. 카메라 intrinsics 재계산 (crop에 맞게)
4. Ψ_g = 0으로 설정 (canonical space)
5. 가상 카메라 배치 (unit sphere)

**장점**: 좌표 정렬 문제 완전 제거
**단점**: 실제 3D 위치 복원 불가

### Option B: 현재 방식 개선
1. MAMMAL global transform의 정확한 의미 파악
2. Learnable offset으로 잔차 보정
3. Per-frame optimization

**장점**: 실제 3D 좌표 유지
**단점**: 캘리브레이션 필요, 오차 누적

---

## 6. 결론

### MoReMouse의 "비밀"
- 좌표 정렬 문제를 **근본적으로 피함**
- 이미지 전처리로 canonical space 생성
- Global transform 완전 무시

### 우리가 겪는 문제의 원인
- **다른 접근법 사용**: 실제 카메라 포즈 + 실제 global transform
- MAMMAL의 좌표계 정의 불명확
- center_rotation.npz의 의미 불분명

### 권장 다음 단계
1. **단기**: 현재 캘리브레이션 (scale=160, neg_yaw) 사용
2. **중기**: MoReMouse 방식 (image crop + canonical space) 구현
3. **장기**: MAMMAL 코드 분석하여 정확한 좌표계 파악

---

## 7. Canonical Space 구현 및 테스트 결과

### 구현 완료 (2025-12-13)

`scripts/test_canonical_space.py` 스크립트 작성:

**MoReMouse 논문 상수:**
```python
MESH_SCALE = 1/180 = 0.005556  # Unit sphere에 들어가도록
CAMERA_RADIUS = 2.22           # 카메라 거리
FOV = 29.86°                   # Field of View
IMAGE_SIZE = 800x800           # 해상도
FOCAL_LENGTH = 1500.15 px      # 계산된 focal length
```

**Canonical 메시 통계:**
```
Bounds: [-0.0016, -0.0016, -0.0025] to [0.0018, 0.0033, 0.0023]
Center: [0.0001, 0.0008, -0.0001]
Extent: 0.0077 (unit sphere 내 ✓)
```

**카메라 배치 (Fibonacci lattice):**
- Camera 0: azimuth=0.0°, elevation=56.4°
- Camera 1: azimuth=222.5°, elevation=30.0°
- Camera 2: azimuth=85.0°, elevation=9.6°
- Camera 3: azimuth=307.5°, elevation=-9.6°
- Camera 4: azimuth=170.0°, elevation=-30.0°
- Camera 5: azimuth=32.5°, elevation=-56.4°

### 시각화 결과

**상단 (Canonical Space):**
- 메시 keypoints가 매우 작은 점으로 표시 (scale 1/180)
- 카메라가 구 위에 배치되어 다양한 각도에서 관찰

**하단 (Real Images - Cropped):**
- GT keypoints 중심으로 crop된 실제 이미지
- 마우스가 이미지 중앙에 정렬됨
- 이 방식으로 canonical space와 유사한 조건 생성

### 핵심 발견

1. **메시가 너무 작음**: 1/180 스케일로 extent=0.0077 → 이미지에서 거의 점으로 표시
2. **Crop 필수**: 실제 이미지를 마우스 중심으로 crop해야 canonical space와 매칭 가능
3. **카메라 intrinsics 조정 필요**: Crop 시 principal point 재계산 필요

---

## 참고

**MoReMouse 논문**: https://arxiv.org/html/2507.04258v2

**핵심 인용**:
- "uniformly scaled down by a factor of 180 so that the mouse fits inside a unit sphere"
- "randomly sample camera centers on a sphere of radius 2.22"
- "we only use 800 frames for training"

**생성된 파일**:
- `scripts/test_canonical_space.py` - Canonical space 테스트 스크립트
- `outputs/canonical_test/canonical_space_comparison.png` - 비교 시각화
- `outputs/canonical_test/canonical_space_info.txt` - 설정 정보
