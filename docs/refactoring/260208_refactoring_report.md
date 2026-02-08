# MoReMouse Refactoring Report

**Date**: 2026-02-08
**Related**: [Refactoring Plan](260208_refactoring_plan.md) | [Pipeline Overview](../pipeline_overview.md) | [Coordinate Analysis](../reports/251213_moremouse_coordinate_alignment_analysis.md)

---

## Executive Summary

Three-phase refactoring of the MoReMouse codebase to eliminate code duplication,
centralize coordinate system logic, and improve portability.

### Before → After Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| `gaussian_avatar.py` lines | 1,608 | ~1,100 | **-32%** |
| Duplicate function instances | 24 | 0 | **-100%** |
| Hardcoded `/home/joon/` paths | 15+ | 0 (code) | **-100%** |
| Dead code lines | 151+ | 0 | **-100%** |
| Shared utility modules | 0 | 3 | +3 new |
| Coordinate constants scattered | 10+ locations | 1 (`transforms.py`) | **Centralized** |

---

## Phase 1: Urgent Fixes

**Commit**: [`7a5a900`](https://github.com/kafkapple/moremouse/commit/7a5a900)

### 1.1 Dead Code Removal (mammal_loader.py)
- **Problem**: `__getitem__` method at lines 1706-1857 was unreachable — placed after `create_mammal_dataloader()` returns, which never instantiates the class containing it
- **Fix**: Removed 151 lines of orphaned code
- **Root Cause**: Iterative development where the dataloader factory pattern replaced direct class usage

### 1.2 Duplicate quaternion_multiply (gaussian_avatar.py)
- **Problem**: `_quaternion_multiply` class method duplicated the module-level function; 3 methods had redundant `import math` statements
- **Fix**: Removed class method, cleaned up imports

### 1.3 DMTet Render Path (train.py)
- **Problem**: Stage 2 DMTet training loop silently did nothing (renderer not implemented), wasting user time
- **Fix**: Explicit warning + skip when `dmtet_epochs > 0`
- **Bonus**: Updated deprecated `torch.cuda.amp.autocast` → `torch.amp.autocast("cuda")`

### 1.4 README Correction
- **Problem**: Documented "ZYX Euler angle" but MAMMAL actually uses axis-angle (140 joints × 3)
- **Fix**: Corrected to "Axis-angle rotation convention (MAMMAL format)"

---

## Phase 2: Structural Improvements

**Commits**: [`be8b1b3`](https://github.com/kafkapple/moremouse/commit/be8b1b3), [`7797706`](https://github.com/kafkapple/moremouse/commit/7797706), [`d79e051`](https://github.com/kafkapple/moremouse/commit/d79e051)

### 2.1 Coordinate Transform Centralization (transforms.py)

**Why**: Coordinate transform constants and functions were scattered across 10+ files with slight inconsistencies (e.g., different PLATFORM_OFFSET values, manual vs. function-based quaternion multiplication).

**Created `src/utils/transforms.py`** with:
- **Constants**: `PLATFORM_OFFSET`, `WORLD_SCALE_DEFAULT`, `BASE_ROTATION_QUAT`, `CANONICAL_*` series
- **Functions**: `quaternion_multiply()`, `yup_to_zup_means()`, `yup_to_zup_quaternions()`, `apply_coordinate_transform()`, `center_rotation_to_world_translation()`, `project_points()`

**Coordinate System Reference**:
```
MAMMAL body model (Y-up)
    ↓ world_scale × 160.0 (local → mm)
    ↓ -90° X rotation (Y-up → Z-up)
    ↓ yaw rotation (Z-axis, from center_rotation.npz)
    ↓ + world_translation (platform offset + center × 100)
Camera world (Z-up)
```
See also: [Coordinate Alignment Analysis](../reports/251213_moremouse_coordinate_alignment_analysis.md)

### 2.2 Keypoint Geometry Module (geometry.py)

**Why**: `KEYPOINT22_JOINT_MAP`, `BONES`, `extract_keypoints22()`, `draw_skeleton()` were copy-pasted across 4+ scripts with minor inconsistencies.

**Created `src/utils/geometry.py`** as the single source of truth for the 22-keypoint mouse skeleton.

### 2.3 Visualization Extraction (avatar_visualization.py)

**Why**: `GaussianAvatarTrainer` class at 1,608 lines mixed training logic with ~410 lines of visualization helper methods.

**Extracted 7 methods** → `src/utils/avatar_visualization.py`:
`project_points_to_2d`, `draw_model_keypoints`, `draw_gt_keypoints`, `compute_2d_procrustes`, `project_joints_3d_to_cropped`, `draw_model_keypoints_x`, `create_debug_panel`

### 2.4 Environment Variable Automation

**Why**: 15+ hardcoded `/home/joon/` paths made the codebase non-portable.

**Solution**:
1. `.env.example` template (git-tracked) with documented variables
2. `scripts/setup_env.sh` installs conda activate/deactivate hooks
3. `conda activate mouse` → auto-loads `.env` → 5 env vars exported
4. `conda deactivate` → auto-unsets all vars

**Naming conflict resolved**: Scripts' `--data-dir` (NeRF captures) was using `MAMMAL_DATA_DIR` same as `mammal_loader.py` (MAMMAL body data). Split into `NERF_DATA_DIR` vs. `MAMMAL_DATA_DIR`.

---

## Phase 3: DRY Consolidation

### 3.1 Yaw Rotation Helper (transforms.py)

**Why**: Identical 40-line Z-axis yaw rotation block appeared in `gaussian_avatar.py` twice (train_step + _save_visualization) and similar patterns in 2 calibration scripts.

**Added to `transforms.py`**:
- `build_z_rotation_matrix(angles)` → [B, 3, 3] rotation matrices
- `apply_yaw_rotation(yaw, params, extra_points)` → rotates means, quaternions, optional points

**gaussian_avatar.py reduction**: 80 lines → 8 lines (2 call sites)

### 3.2 Script Deduplication (6 files)

| Script | Removed | Now Imports From |
|--------|---------|-----------------|
| `test_procrustes_transform.py` | 5 definitions (58 lines) | `geometry`, `transforms` |
| `compute_optimal_transform.py` | 2 definitions (44 lines) | `geometry` |
| `test_canonical_space.py` | 8 definitions (52 lines) + 4 constants | `geometry`, `transforms` |
| `calibrate_grid_compare.py` | 5 definitions (91 lines) | `geometry`, `transforms` |
| `calibrate_transforms.py` | 2 definitions (30 lines) | `transforms` |
| `render_avatar.py` | 2 definitions (49 lines) | `transforms` |

---

## Remaining Opportunities

| Item | Priority | Description |
|------|----------|-------------|
| Formal test suite | Medium | No pytest — only ad-hoc test scripts |
| `create_rotation_cameras()` | Low | Similar function in `render_avatar.py` and `visualize_3d.py` |
| Quaternion loop vectorization | Low | `apply_yaw_rotation()` uses batch loop; could be fully vectorized |
| DMTet Stage 2 | Future | Training loop skipped — renderer not yet implemented |

---

## File Change Summary

### New Files
| File | Lines | Purpose |
|------|-------|---------|
| `src/utils/transforms.py` | ~250 | Coordinate constants + transform functions |
| `src/utils/geometry.py` | ~140 | Keypoint definitions + extraction + drawing |
| `src/utils/avatar_visualization.py` | ~400 | Extracted visualization helpers |
| `scripts/setup_env.sh` | ~125 | Conda env hook installer |
| `.env.example` | ~25 | Environment variable template |
| `docs/refactoring/260208_refactoring_plan.md` | This doc |
| `docs/refactoring/260208_refactoring_report.md` | This doc |

### Modified Files
| File | Change | Lines Δ |
|------|--------|---------|
| `src/models/gaussian_avatar.py` | Major refactoring | ~-500 |
| `src/data/mammal_loader.py` | Dead code + env vars | ~-160 |
| `scripts/train.py` | DMTet skip + autocast fix | ~+10 |
| `configs/config.yaml` | Env var name alignment | ~2 |
| 8 scripts | Hardcoded paths → env vars + shared imports | ~-400 total |

### Deleted Files
| File | Reason |
|------|--------|
| `src/models/calibrate_transforms.py` | Identical duplicate of `scripts/` version |

---

*MoReMouse Refactoring Report | 2026-02-08*
