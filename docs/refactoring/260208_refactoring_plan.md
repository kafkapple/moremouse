# MoReMouse Code Refactoring Plan

**Date**: 2026-02-08
**Status**: Completed (Phase 1-3)
**Related**: [Refactoring Report](260208_refactoring_report.md) | [Pipeline Overview](../pipeline_overview.md)

---

## Motivation

A comprehensive code review identified systemic issues hindering maintainability:

| Category | Count | Severity |
|----------|-------|----------|
| Dead/unreachable code | 151+ lines | High |
| Duplicate functions across files | 24 instances | High |
| Hardcoded user-specific paths | 15+ locations | Medium |
| Inconsistent coordinate transforms | 4+ scattered implementations | High |
| Missing shared utilities | Core functions copy-pasted | Medium |

## Phase 1: Urgent Fixes

**Commit**: `7a5a900` | **Files**: 4

| Task | Description | Impact |
|------|-------------|--------|
| Dead code removal | Orphaned `__getitem__` after `create_mammal_dataloader()` return | -151 lines |
| Duplicate quaternion_multiply | Removed class method duplicate + 3 redundant `import math` | Cleaner model |
| DMTet render path | Silent failure → explicit skip with warning | Prevents confusion |
| README accuracy | "ZYX Euler angle" → "Axis-angle (MAMMAL format)" | Correct documentation |

## Phase 2: Structural Improvements

**Commit**: `be8b1b3` | **Files**: 15

| Task | Description | Impact |
|------|-------------|--------|
| `src/utils/transforms.py` | Centralized coordinate constants & transform functions | Single source of truth |
| `src/utils/geometry.py` | Shared keypoint definitions, extraction, skeleton drawing | Eliminated 4x duplication |
| Visualization extraction | 7 methods from `gaussian_avatar.py` → `avatar_visualization.py` | -416 lines (26% reduction) |
| Duplicate script removal | Deleted `src/models/calibrate_transforms.py` (identical to scripts/) | -315 lines |
| Hardcoded paths | 8 scripts + `mammal_loader.py` → environment variables | Portability |
| Env var standardization | `MOUSE_MODEL_DIR`, `NERF_DATA_DIR`, `MAMMAL_DATA_DIR`, `MAMMAL_POSE_DIR`, `MAMMAL_RESULTS_DIR` | No naming conflicts |
| Conda automation | `scripts/setup_env.sh` + activate/deactivate hooks | Auto-load `.env` |

## Phase 3: DRY Consolidation

**Commit**: (this commit) | **Files**: 10

| Task | Description | Impact |
|------|-------------|--------|
| `build_z_rotation_matrix()` | New shared function in `transforms.py` | Replaces 2 duplicates |
| `apply_yaw_rotation()` | New shared function in `transforms.py` | Replaces 2x ~40-line blocks |
| Yaw rotation dedup | `gaussian_avatar.py`: 80 lines → 8 lines | -72 lines |
| Script imports (6 files) | `test_procrustes_transform.py`, `compute_optimal_transform.py`, `test_canonical_space.py`, `calibrate_grid_compare.py`, `calibrate_transforms.py`, `render_avatar.py` | -~400 lines total |
| Canonical constants | `test_canonical_space.py` local constants → imported from `transforms.py` | Consistency |

## Architecture: Shared Utilities

```
src/utils/
├── __init__.py              # Unified exports
├── transforms.py            # Coordinate systems & math
│   ├── Constants: PLATFORM_OFFSET, WORLD_SCALE_DEFAULT, BASE_ROTATION_QUAT,
│   │              CANONICAL_MESH_SCALE, CANONICAL_CAMERA_RADIUS, etc.
│   ├── quaternion_multiply()
│   ├── yup_to_zup_means(), yup_to_zup_quaternions()
│   ├── apply_coordinate_transform()
│   ├── center_rotation_to_world_translation()
│   ├── build_z_rotation_matrix()      ← Phase 3
│   ├── apply_yaw_rotation()           ← Phase 3
│   └── project_points()
├── geometry.py              # Keypoint & skeleton definitions
│   ├── KEYPOINT22_JOINT_MAP, BONES, KEYPOINT22_NAMES
│   ├── extract_keypoints22()
│   └── draw_skeleton()
├── avatar_visualization.py  # Gaussian avatar debug visualization
│   ├── project_points_to_2d()
│   ├── draw_model_keypoints(), draw_gt_keypoints()
│   ├── compute_2d_procrustes()
│   ├── project_joints_3d_to_cropped()
│   ├── draw_model_keypoints_x()
│   └── create_debug_panel()
├── visualization.py         # General rendering & video utilities
├── metrics.py               # PSNR, SSIM, LPIPS
└── logging.py               # Logging setup
```

## Environment Variables

| Variable | Purpose | Used By |
|----------|---------|---------|
| `MOUSE_MODEL_DIR` | MAMMAL mouse body model | Scripts, Hydra config |
| `MAMMAL_DATA_DIR` | MAMMAL body model data | `mammal_loader.py`, Hydra config |
| `NERF_DATA_DIR` | NeRF multi-view captures | Scripts (`--data-dir`) |
| `MAMMAL_POSE_DIR` | Specific pose experiment | Scripts (`--pose-dir`) |
| `MAMMAL_RESULTS_DIR` | Pose results root | `mammal_loader.py` |

Setup: `bash scripts/setup_env.sh [conda_env_name]`

---

*MoReMouse Refactoring Plan | 2026-02-08*
