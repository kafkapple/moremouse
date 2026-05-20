# Dataset Plan

## Candidate Sources

### MAMMAL

Local repo: `/Users/joon/dev/MAMMAL_mouse`

Relevant known dataset from prior notes:

- `markerless_mouse_1_nerf`
- existing fitting outputs under `results/fitting/`
- mesh refit reports and OBJ outputs

Use for:

- canonical mesh and fitted pose parameters
- visual hull or silhouette quality checks
- AGAM-style avatar control signals

### FaceLift

Local repo: `/Users/joon/dev/FaceLift`

Known configs:

- `configs/datasets/M5t2.yaml`
- `configs/datasets/M5t.yaml`
- `configs/mouse/temporal_multiframe_M5t2.yaml`

Use for:

- real multi-view image format references
- camera conventions
- evaluation/adaptation after canonical source confirmation

### BehaviorSplatter

Local repo: `/Users/joon/dev/BehaviorSplatter`

Use for:

- visualization and camera-ready conventions
- behavior annotations
- existing temporal deformation experiments as a reference, not as a dependency

## Required Manifest Fields

Each dataset split must record:

- dataset id
- root path
- image list path
- camera calibration path
- mask path or mask generation method
- mesh path or mesh parameter path
- frame ids
- view ids
- train/val/test split role
- checksum or generation timestamp

## First Dataset Gate

Decision for the first implementation pass:

- source sequence: `markerless_mouse_1_nerf`
- RGB: `videos_undist`
- masks: `simpleclick_undist`
- 2D keypoints: `keypoints2d_undist`
- 3D keypoints: `add_labels_3d_8keypoints.pkl`
- primary fitting: `production_keyframes_part1..4`
- bad-frame overrides: `refit_accurate_23`

See [Canonical Data Decision](canonical_data_decision.md).

Before training, still confirm:

- exact output cache root on gpu03
- whether synthetic rendered data may be generated from existing MAMMAL outputs
