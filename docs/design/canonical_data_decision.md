# Canonical Data Decision

Date: 2026-05-20

## Decision

Use `markerless_mouse_1_nerf` as the first canonical source dataset, with MAMMAL preprocessing and high-quality MAMMAL fitting outputs.

## Source Dataset

gpu03 root:

```text
/home/joon/dev/MAMMAL_mouse/data/raw/markerless_mouse_1_nerf
```

Required components found:

- RGB videos: `videos_undist/{0..5}.mp4`
- segmentation masks: `simpleclick_undist/{0..5}.mp4`
- 2D keypoints: `keypoints2d_undist/result_view_{0..5}.pkl`
- 3D keypoints: `add_labels_3d_8keypoints.pkl`
- camera calibration: `new_cam.pkl`, `camera_params.h5`, `center_rotation.npz`

## Mesh/Fitting Choice

Primary fitting source:

```text
/home/joon/dev/MAMMAL_mouse/results/fitting/production_keyframes_part{1,2,3,4}
```

These are 900 accurate 6-view MAMMAL keyframes at interval 20:

- part1: frames 000000 to 004480, 225 OBJ and 225 step-2 params
- part2: frames 004500 to 008980, 225 OBJ and 225 step-2 params
- part3: frames 009000 to 013480, 225 OBJ and 225 step-2 params
- part4: frames 013500 to 017980, 225 OBJ and 225 step-2 params

Override fitting source for known bad frames:

```text
/home/joon/dev/MAMMAL_mouse/results/fitting/refit_accurate_23
```

This contains 23 accurate refits and should override production keyframes for those exact frame ids.

## Rationale

The MAMMAL refit report identifies accurate fitting with six views, 22 keypoints, and `mask_step2=3000` as the best available high-quality path. The same report notes a sweep where `s1=400, mask_step2=3000` was best overall, but the available production-scale artifacts use the near-optimal accurate config (`s1=200, mask_step2=3000`) and are already generated for 900 keyframes. For immediate MoReMouse work, available high-quality generated artifacts are more valuable than an ungenerated marginally better setting.

## Training Use

- Stage M1 manifest: video frame extraction and validation only.
- Stage M2 geodesic/mesh validation: use production keyframe OBJ/params, with `refit_accurate_23` overrides.
- Stage M3 dense-view synthetic generation: render from MAMMAL-fitted meshes/params after visual checks.
- Stage M4/M5 training: use synthetic dense-view data generated from the validated MAMMAL source.

## Open Risk

Some frames not in the 900 keyframe set require interpolation or additional fitting. Do not silently use nearest-neighbor mesh for training frames unless the experiment config explicitly says so.
