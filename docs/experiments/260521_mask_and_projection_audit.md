# 260521 Mask and Projection Audit

## Mask Finding

The segmentation videos under `simpleclick_undist/{0..5}.mp4` are compressed mp4 files, not lossless binary masks. Extracted mask frames contain many gray values between 0 and 255.

Example for frame 6000 view 0:

- unique gray values: 220
- foreground ratio with `mask > 0`: `0.06270938449435765`
- foreground ratio with `mask > 127`: `0.05976528591579861`

The earlier sanity overlay incorrectly used `mask > 0`, so low-valued compression artifacts were included as foreground. That made the mask look thicker or different from the expected GT mask. The code now uses YAML-configured `dataset.visualization.mask_binary_threshold: 127` for overlays and tiny mask training targets.

## Keypoint Color Meaning

The first overlay used cyclic colors only for visual separation. It was not a semantic color legend. The updated code fixes a stable 22-index keypoint palette matching the BehaviorSplatter keypoint order:

`L_ear, R_ear, nose, neck, body_middle, tail_root, tail_middle, tail_end, L_paw, L_paw_end, L_elbow, L_shoulder, R_paw, R_paw_end, R_elbow, R_shoulder, L_foot, L_knee, L_hip, R_foot, R_knee, R_hip`.

## Mask/Keypoint Algorithm Used

No new segmentation or keypoint estimation was run. The scripts only loaded existing assets:

- RGB: `videos_undist/{view}.mp4`
- mask: `simpleclick_undist/{view}.mp4`
- 2D keypoints: `keypoints2d_undist/result_view_{view}.pkl`

The sanity algorithm was:

1. extract RGB and mask frames with ffmpeg
2. threshold mask with `mask > 127`
3. overlay mask in cyan on RGB
4. draw visible 2D keypoint coordinates using a stable index palette
5. write grid images and mp4

## Camera Convention Finding

For MAMMAL `new_cam.pkl`, the convention that matches image/mask coordinates is:

```text
x_cam = R @ x_world + T
pixel_h = K @ x_cam
pixel = pixel_h[:2] / pixel_h[2]
```

Frame 6000, view 0:

- mask bbox: `(315, 546, 636, 954)`
- projected mesh bbox under `R @ X + T`: `(319.9, 557.3, 628.7, 948.7)`
- all tested incorrect alternatives (`R.T @ X + T`, `R @ (X-T)`, `R.T @ (X-T)`) fail by depth, image bounds, or bbox mismatch.

## Mesh Interpretation

The earlier `mammal_mesh_preview_260521` orthographic preview is useful for topology and gross shape, but it is not a camera-aligned quality test. It ignores real camera intrinsics/extrinsics and can make a valid fitted mesh look bad from arbitrary axes. The next diagnostic is therefore camera-projected mesh overlay against RGB/mask, not free orthographic preview.
