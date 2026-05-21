# 260520 Initialization

## Done

- Read MoReMouse arXiv v2 and project page.
- Checked coding principles.
- Created local project scaffold.
- Created local conda env `moremouse`.
- Ran scaffold tests: `8 passed in 6.93s`.
- Added gpu03-specific CUDA conda environment file: `environment.gpu03.yml`.
- Created gpu03 conda env `moremouse`.
- Initial gpu03 PyTorch 2.5.1/cu121 env reported CUDA availability but failed real kernels on Blackwell `sm_120`.
- Rebuilt gpu03 env from `environment.gpu03.yml` with PyTorch 2.7.1+cu128 and torchvision 0.22.1+cu128.
- Ran gpu03 scaffold tests: `11 passed in 0.34s`.
- Verified gpu03 CUDA kernel execution: torch 2.7.1+cu128, CUDA 12.8, 8 devices visible, arch list includes `sm_120`, 1024x1024 CUDA matmul passed.
- Confirmed first dataset direction: `markerless_mouse_1_nerf` with MAMMAL preprocessing, segmentation masks, keypoints, and accurate MAMMAL fitting outputs.
- Built gpu03 fitting asset manifest at `/home/joon/results/MoReMouse/datasets/markerless_mouse_1_nerf_mammal_accurate_manifest.json`.
- Manifest contains 900 fitted keyframe assets: 877 primary production keyframes and 23 `refit_accurate_23` overrides.
- Checked gpu03 resources:
  - disk: about 408GB available on `/`
  - GPUs: 8 RTX PRO 6000 Blackwell Max-Q devices
  - GPUs 0, 4, 5, 6, and 7 were effectively idle at check time; GPU 1 was using about 50GB.
  - `/node_data` is 96% full, so MoReMouse generated outputs should stay under `/home/joon/results/MoReMouse` unless storage is revisited.
- Force-pushed the canonical scaffold to the existing private GitHub repo after the prior failed remote contents were declared disposable.

## Current Blockers

- Dense-view generation has not started; first run should begin with manifest extraction and visual sanity checks.

## Next Step

Run local scaffold tests and coding-principles checks, commit the Blackwell CUDA env fix, then sync gpu03 with `origin/main`.
