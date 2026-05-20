# 260520 Initialization

## Done

- Read MoReMouse arXiv v2 and project page.
- Checked coding principles.
- Created local project scaffold.
- Created local conda env `moremouse`.
- Ran scaffold tests: `8 passed in 6.93s`.
- Added gpu03-specific CUDA conda environment file: `environment.gpu03.yml`.
- Created gpu03 conda env `moremouse`.
- Ran gpu03 scaffold tests: `8 passed in 0.33s`.
- Verified gpu03 PyTorch CUDA: torch 2.5.1, CUDA available, 8 devices visible.
- Checked gpu03 resources:
  - disk: about 408GB available on `/`
  - GPUs: 8 RTX PRO 6000 Blackwell Max-Q devices
  - GPU 7 was almost idle at check time
- Found GitHub CLI token is invalid, so private remote creation is blocked until re-authentication.

## Current Blockers

- GitHub private repo creation needs `gh auth login`.
- Canonical dataset and mesh choice must be confirmed before generating dense-view training data.

## Next Step

Run local scaffold tests, then mirror the repository to gpu03 after remote/auth is resolved.
