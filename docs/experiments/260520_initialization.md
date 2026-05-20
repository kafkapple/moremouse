# 260520 Initialization

## Done

- Read MoReMouse arXiv v2 and project page.
- Checked coding principles.
- Created local project scaffold.
- Created local conda env `moremouse`.
- Ran scaffold tests: `8 passed in 6.93s`.
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
