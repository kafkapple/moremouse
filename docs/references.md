# References

This file is the living SSOT index for dataset paths, camera conventions, and cross-system transforms used by MoReMouse.

## Read First

- `docs/README.md` - project docs MOC
- `docs/design/PRD.md` - overall product/research scope
- `docs/design/canonical_data_decision.md` - canonical dataset and fitting-source decision
- `docs/research_notes/260520_moremouse_paper_notes.md` - paper-derived implementation notes

## Dataset SSOT

| Item | SSOT path |
| --- | --- |
| Mouse M5 preprocessed | `~/data/preprocessed/FaceLift_mouse/M5/` |
| MAMMAL raw (mouse) | `~/data/raw/MAMMAL_mouse/` |
| Rat7M preprocessed | `/node_data/rat7m/frames/{s1d1_v3,combined_v3}/` |
| Dataset config | `configs/datasets/{M5t2,Rat7M_s1d1_v3,Rat7M_combined_v3}.yaml` |
| Preprocessing config | `configs/preprocessing/rat7m_s1d1*.yaml` |
| Preprocessing theory | `docs/preprocessing_theory.md` |
| Reproduction guide | `reproducibility/icml2026/REPRODUCE_paper_260520.md` |

## Camera / Coordinate SSOT

| Topic | SSOT path |
| --- | --- |
| Global camera summary | `CLAUDE.md` section `Camera Conventions` |
| BehaviorSplatter camera notes | `docs/camera_conventions.md` |
| FaceLift calibration model | `mouse_extensions/docs/CAMERA_CALIBRATION.md` |
| FaceLift coordinate systems | `mouse_extensions/docs/COORDINATE_SYSTEMS.md` |
| Cross-system transforms | `CLAUDE.md` section `Cross-System Coordinate Transforms` |
| Transform artifact | `_archive/common_paper_260416/results/metrics/mammal_to_ps_transform.json` |

## Canonical Conventions

- OpenCV camera convention: `X-right`, `Y-down`, `Z-forward`
- Visualization world for dense preview: `Z-up`
- Default intrinsics used in this project: `fx = fy = 548.9938`, `cx = cy = 256` for `512x512`
- FaceLift M5 scene center: `[59.672, 51.517, 107.099]`
- FaceLift M5 distance scale: `2.7 / 307.785 ≈ 0.008781`

## Working Rule

Before touching dataset loading, mesh rendering, or camera projection code, verify the active convention against the files above and keep that choice explicit in code, docs, and reports.
