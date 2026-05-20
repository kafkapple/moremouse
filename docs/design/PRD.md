# MoReMouse PRD

## Objective

Build a reproducible local implementation and evaluation pipeline for MoReMouse-style monocular dense 3D mouse reconstruction, using existing FaceLift, BehaviorSplatter, and MAMMAL assets where appropriate.

## Final Goal

Given a single mouse image, produce:

- dense geometry or mesh
- novel-view RGB renders
- normal maps
- geodesic semantic embedding renders
- image grids and videos for inspection
- quantitative metrics against held-out views or masks

## Non-goals for Phase 1

- Full paper-quality reproduction on day one.
- Changing existing FaceLift, BehaviorSplatter, or MAMMAL repos.
- Using ambiguous datasets without explicit confirmation.

## Constraints

- Use an isolated conda environment named `moremouse`.
- Store generated results under `/Users/joon/results/MoReMouse`.
- On gpu03, check disk and GPU state before long jobs.
- Use OmegaConf YAML config; avoid hardcoded experiment constants.
- Use loguru, not print.
- Add tests for every public utility.
- Fail fast on invalid paths, shapes, NaN, or Inf.

## Milestones

### M0: Repository and Documentation

- Create local repo.
- Write paper notes, PRD, dataset plan, and verification plan.
- Confirm gpu03 resources.

Exit criteria:

- Docs exist and are linked from `docs/README.md`.
- `pytest` passes for scaffold modules.

### M1: Data Audit

- Audit local and gpu03 paths for markerless mouse, FaceLift M5/M5t2, BehaviorSplatter annotations, and MAMMAL outputs.
- Produce a machine-readable dataset manifest.

Exit criteria:

- No training starts until canonical source data is confirmed.
- Manifest records image lists, camera files, meshes, masks, and fitted parameters.

### M2: Geometry Priors

- Implement mesh loading contract.
- Implement geodesic embedding on small meshes and MAMMAL mesh.
- Render/debug color embedding grids.

Exit criteria:

- Unit tests cover exact small-graph geodesics and embedding validation.
- A saved grid image validates color separability.

### M3: Dense-view Data Builder

- Build camera sphere sampler.
- Build dataset writer using consistent metadata schema.
- Generate a small smoke dataset.

Exit criteria:

- One smoke split can be loaded by tests.
- Grid and video outputs are generated.

### M4: NeRF-stage Model

- Implement DINOv2 feature extraction interface.
- Implement transformer to triplane scaffold.
- Train a small NeRF-stage overfit run.

Exit criteria:

- Single-batch sanity check passes.
- Overfit run improves PSNR on a tiny subset.

### M5: DMTet-stage Model

- Add DMTet or an equivalent explicit surface backend.
- Export meshes and normals.

Exit criteria:

- Mesh export and normal renders are valid.
- Quantitative metrics are logged.

## Open Decisions

- Canonical mesh and fitting source: decided for first pass as MAMMAL accurate 6-view production keyframes with `refit_accurate_23` overrides.
- Whether to port code from BehaviorSplatter GSLRM modules or keep this repo independent.
- GPU03 result root and dataset cache root.
- Whether to use official MoReMouse code if released later.
