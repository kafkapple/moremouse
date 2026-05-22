# Full Reproduction Checklist

Date: 2026-05-22

## Purpose

This checklist fixes the author-level MoReMouse work into explicit gates so long-running training can be resumed, audited, and reported without changing data contracts midway.

## SSOT Inputs

- [References](../references.md)
- [Author-Level MoReMouse PRD](author_level_moremouse_prd.md)
- [Author-Level Reproduction Plan](author_level_reproduction_plan.md)
- [Test and Verification Plan](test_verification_plan.md)

## Hard Constraints

- Use markerless mouse + MAMMAL-preprocessed assets as the local canonical dataset.
- Keep camera convention explicit and unchanged at every boundary.
- Keep all outputs under `/Users/joon/results/MoReMouse`.
- Use conda environment `moremouse` on gpu03.
- Stop before scale-up if the mesh source, manifest, or camera convention is uncertain.

## Checklist

### A. Data and Convention Gate

- [x] Confirm dataset and camera SSOT locations in `docs/references.md`.
- [x] Resolve canonical fitted mesh source for the chosen frame range.
- [x] Verify projection audit on the canonical frame set.
- [x] Verify dense preview world convention and upright orientation.

### B. AGAM Gate

- [x] Implement anchor-based AGAM proxy modules.
- [x] Train pilot AGAM on gpu03.
- [x] Save eval grids for frames 0, 2760, and 10080.
- [x] Save dense preview diagnostics.
- [ ] Reproduce author-scale AGAM optimizer/loss schedule exactly as paper code.
- [ ] Run a multi-day AGAM training schedule on the full frame range.

### C. Dense Supervision Gate

- [x] Render 64-view synthetic supervision artifacts.
- [x] Store Gaussian, geodesic, and normal previews.
- [ ] Generate the full author-scale dense dataset used for triplane training.
- [ ] Save manifest with per-frame camera metadata and provenance.

### D. Triplane Gate

- [x] Implement DINOv2-backed triplane reconstruction module.
- [x] Maintain a local triplane proxy training path.
- [ ] Match the paper's dense supervision training regime.
- [ ] Run long-horizon triplane optimization on the generated dense dataset.

### E. DMTet Gate

- [x] Implement marching-tetrahedra extraction utilities.
- [x] Save a DMTet proxy artifact and preview.
- [ ] Fine-tune a paper-scale DMTet stage.
- [ ] Export final mesh and multi-view comparison grids.

### F. Reporting Gate

- [x] Embed stage outputs into the local HTML report.
- [ ] Add the final stage-by-stage long-run metrics table.
- [ ] Add final paper-style conclusions and residual gap analysis.

## Stage Schedule

1. AGAM pilot and scaling.
2. Dense dataset rendering.
3. Triplane proxy training and then long-run triplane training.
4. DMTet proxy and then long-run DMTet refinement.
5. Final HTML consolidation.

## Resource Policy

- Check GPU free memory before each long-run stage.
- Check `/home/joon` free space before each stage and before large artifact writes.
- Do not start a stage if the selected GPU has no safe headroom.
- Keep logs, checkpoints, and previews under stage-specific output directories.

## Resume Rule

Any stage can be resumed only from the last valid artifact snapshot for that stage, with the matching config snapshot preserved in the report.
