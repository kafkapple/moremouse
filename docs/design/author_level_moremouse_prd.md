# Author-Level MoReMouse PRD

Date: 2026-05-22

## Objective

Reproduce the paper-level MoReMouse pipeline as closely as the available public paper, project page, AAAI PDF, and local MAMMAL/markerless mouse assets allow. The goal is not only to obtain a working local prototype, but to maintain a staged codebase that can absorb longer AGAM, dense-view, triplane-NeRF, and DMTet training runs without changing the data contract later.

## SSOT Inputs

- Paper: `arXiv:2507.04258`
- Project page: `https://zyyw-eric.github.io/MoreMouse-webpage/`
- AAAI PDF: `https://ojs.aaai.org/index.php/AAAI/article/download/38360/42322`
- Dataset and coordinate references: `docs/references.md`
- Canonical dataset decision: `docs/design/canonical_data_decision.md`
- Existing reproduction plan: `docs/design/author_level_reproduction_plan.md`
- Paper notes: `docs/research_notes/260520_moremouse_paper_notes.md`

## Non-negotiable Constraints

- Use the existing markerless mouse plus MAMMAL-preprocessed assets as canonical local data.
- Do not silently change camera convention or mesh coordinate convention.
- Store all outputs under `/Users/joon/results/MoReMouse`.
- Keep each stage isolated by config, script, report, and tests.
- Fail fast on invalid paths, missing meshes, NaN/Inf tensors, or inconsistent frame indices.

## Target Paper Facts

The public sources support these implementation targets:

- AGAM-style Gaussian mouse avatar built from fitted real multi-view videos.
- Dense synthetic training data rendered from that avatar.
- DINOv2 image encoder.
- Transformer-based image-to-triplane decoder.
- Triplane-NeRF pretraining.
- DMTet fine-tuning.
- Geodesic correspondence embeddings used as a semantic surface prior.

The public sources do not expose the authors' repository, so any exact low-level optimizer or schedule details not described in the paper must be implemented as explicit local assumptions and documented as such.

## Current Local Baseline

The repo already contains:

- MAMMAL fitting asset manifest and projection audits.
- 64-view dense supervision preview generation.
- A working Gaussian surface renderer.
- A DINOv2-backed triplane scaffold.
- A mesh PCA single-view MVP.
- A local full-stack reconstruction prototype.

The PRD below refines that baseline into paper-aligned modules.

## Product Goal

Given a single monocular mouse frame, produce:

- AGAM-style Gaussian avatar parameters.
- Dense multi-view synthetic supervision.
- Triplane NeRF outputs.
- DMTet mesh extraction.
- Per-view RGB, mask, geodesic, normal, and mesh overlays.
- HTML reports with embedded grids and videos.

## Module Breakdown

### 1. Dataset and Convention Layer

Responsibilities:

- Resolve canonical markerless mouse paths.
- Resolve MAMMAL meshes and fit reports.
- Resolve RGB, mask, keypoint, and camera calibration files.
- Normalize frame ids and view ids.
- Enforce the selected camera convention at the boundary.

Inputs:

- `configs/datasets/markerless_mammal.yaml`
- `docs/references.md`
- `docs/design/canonical_data_decision.md`

Outputs:

- Machine-readable manifest objects.
- Audit reports for mesh validity, projection consistency, and frame indexing.

Acceptance:

- No stage can start unless the manifest and camera convention resolve successfully.
- Projection audit and mesh integrity audit pass before training.

### 2. AGAM Stage

Responsibilities:

- Build an anchor-based Gaussian avatar template from a canonical fitted mesh.
- Predict pose-conditioned Gaussian deltas from a monocular input.
- Emit Gaussian centers, RGB, opacity, isotropic scale, and a rotation proxy.
- Render preview grids for train and eval frames.

Why an anchor-based template:

- The paper uses UV texels and anisotropic Gaussian parameters.
- The local codebase has high-quality fitted meshes with consistent topology, but no public author code and no ready differentiable Gaussian rasterizer.
- An anchor-based avatar keeps the parameter count tractable while preserving the paper's key idea: an animatable Gaussian surface avatar driven by fitted geometry.

Acceptance:

- The model can overfit a small subset of frames.
- Predicted avatar renders produce meaningful silhouette overlap on held-out views.
- Preview grids show the avatar is upright under the chosen convention.

### 3. Dense Synthetic Dataset Stage

Responsibilities:

- Render 64-view synthetic supervision from the AGAM avatar.
- Produce per-view RGB proxy, geodesic embedding render, and normal render.
- Save camera metadata, view ids, and source mesh provenance.

Acceptance:

- Dense-view previews are upright and consistent with the camera convention.
- Results are written to a single stable result root.

### 4. Triplane NeRF Stage

Responsibilities:

- Encode monocular input with DINOv2.
- Decode image tokens into triplane features.
- Query the triplanes with a volumetric field.
- Optimize RGB, mask, feature, and depth-style losses on synthetic supervision.

Acceptance:

- A short smoke run reaches finite loss and produces renderable outputs.
- The stage writes a report with PSNR/SSIM/LPIPS-style placeholders or directly computed proxies.

### 5. DMTet Stage

Responsibilities:

- Convert the learned triplane stage into an explicit mesh.
- Run marching tetrahedra on a fixed grid.
- Export OBJ meshes and 6-view overlays.

Acceptance:

- The extracted mesh is non-empty.
- Mesh previews and overlay audits render successfully.

## Loss Design

### AGAM

Primary losses:

- center regression loss
- RGB regression loss
- opacity regression loss
- isotropic scale regression loss
- optional rotation regularizer

Secondary monitoring:

- silhouette IoU on selected views
- preview grid inspection

### Triplane NeRF

Primary losses:

- RGB MSE or Smooth L1
- mask BCE
- feature embedding regression
- depth consistency inside object mask
- regularization on triplane magnitude

### DMTet

Primary losses:

- RGB and silhouette consistency
- mesh regularization
- normal smoothness
- surface compactness

## Training Schedule

### Stage A: AGAM Pilot

- Train on a small canonical frame subset first.
- Use a fixed template built from a canonical fitted mesh.
- Overfit until the avatar renders stable silhouettes and colors.

### Stage B: AGAM Scaling

- Expand from pilot subset to the canonical train range.
- Keep the same template and anchor ids.
- Add periodic preview rendering and report snapshots.

### Stage C: Dense Supervision Generation

- Render all required orbit views for the selected train poses.
- Write per-frame asset triplets and grid previews.

### Stage D: Triplane Training

- Train on the generated dense views.
- Start from frozen DINO tokens if possible to reduce early instability.

### Stage E: DMTet Fine-Tuning

- Fine-tune the explicit mesh stage after the triplane stage converges.
- Export final meshes and comparison grids.

## Required Code Modules

- `src/moremouse/data/agam.py`
- `src/moremouse/models/agam.py`
- `src/moremouse/training/agam_losses.py`
- `scripts/train_agam.py`
- `scripts/render_author_level_dense_dataset.py`
- `scripts/train_triplane_nerf.py`
- `scripts/train_dmtet.py`
- `scripts/reproduce_author_level_moremouse.py`
- `configs/experiments/moremouse_author_level.yaml`

## Required Visualization Artifacts

- input/target/predicted AGAM preview grid
- 64-view dense supervision preview grid
- 6-view reconstruction comparison grid
- mesh preview triplets
- camera projection audit grids
- HTML summary with embedded figures

## Stop Conditions

Stop and ask before scaling up if:

- the camera convention is uncertain,
- the canonical mesh source changes,
- the MAMMAL manifest is missing or inconsistent,
- the preview grid shows a sideways or inverted mouse,
- the long-run job would exceed local GPU or disk budget.

## Immediate Next Steps

1. Implement the anchor-based AGAM data/model/loss modules.
2. Add the author-level experiment config.
3. Add the AGAM training script and preview writer.
4. Run a small gpu03 pilot and inspect the output grids.
5. Roll the validated settings into the broader triplane and DMTet stages.
