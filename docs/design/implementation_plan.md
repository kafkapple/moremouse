# Implementation Plan

## Package Layout

```text
src/moremouse/
  config/          OmegaConf loading and readonly config
  data/            manifests, validation, dataset indexing
  geometry/        mesh contracts, geodesic embedding
  rendering/       camera sampling and render metadata
  training/        reproducibility and sanity checks
  visualization/   grids, videos, debug outputs
```

## Immediate Implementation Order

1. Config loader and result path resolver.
2. Dataset manifest schema and validation.
3. Camera sphere sampler.
4. Geodesic embedding utilities.
5. Visualization grid helpers.
6. GPU03 setup scripts after data source confirmation.

## Experiment Naming

Use:

```text
{date}_{stage}_{dataset_id}_{short_hash}
```

Example:

```text
260520_m0_scaffold_local_a1b2c3d
```

## Artifact Roots

Local:

```text
/Users/joon/results/MoReMouse
```

gpu03:

```text
/home/joon/results/MoReMouse
```

If `/home/joon` is storage-constrained, stop and confirm an alternative such as `/node_data/joon/results/MoReMouse`.

