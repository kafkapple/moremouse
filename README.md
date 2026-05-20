# MoReMouse

Monocular dense 3D reconstruction of laboratory mice, based on the MoReMouse paper and adapted for local FaceLift, BehaviorSplatter, and MAMMAL assets.

## Status

This repository is in the planning and validation scaffold stage. The first milestones are:

1. Validate local and gpu03 data availability.
2. Build the synthetic dense-view data contract.
3. Implement geodesic correspondence embedding.
4. Add visualization outputs: image grids, camera sweeps, and videos.
5. Train and evaluate NeRF-stage, then DMTet-stage reconstruction.

## Documentation

Start from [docs/README.md](docs/README.md).

## Results

All generated artifacts must live under:

```text
/Users/joon/results/MoReMouse
```

On gpu03, use the equivalent project result root after confirming disk layout.

