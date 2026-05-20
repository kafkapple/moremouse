"""Build a MAMMAL fitting asset manifest from the dataset config."""

from pathlib import Path
import json

from loguru import logger
from omegaconf import OmegaConf

from moremouse.data import discover_fitting_assets


def main() -> None:
    """Build and write the canonical markerless/MAMMAL fitting manifest."""
    config_path = Path("configs/datasets/markerless_mammal.yaml")
    cfg = OmegaConf.load(config_path)
    dataset = cfg.dataset
    fitting = dataset.fitting

    index = discover_fitting_assets(
        dataset_id=str(dataset.id),
        obj_dirs=tuple(Path(path) for path in fitting.primary.obj_dirs),
        param_dirs=tuple(Path(path) for path in fitting.primary.param_dirs),
        override_obj_dirs=(Path(fitting.overrides.bad_frame_refit.obj_dir),),
        override_param_dirs=(Path(fitting.overrides.bad_frame_refit.param_dir),),
    )

    output_path = Path(dataset.manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": index.dataset_id,
        "asset_count": len(index.assets),
        "assets": [
            {
                "frame_id": asset.frame_id,
                "obj_path": str(asset.obj_path),
                "param_path": str(asset.param_path),
                "source": asset.source,
            }
            for asset in index.assets
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote {} assets to {}", len(index.assets), output_path)


if __name__ == "__main__":
    main()
