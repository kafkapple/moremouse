"""MAMMAL fitting asset discovery."""

from pathlib import Path
import re

from pydantic import BaseModel, Field, field_validator, model_validator


FRAME_PATTERN = re.compile(r"step_2_frame_(\d+)\.(obj|pkl)$")


class FittingAsset(BaseModel, frozen=True):
    """One MAMMAL fitted frame asset pair."""

    frame_id: int = Field(ge=0)
    obj_path: Path
    param_path: Path
    source: str = Field(min_length=1)

    @field_validator("obj_path", "param_path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        """Require fitting files to exist."""
        if not value.exists():
            raise FileNotFoundError(f"Fitting asset does not exist: {value}")
        return value


class FittingAssetIndex(BaseModel, frozen=True):
    """Validated index of MAMMAL fitting assets."""

    dataset_id: str = Field(min_length=1)
    assets: tuple[FittingAsset, ...]
    override_sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_unique_frames(self) -> "FittingAssetIndex":
        """Reject duplicated frame ids after override resolution."""
        frame_ids = [asset.frame_id for asset in self.assets]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("Duplicate frame ids in fitting asset index")
        return self


def parse_step2_frame_id(path: Path) -> int:
    """Parse a MAMMAL step-2 frame id from an asset path."""
    match = FRAME_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Not a step-2 MAMMAL asset path: {path}")
    return int(match.group(1))


def discover_fitting_assets(
    dataset_id: str,
    obj_dirs: tuple[Path, ...],
    param_dirs: tuple[Path, ...],
    override_obj_dirs: tuple[Path, ...] = (),
    override_param_dirs: tuple[Path, ...] = (),
) -> FittingAssetIndex:
    """Discover OBJ/parameter pairs with override directories applied last."""
    if len(obj_dirs) != len(param_dirs):
        raise ValueError("obj_dirs and param_dirs must have the same length")
    if len(override_obj_dirs) != len(override_param_dirs):
        raise ValueError("override obj/param dirs must have the same length")

    assets_by_frame: dict[int, FittingAsset] = {}
    for obj_dir, param_dir in zip(obj_dirs, param_dirs, strict=True):
        _add_assets_from_pair(dataset_id, obj_dir, param_dir, "primary", assets_by_frame)
    for obj_dir, param_dir in zip(override_obj_dirs, override_param_dirs, strict=True):
        _add_assets_from_pair(dataset_id, obj_dir, param_dir, "override", assets_by_frame)

    assets = tuple(assets_by_frame[frame_id] for frame_id in sorted(assets_by_frame))
    return FittingAssetIndex(dataset_id=dataset_id, assets=assets, override_sources=("override",))


def _add_assets_from_pair(
    dataset_id: str,
    obj_dir: Path,
    param_dir: Path,
    source: str,
    assets_by_frame: dict[int, FittingAsset],
) -> None:
    """Add all matched OBJ/parameter pairs from one directory pair."""
    if not obj_dir.exists():
        raise FileNotFoundError(f"OBJ directory does not exist: {obj_dir}")
    if not param_dir.exists():
        raise FileNotFoundError(f"Parameter directory does not exist: {param_dir}")

    for obj_path in sorted(obj_dir.glob("step_2_frame_*.obj")):
        frame_id = parse_step2_frame_id(obj_path)
        param_path = param_dir / f"step_2_frame_{frame_id:06d}.pkl"
        if not param_path.exists():
            raise FileNotFoundError(f"Missing parameter file for frame {frame_id}: {param_path}")
        assets_by_frame[frame_id] = FittingAsset(
            frame_id=frame_id,
            obj_path=obj_path,
            param_path=param_path,
            source=f"{dataset_id}:{source}",
        )
