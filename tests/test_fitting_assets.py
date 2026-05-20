from pathlib import Path

import pytest

from moremouse.data import discover_fitting_assets, parse_step2_frame_id


def _write_pair(root: Path, frame_id: int) -> tuple[Path, Path]:
    """Create a matching OBJ/parameter pair for a frame."""
    obj_dir = root / "obj"
    param_dir = root / "params"
    obj_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    (obj_dir / f"step_2_frame_{frame_id:06d}.obj").write_text("# obj\n", encoding="utf-8")
    (param_dir / f"step_2_frame_{frame_id:06d}.pkl").write_bytes(b"pkl")
    return obj_dir, param_dir


def test_parse_step2_frame_id() -> None:
    """Parse frame ids from MAMMAL step-2 asset names."""
    frame_id = parse_step2_frame_id(Path("step_2_frame_001320.obj"))

    assert frame_id == int("1320")


def test_discover_fitting_assets_applies_override(tmp_path: Path) -> None:
    """Use override assets when the same frame exists in primary and override dirs."""
    primary_obj, primary_param = _write_pair(tmp_path / "primary", int("20"))
    override_obj, override_param = _write_pair(tmp_path / "override", int("20"))

    index = discover_fitting_assets(
        dataset_id="markerless",
        obj_dirs=(primary_obj,),
        param_dirs=(primary_param,),
        override_obj_dirs=(override_obj,),
        override_param_dirs=(override_param,),
    )

    assert len(index.assets) == int("1")
    assert index.assets[0].obj_path.parent == override_obj
    assert index.assets[0].source == "markerless:override"


def test_discover_fitting_assets_rejects_missing_param(tmp_path: Path) -> None:
    """Reject incomplete OBJ/parameter pairs."""
    obj_dir = tmp_path / "obj"
    param_dir = tmp_path / "params"
    obj_dir.mkdir()
    param_dir.mkdir()
    (obj_dir / "step_2_frame_000020.obj").write_text("# obj\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing parameter"):
        discover_fitting_assets("markerless", (obj_dir,), (param_dir,))
