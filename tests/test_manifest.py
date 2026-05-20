from pathlib import Path

import pytest

from moremouse.data import DatasetManifest, DatasetView


def test_manifest_rejects_duplicate_frame_view(tmp_path: Path) -> None:
    """Reject duplicated frame/view keys in a manifest."""
    image = tmp_path / "image.png"
    camera = tmp_path / "camera.json"
    image.write_bytes(b"placeholder")
    camera.write_text("{}", encoding="utf-8")
    view = DatasetView(frame_id=1, view_id="cam0", image_path=image, camera_path=camera)

    with pytest.raises(ValueError, match="Duplicate"):
        DatasetManifest(dataset_id="smoke", root=tmp_path, split="smoke", views=(view, view))


def test_manifest_rejects_missing_image(tmp_path: Path) -> None:
    """Reject manifests that point at missing image files."""
    camera = tmp_path / "camera.json"
    camera.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        DatasetView(frame_id=1, view_id="cam0", image_path=tmp_path / "missing.png", camera_path=camera)
