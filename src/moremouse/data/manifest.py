"""Dataset manifest schema."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class DatasetView(BaseModel, frozen=True):
    """One frame/view observation in a multi-view mouse dataset."""

    frame_id: int = Field(ge=0)
    view_id: str = Field(min_length=1)
    image_path: Path
    camera_path: Path
    mask_path: Path | None = None

    @field_validator("image_path", "camera_path", "mask_path")
    @classmethod
    def validate_existing_path(cls, value: Path | None) -> Path | None:
        """Require explicitly declared files to exist."""
        if value is not None and not value.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {value}")
        return value


class DatasetManifest(BaseModel, frozen=True):
    """Validated dataset manifest used at training boundaries."""

    dataset_id: str = Field(min_length=1)
    root: Path
    split: str = Field(pattern="^(train|val|test|smoke)$")
    mesh_path: Path | None = None
    views: tuple[DatasetView, ...]

    @field_validator("root", "mesh_path")
    @classmethod
    def validate_root_path(cls, value: Path | None) -> Path | None:
        """Require root and optional mesh files to exist."""
        if value is not None and not value.exists():
            raise FileNotFoundError(f"Manifest path does not exist: {value}")
        return value

    @model_validator(mode="after")
    def validate_unique_frame_views(self) -> "DatasetManifest":
        """Reject duplicated frame/view observations."""
        keys = [(view.frame_id, view.view_id) for view in self.views]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate frame_id/view_id entries in manifest")
        return self

