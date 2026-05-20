"""Image grid generation."""

from pathlib import Path

from PIL import Image


def make_image_grid(image_paths: tuple[Path, ...], columns: int, output_path: Path) -> Path:
    """Create a fixed-cell image grid.

    Parameters
    ----------
    image_paths:
        Input image paths.
    columns:
        Number of grid columns.
    output_path:
        Output image path.

    Returns
    -------
    Path
        Written output path.
    """
    if not image_paths:
        raise ValueError("image_paths must not be empty")
    if columns <= 0:
        raise ValueError("columns must be positive")
    images = [Image.open(path).convert("RGB") for path in image_paths]
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("All images must have the same size")
    rows = (len(images) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), color=(255, 255, 255))
    for index, image in enumerate(images):
        x = index % columns * width
        y = index // columns * height
        grid.paste(image, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    return output_path

