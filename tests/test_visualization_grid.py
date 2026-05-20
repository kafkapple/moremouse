from pathlib import Path

from PIL import Image

from moremouse.visualization import make_image_grid


def test_make_image_grid(tmp_path: Path) -> None:
    """Write a fixed-size image grid from equal-sized images."""
    paths = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 6), color=(index * 20, 0, 0)).save(path)
        paths.append(path)

    output = make_image_grid(tuple(paths), columns=2, output_path=tmp_path / "grid.png")

    assert output.exists()
    assert Image.open(output).size == (16, 12)
