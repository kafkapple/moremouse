"""Train a tiny RGB-to-mask overfit baseline for data path validation."""

from pathlib import Path
import json
import shutil

from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from moremouse.data.video_frames import encode_video, extract_frame


def main() -> None:
    """Run a minimal GPU training smoke test and save visual outputs."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load("configs/datasets/markerless_mammal.yaml").dataset
    root = Path(cfg.root)
    output_root = Path(cfg.outputs.tiny_mask_baseline_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "frames").mkdir(parents=True)
    (output_root / "figures").mkdir()

    frame_ids = [0, 20, 40, 60, 80, 100]
    views = [0, 1, 2, 3, 4, 5]
    inputs, targets = _load_samples(
        root,
        frame_ids,
        views,
        output_root / "frames",
        int(cfg.visualization.mask_binary_threshold),
    )
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=6, shuffle=True)

    model = _TinyMaskNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(40):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        losses.append(epoch_loss / len(loader))

    with torch.no_grad():
        logits = model(inputs.to(device)).cpu()
        predictions = torch.sigmoid(logits)
    grid_path = output_root / "figures" / "tiny_mask_predictions.png"
    _save_prediction_grid(inputs, targets, predictions, grid_path)
    video_path = output_root / "figures" / "tiny_mask_predictions.mp4"
    encode_video(str(output_root / "figures" / "tiny_mask_predictions.png"), video_path, fps=1)
    report = {
        "device": str(device),
        "torch": torch.__version__,
        "frame_ids": frame_ids,
        "views": views,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "grid": str(grid_path),
        "video": str(video_path),
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Tiny baseline report: {}", report)


class _TinyMaskNet:
    """Lazy wrapper so tests do not require torch imports."""

    def __new__(cls):  # noqa: D102
        """Build the tiny torch module at runtime."""
        import torch
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Upsample(size=(128, 128), mode="bilinear", align_corners=False),
        )


def _load_samples(
    root: Path,
    frame_ids: list[int],
    views: list[int],
    frame_dir: Path,
    mask_threshold: int,
):
    import numpy as np
    import torch

    images = []
    masks = []
    for frame_id in frame_ids:
        for view in views:
            rgb_path = frame_dir / f"rgb_v{view}_f{frame_id:06d}.png"
            mask_path = frame_dir / f"mask_v{view}_f{frame_id:06d}.png"
            extract_frame(root / "videos_undist" / f"{view}.mp4", frame_id, rgb_path)
            extract_frame(root / "simpleclick_undist" / f"{view}.mp4", frame_id, mask_path)
            rgb = Image.open(rgb_path).convert("RGB").resize((128, 128))
            mask = Image.open(mask_path).convert("L").resize((128, 128))
            images.append(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            masks.append(
                (np.asarray(mask, dtype=np.float32)[None, ...] > mask_threshold).astype(
                    np.float32
                )
            )
    return torch.tensor(np.stack(images)), torch.tensor(np.stack(masks))


def _save_prediction_grid(inputs, targets, predictions, output_path: Path) -> None:
    cells = []
    limit = min(8, inputs.shape[0])
    for index in range(limit):
        rgb = _tensor_rgb(inputs[index])
        target = _tensor_mask(targets[index], (40, 220, 120))
        pred = _tensor_mask(predictions[index], (255, 90, 70))
        cells.append(_triplet(rgb, target, pred))
    width = cells[0].width
    height = cells[0].height
    output = Image.new("RGB", (width * 2, height * 4), (20, 20, 20))
    for index, cell in enumerate(cells):
        output.paste(cell, ((index % 2) * width, (index // 2) * height))
    output.save(output_path)


def _tensor_rgb(tensor) -> Image.Image:
    import numpy as np

    array = (tensor.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB")


def _tensor_mask(tensor, color: tuple[int, int, int]) -> Image.Image:
    import numpy as np

    mask = (tensor.squeeze().numpy() > 0.5).astype(np.uint8) * 180
    return Image.fromarray(mask, "L").convert("RGB").point(lambda value: min(value, color[0]))


def _triplet(rgb: Image.Image, target: Image.Image, pred: Image.Image) -> Image.Image:
    output = Image.new("RGB", (rgb.width * 3, rgb.height + 20), (20, 20, 20))
    for index, (label, image) in enumerate([("rgb", rgb), ("target", target), ("pred", pred)]):
        output.paste(image, (index * rgb.width, 20))
        draw = ImageDraw.Draw(output)
        draw.text((index * rgb.width + 4, 4), label, fill=(255, 255, 255))
    return output


if __name__ == "__main__":
    main()
