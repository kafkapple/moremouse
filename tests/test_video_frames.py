"""Tests for video frame command helpers."""

from pathlib import Path
from unittest.mock import patch

from moremouse.data.video_frames import encode_video, extract_frame


def test_extract_frame_invokes_ffmpeg(tmp_path: Path) -> None:
    """Verify frame extraction builds the expected ffmpeg command."""
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "frame.png"
    video_path.write_bytes(b"fake")

    def fake_run(command: list[str], check: bool) -> None:
        """Mock subprocess.run and create the expected output file."""
        assert check is True
        assert command[0] == "ffmpeg"
        assert "select=eq(n\\,7)" in command
        output_path.write_bytes(b"png")

    with patch("moremouse.data.video_frames.subprocess.run", side_effect=fake_run):
        assert extract_frame(video_path, 7, output_path) == output_path


def test_encode_video_invokes_ffmpeg(tmp_path: Path) -> None:
    """Verify video encoding builds the expected ffmpeg command."""
    output_path = tmp_path / "out.mp4"

    def fake_run(command: list[str], check: bool) -> None:
        """Mock subprocess.run and create the expected video file."""
        assert check is True
        assert "-pattern_type" in command
        output_path.write_bytes(b"mp4")

    with patch("moremouse.data.video_frames.subprocess.run", side_effect=fake_run):
        assert encode_video(str(tmp_path / "*.png"), output_path, fps=3) == output_path


def test_encode_video_accepts_single_image(tmp_path: Path) -> None:
    """Verify a single image can be encoded as a static video."""
    image_path = tmp_path / "frame.png"
    output_path = tmp_path / "out.mp4"
    image_path.write_bytes(b"png")

    def fake_run(command: list[str], check: bool) -> None:
        """Mock subprocess.run and create the expected static video."""
        assert check is True
        assert "-loop" in command
        output_path.write_bytes(b"mp4")

    with patch("moremouse.data.video_frames.subprocess.run", side_effect=fake_run):
        assert encode_video(str(image_path), output_path, fps=1) == output_path
