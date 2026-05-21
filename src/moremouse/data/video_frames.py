"""Video frame extraction helpers."""

from pathlib import Path
import subprocess


def extract_frame(video_path: Path, frame_id: int, output_path: Path) -> Path:
    """Extract one zero-based frame with ffmpeg."""
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_id})",
        "-frames:v",
        "1",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg did not write frame: {output_path}")
    return output_path


def encode_video(frame_glob: str, output_path: Path, fps: int = 6) -> Path:
    """Encode a numbered PNG sequence to an mp4 video."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-pattern_type",
        "glob",
        "-i",
        frame_glob,
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg did not write video: {output_path}")
    return output_path
