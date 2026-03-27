from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _get_ffprobe_path() -> str:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path is None:
        raise RuntimeError(
            "ffprobe is required to inspect .mp4 clips. Install ffmpeg with your package manager."
        )
    return ffprobe_path


def _escape_lavfi_path(video_path: Path) -> str:
    return str(video_path).replace("\\", "\\\\").replace("'", "\\'").replace(",", "\\,")


def _parse_ffprobe_csv_float(line: str) -> float:
    """Parse ffprobe CSV output, tolerating trailing separators in single-value rows."""
    value = line.strip().split(",", 1)[0].strip()
    return float(value)


def get_video_metadata(video_path: str | Path) -> dict[str, float | int]:
    """Return basic ffprobe metadata for a video clip."""
    video_path = Path(video_path)
    ffprobe_path = _get_ffprobe_path()

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 2:
        raise RuntimeError(f"Unexpected ffprobe output for {video_path}: {result.stdout}")

    return {
        "duration_seconds": float(lines[0]),
        "size_bytes": int(float(lines[1])),
    }


def get_sampled_frame_timestamps(
    video_path: str | Path,
    sample_every_seconds: float,
) -> list[float]:
    """Return exact PTS timestamps for frames sampled by the configured fps filter."""
    video_path = Path(video_path)
    ffprobe_path = _get_ffprobe_path()
    lavfi_path = _escape_lavfi_path(video_path)
    filter_graph = f"movie='{lavfi_path}',fps=1/{sample_every_seconds}"

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        filter_graph,
        "-show_entries",
        "frame=pts_time",
        "-of",
        "csv=p=0",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return [
        round(_parse_ffprobe_csv_float(line), 6)
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_every_seconds: float,
) -> list[Path]:
    """Extract JPG frames from a video clip using ffmpeg."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to process .mp4 clips. Install it with your package manager."
        )

    frame_pattern = output_dir / "frame_%06d.jpg"
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{sample_every_seconds}",
        "-q:v",
        "2",
        str(frame_pattern),
    ]

    subprocess.run(command, check=True)
    return sorted(output_dir.glob("frame_*.jpg"))
