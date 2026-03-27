import shutil
from pathlib import Path


def list_mp4_files(root: str | Path) -> list[Path]:
    """Return all MP4 files under the given root directory."""
    return sorted(Path(root).rglob("*.mp4"))


def safe_report_name(file_path: str | Path, root: str | Path) -> str:
    """Create a stable JSON filename from a clip path relative to the mount root."""
    file_path = Path(file_path)
    root = Path(root)

    try:
        relative_path = file_path.relative_to(root)
    except ValueError:
        relative_path = file_path.name

    if isinstance(relative_path, Path):
        relative_text = str(relative_path)
    else:
        relative_text = relative_path

    return relative_text.replace("/", "__").replace(".mp4", ".json")


def copy_clip_preserving_relative_path(
    clip_path: str | Path,
    root: str | Path,
    destination_root: str | Path,
) -> Path:
    """Copy a clip to a destination root while preserving its path relative to root."""
    clip_path = Path(clip_path)
    root = Path(root)
    destination_root = Path(destination_root)

    try:
        relative_path = clip_path.relative_to(root)
    except ValueError:
        relative_path = Path(clip_path.name)

    destination_path = destination_root / relative_path
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if not destination_path.exists() or clip_path.stat().st_size != destination_path.stat().st_size:
        shutil.copy2(clip_path, destination_path)

    return destination_path
