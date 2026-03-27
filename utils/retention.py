from __future__ import annotations

import time
from pathlib import Path


def prune_old_files(root: str | Path, max_age_days: int) -> list[Path]:
    """Delete files older than max_age_days under root and remove empty directories."""
    root = Path(root)
    if max_age_days <= 0 or not root.exists():
        return []

    cutoff = time.time() - (max_age_days * 86400)
    deleted: list[Path] = []

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink()
            deleted.append(path)

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    return deleted
