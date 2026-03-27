import os
import subprocess
from pathlib import Path


def mount_exists(mount_path: str | Path) -> bool:
    """Return True when the configured mount path exists."""
    return Path(mount_path).exists()


def mount_is_ready(mount_path: str | Path) -> bool:
    """Return True when the path exists inside a non-root mounted filesystem."""
    path = Path(mount_path)
    if not path.is_dir():
        return False

    if os.path.ismount(path):
        return True

    result = subprocess.run(
        ["findmnt", "-T", str(path), "-o", "TARGET", "--noheadings"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False

    mount_target = result.stdout.strip()
    return bool(mount_target) and mount_target != "/"
