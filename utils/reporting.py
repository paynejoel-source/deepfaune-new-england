from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json_report(source: str | Path) -> dict[str, Any]:
    """Read a JSON report from disk."""
    source = Path(source)
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_report(report: dict[str, Any], destination: str | Path) -> Path:
    """Write a JSON report to disk."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return destination
