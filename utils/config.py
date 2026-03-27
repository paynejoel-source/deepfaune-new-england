from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "BEAST_MOUNT_PATH",
    "DETECTOR_MODEL_PATH",
    "CLASSIFIER_MODEL_PATH",
    "CONFIDENCE_THRESHOLD",
}


def load_pipeline_settings(config_path: str | Path) -> dict:
    """Load YAML configuration and verify the required keys are present."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}

    missing = sorted(REQUIRED_KEYS - settings.keys())
    if missing:
        missing_keys = ", ".join(missing)
        raise KeyError(f"Missing required config keys: {missing_keys}")

    return settings
