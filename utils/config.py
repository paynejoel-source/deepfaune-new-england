from __future__ import annotations

import shutil
from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "BEAST_MOUNT_PATH",
    "DETECTOR_MODEL_PATH",
    "CLASSIFIER_MODEL_PATH",
    "CONFIDENCE_THRESHOLD",
}


def validate_pipeline_settings(settings: dict, config_path: str | Path) -> dict:
    """Validate config values and return normalized path fields."""
    config_path = Path(config_path)
    config_dir = config_path.parent

    mount_root = Path(settings["BEAST_MOUNT_PATH"])
    detector_model_path = Path(settings["DETECTOR_MODEL_PATH"])
    classifier_model_path = Path(settings["CLASSIFIER_MODEL_PATH"])

    if not detector_model_path.is_absolute():
        detector_model_path = config_dir.parent / detector_model_path
    if not classifier_model_path.is_absolute():
        classifier_model_path = config_dir.parent / classifier_model_path

    errors: list[str] = []

    if not config_path.is_file():
        errors.append(f"Config file not found: {config_path}")

    if not mount_root.exists():
        errors.append(
            "BEAST_MOUNT_PATH does not exist: "
            f"{mount_root}. Update config/pipeline_settings.yaml for your local clip tree."
        )

    if not detector_model_path.is_file():
        errors.append(
            "Detector model not found: "
            f"{detector_model_path}. See docs/MODELS.md for expected model setup."
        )

    if not classifier_model_path.is_file():
        errors.append(
            "Classifier model not found: "
            f"{classifier_model_path}. See docs/MODELS.md for expected model setup."
        )

    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            errors.append(
                f"Required system binary not found on PATH: {binary}. Install ffmpeg first."
            )

    if errors:
        raise ValueError("\n".join(errors))

    return {
        "mount_root": mount_root,
        "detector_model_path": detector_model_path,
        "classifier_model_path": classifier_model_path,
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

    validated_paths = validate_pipeline_settings(settings, config_path)
    settings.update(validated_paths)

    return settings
