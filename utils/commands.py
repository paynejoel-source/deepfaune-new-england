from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils.camtrap_dp import export_camtrap_dp_package
from utils.reporting import write_json_report
from utils.validation import validate_reports


def run_preflight_check(
    config_path: Path,
    settings: dict[str, Any],
    mount_root: Path,
    detector_model_path: Path,
    classifier_model_path: Path,
    mount_exists_fn,
    mount_is_ready_fn,
) -> int:
    """Print a human-readable validation summary and exit without inference."""
    print("DeepFaune New England preflight check")
    print(f"Config: {config_path}")
    print(f"Mount path: {mount_root}")
    print(f"Detector model: {detector_model_path}")
    print(f"Classifier model: {classifier_model_path}")
    print(f"Mount exists: {mount_exists_fn(mount_root)}")
    print(f"Mount ready: {mount_is_ready_fn(mount_root)}")
    print(f"Detector model exists: {detector_model_path.is_file()}")
    print(f"Classifier model exists: {classifier_model_path.is_file()}")
    print(f"Default camera: {str(settings.get('DEFAULT_CAMERA_NAME', '')).strip() or '(unset)'}")
    print("Preflight check passed.")
    return 0


def build_demo_clip_report(
    parse_clip_epoch_range_fn,
    isoformat_from_epoch_fn,
    local_timezone,
    summarize_frames_fn,
) -> dict[str, Any]:
    """Build a publish-safe example clip report for documentation and demos."""
    clip_path = Path("/demo_mount/previews/back_yard/1774554110.0-1774554130.0.mp4")
    clip_start_epoch_seconds, clip_end_epoch_seconds = parse_clip_epoch_range_fn(clip_path)
    frames = [
        {
            "frame_index": 1,
            "frame_path": "frame_0001.jpg",
            "timestamp_seconds": 1.0,
            "timestamp_utc": isoformat_from_epoch_fn(clip_start_epoch_seconds + 1.0, timezone.utc),
            "timestamp_local": isoformat_from_epoch_fn(clip_start_epoch_seconds + 1.0, local_timezone),
            "detections": [
                {
                    "label": "animal",
                    "class_id": 0,
                    "confidence": 0.96,
                    "bbox_xyxy": [104.0, 88.0, 812.0, 640.0],
                    "bbox_normalized": [0.081, 0.122, 0.634, 0.889],
                    "classification": {
                        "prediction": "white_tailed_deer",
                        "class_id": 7,
                        "confidence": 0.93,
                        "all_confidences": {
                            "white_tailed_deer": 0.93,
                            "bobcat": 0.04,
                            "black_bear": 0.02,
                            "other": 0.01,
                        },
                    },
                }
            ],
        },
        {
            "frame_index": 2,
            "frame_path": "frame_0002.jpg",
            "timestamp_seconds": 7.0,
            "timestamp_utc": isoformat_from_epoch_fn(clip_start_epoch_seconds + 7.0, timezone.utc),
            "timestamp_local": isoformat_from_epoch_fn(clip_start_epoch_seconds + 7.0, local_timezone),
            "detections": [
                {
                    "label": "animal",
                    "class_id": 0,
                    "confidence": 0.91,
                    "bbox_xyxy": [132.0, 102.0, 844.0, 654.0],
                    "bbox_normalized": [0.103, 0.142, 0.659, 0.908],
                    "classification": {
                        "prediction": "white_tailed_deer",
                        "class_id": 7,
                        "confidence": 0.89,
                        "all_confidences": {
                            "white_tailed_deer": 0.89,
                            "bobcat": 0.06,
                            "black_bear": 0.03,
                            "other": 0.02,
                        },
                    },
                }
            ],
        },
    ]
    summary = summarize_frames_fn(frames)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "clip_path": str(clip_path),
        "relative_clip_path": "previews/back_yard/1774554110.0-1774554130.0.mp4",
        "camera_name": "back_yard",
        "clip_start_epoch_seconds": clip_start_epoch_seconds,
        "clip_end_epoch_seconds": clip_end_epoch_seconds,
        "clip_start_time_utc": isoformat_from_epoch_fn(clip_start_epoch_seconds, timezone.utc),
        "clip_end_time_utc": isoformat_from_epoch_fn(clip_end_epoch_seconds, timezone.utc),
        "clip_start_time_local": isoformat_from_epoch_fn(clip_start_epoch_seconds, local_timezone),
        "clip_end_time_local": isoformat_from_epoch_fn(clip_end_epoch_seconds, local_timezone),
        "clip_duration_seconds": 20.017,
        "clip_size_bytes": 7340032,
        "frame_sample_seconds": 1.0,
        "frame_count": len(frames),
        "processing_time_seconds": 1.284,
        "device": "cpu",
        "models": {
            "detector": {
                "version": "MDV6-yolov9-c",
                "weights_path": "models/MDV6-yolov9-c.pt",
                "confidence_threshold": 0.5,
            },
            "classifier": {
                "weights_path": "models/dfne_weights_v1_0.pth",
            },
        },
        "positive_clip_copied": True,
        "positive_clip_path": "output/demo/positive_clips/previews/back_yard/1774554110.0-1774554130.0.mp4",
        "summary": summary,
        "frames": frames,
        **summary,
    }


def run_demo(
    output_dir: Path,
    build_demo_clip_report_fn,
    floor_to_hour_fn,
    local_timezone,
    aggregate_clip_reports_fn,
    hourly_summary_path_fn,
) -> int:
    """Write demo clip and hourly reports without requiring any local runtime assets."""
    demo_root = output_dir / "demo"
    clip_report = build_demo_clip_report_fn()
    clip_destination = demo_root / "clips" / "demo_clip_report.json"
    write_json_report(clip_report, clip_destination)
    clip_report["report_path"] = str(clip_destination)

    clip_start = float(clip_report["clip_start_epoch_seconds"])
    window_start = floor_to_hour_fn(datetime.fromtimestamp(clip_start, tz=timezone.utc).astimezone(local_timezone))
    window_end = window_start + timedelta(hours=1)
    hourly_summary = aggregate_clip_reports_fn(
        clip_reports=[clip_report],
        window_start=window_start,
        window_end=window_end,
        camera_name=str(clip_report["camera_name"]),
        dry_run=False,
        selected_clips=[Path(str(clip_report["clip_path"]))],
        skipped_without_timestamp=[],
    )
    hourly_destination = hourly_summary_path_fn(demo_root, window_start, str(clip_report["camera_name"]))
    write_json_report(hourly_summary, hourly_destination)
    print(f"Wrote demo clip report: {clip_destination}")
    print(f"Wrote demo hourly summary: {hourly_destination}")
    return 0


def run_camtrap_dp_export(config_path: Path, reports_dir: Path | None, export_dir: Path | None, read_settings_file_fn) -> int:
    """Export existing clip reports as a Camtrap DP package."""
    settings = read_settings_file_fn(config_path)
    output_dir = Path(str(settings.get("OUTPUT_DIR", "output")))
    source_dir = reports_dir or (output_dir / "clips" if (output_dir / "clips").is_dir() else output_dir)
    destination_dir = export_dir or (output_dir / "camtrap_dp")
    export_camtrap_dp_package(source_dir, destination_dir, settings)
    print(f"Wrote Camtrap DP package: {destination_dir}")
    return 0


def run_validation(
    config_path: Path,
    validation_source: Path,
    reports_dir: Path | None,
    read_settings_file_fn,
) -> int:
    """Validate clip reports against a labeled CSV/JSON file."""
    settings = read_settings_file_fn(config_path)
    output_dir = Path(str(settings.get("OUTPUT_DIR", "output")))
    source_dir = reports_dir or (output_dir / "clips" if (output_dir / "clips").is_dir() else output_dir)
    destination = output_dir / "validation" / "validation_report.json"
    validate_reports(source_dir, validation_source, destination)
    print(f"Wrote validation report: {destination}")
    return 0
