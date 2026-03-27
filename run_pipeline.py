#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from time import perf_counter
from pathlib import Path
import re
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_CACHE_DIR = Path(".cache").resolve()
os.makedirs(PROJECT_CACHE_DIR / "torch", exist_ok=True)
os.makedirs(PROJECT_CACHE_DIR / "matplotlib", exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(PROJECT_CACHE_DIR / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_CACHE_DIR / "matplotlib"))

from utils.camtrap_dp import export_camtrap_dp_package
from utils.file_handling import (
    copy_clip_preserving_relative_path,
    list_mp4_files,
    safe_report_name,
)
from utils.locking import file_lock
from utils.mount_checks import mount_exists, mount_is_ready
from utils.retention import prune_old_files
from utils.reporting import read_json_report, write_json_report
from utils.review import write_review_bundle
from utils.validation import validate_reports
from utils.video import extract_frames, get_sampled_frame_timestamps, get_video_metadata

LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo or ZoneInfo("UTC")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepFaune New England clip pipeline.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Write publish-safe example reports without requiring models, clips, or a mounted source tree.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_settings.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help="Optional path to a single clip to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of clips to process.",
    )
    parser.add_argument(
        "--frame-sample-seconds",
        type=float,
        default=None,
        help="Override the frame extraction interval in seconds for this run.",
    )
    parser.add_argument(
        "--hourly",
        action="store_true",
        help="Process hourly windows instead of scanning the entire tree.",
    )
    parser.add_argument(
        "--window-start",
        default=None,
        help="Explicit window start in ISO8601 format.",
    )
    parser.add_argument(
        "--window-end",
        default=None,
        help="Explicit window end in ISO8601 format.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Optional camera name filter, such as back_yard.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which clips would be processed and write hourly summaries without running inference.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate config, model paths, mount path, and required system binaries without running inference.",
    )
    parser.add_argument(
        "--export-camtrap-dp",
        action="store_true",
        help="Export existing clip reports as a Camtrap DP package without changing the native JSON reports.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Optional directory containing clip report JSON files to export.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional destination directory for Camtrap DP export files.",
    )
    parser.add_argument(
        "--validate",
        default=None,
        help="Path to a CSV or JSON file with validation cases (relative_clip_path, expected_label).",
    )
    return parser


def select_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_clip_epoch_range(clip_path: Path) -> tuple[float | None, float | None]:
    """Parse Frigate-style clip filenames of the form <start>-<end>.mp4."""
    match = re.match(r"(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)\.mp4$", clip_path.name)
    if match is None:
        return None, None
    return float(match.group("start")), float(match.group("end"))


def isoformat_from_epoch(epoch_seconds: float, tzinfo: Any) -> str:
    """Convert epoch seconds to an ISO8601 timestamp in the requested timezone."""
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).astimezone(tzinfo).isoformat()


def parse_datetime_arg(value: str) -> datetime:
    """Parse an ISO8601 datetime, assuming local timezone when omitted."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed


def floor_to_hour(value: datetime) -> datetime:
    """Round a timezone-aware datetime down to the start of its hour."""
    return value.replace(minute=0, second=0, microsecond=0)


def iterate_hour_windows(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Split a range into contiguous one-hour windows."""
    windows: list[tuple[datetime, datetime]] = []
    current = start
    while current < end:
        next_hour = current + timedelta(hours=1)
        windows.append((current, next_hour))
        current = next_hour
    return windows


def sanitize_name(value: str) -> str:
    """Return a filesystem-safe identifier fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "all"


def get_clip_camera_name(clip_path: Path) -> str:
    """Return the inferred camera name from the clip path."""
    return clip_path.parent.name


def select_clips_for_window(
    clip_paths: list[Path],
    window_start: datetime,
    window_end: datetime,
    camera_name: str | None,
) -> tuple[list[Path], list[Path]]:
    """Select clips whose parsed start timestamp falls inside the requested window."""
    window_start_epoch = window_start.astimezone(timezone.utc).timestamp()
    window_end_epoch = window_end.astimezone(timezone.utc).timestamp()
    selected: list[Path] = []
    skipped_without_timestamp: list[Path] = []

    for clip_path in clip_paths:
        if camera_name and get_clip_camera_name(clip_path) != camera_name:
            continue

        clip_start_epoch, _ = parse_clip_epoch_range(clip_path)
        if clip_start_epoch is None:
            skipped_without_timestamp.append(clip_path)
            continue

        if window_start_epoch <= clip_start_epoch < window_end_epoch:
            selected.append(clip_path)

    return selected, skipped_without_timestamp


def state_file_path(state_dir: Path, camera_name: str | None) -> Path:
    """Return the state file path for a camera-specific hourly scheduler."""
    suffix = sanitize_name(camera_name or "all")
    return state_dir / f"hourly_state_{suffix}.json"


def load_state(state_path: Path) -> dict[str, Any] | None:
    """Load scheduler state from disk when present."""
    if not state_path.is_file():
        return None
    return read_json_report(state_path)


def run_preflight_check(
    config_path: Path,
    settings: dict[str, Any],
    mount_root: Path,
    detector_model_path: Path,
    classifier_model_path: Path,
) -> int:
    """Print a human-readable validation summary and exit without inference."""
    print("DeepFaune New England preflight check")
    print(f"Config: {config_path}")
    print(f"Mount path: {mount_root}")
    print(f"Detector model: {detector_model_path}")
    print(f"Classifier model: {classifier_model_path}")
    print(f"Mount exists: {mount_exists(mount_root)}")
    print(f"Mount ready: {mount_is_ready(mount_root)}")
    print(f"Detector model exists: {detector_model_path.is_file()}")
    print(f"Classifier model exists: {classifier_model_path.is_file()}")
    print(f"Default camera: {str(settings.get('DEFAULT_CAMERA_NAME', '')).strip() or '(unset)'}")
    print("Preflight check passed.")
    return 0


def build_demo_clip_report() -> dict[str, Any]:
    """Build a publish-safe example clip report for documentation and demos."""
    clip_path = Path("/demo_mount/previews/back_yard/1774554110.0-1774554130.0.mp4")
    clip_start_epoch_seconds, clip_end_epoch_seconds = parse_clip_epoch_range(clip_path)
    frames = [
        {
            "frame_index": 1,
            "frame_path": "frame_0001.jpg",
            "timestamp_seconds": 1.0,
            "timestamp_utc": isoformat_from_epoch(clip_start_epoch_seconds + 1.0, timezone.utc),
            "timestamp_local": isoformat_from_epoch(clip_start_epoch_seconds + 1.0, LOCAL_TIMEZONE),
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
            "timestamp_utc": isoformat_from_epoch(clip_start_epoch_seconds + 7.0, timezone.utc),
            "timestamp_local": isoformat_from_epoch(clip_start_epoch_seconds + 7.0, LOCAL_TIMEZONE),
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
    summary = summarize_frames(frames)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "clip_path": str(clip_path),
        "relative_clip_path": "previews/back_yard/1774554110.0-1774554130.0.mp4",
        "camera_name": "back_yard",
        "clip_start_epoch_seconds": clip_start_epoch_seconds,
        "clip_end_epoch_seconds": clip_end_epoch_seconds,
        "clip_start_time_utc": isoformat_from_epoch(clip_start_epoch_seconds, timezone.utc),
        "clip_end_time_utc": isoformat_from_epoch(clip_end_epoch_seconds, timezone.utc),
        "clip_start_time_local": isoformat_from_epoch(clip_start_epoch_seconds, LOCAL_TIMEZONE),
        "clip_end_time_local": isoformat_from_epoch(clip_end_epoch_seconds, LOCAL_TIMEZONE),
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


def run_demo(output_dir: Path) -> int:
    """Write demo clip and hourly reports without requiring any local runtime assets."""
    demo_root = output_dir / "demo"
    clip_report = build_demo_clip_report()
    clip_destination = demo_root / "clips" / "demo_clip_report.json"
    write_json_report(clip_report, clip_destination)
    clip_report["report_path"] = str(clip_destination)

    clip_start = float(clip_report["clip_start_epoch_seconds"])
    window_start = floor_to_hour(datetime.fromtimestamp(clip_start, tz=timezone.utc).astimezone(LOCAL_TIMEZONE))
    window_end = window_start + timedelta(hours=1)
    hourly_summary = aggregate_clip_reports(
        clip_reports=[clip_report],
        window_start=window_start,
        window_end=window_end,
        camera_name=str(clip_report["camera_name"]),
        dry_run=False,
        selected_clips=[Path(str(clip_report["clip_path"]))],
        skipped_without_timestamp=[],
    )
    hourly_destination = hourly_summary_path(demo_root, window_start, str(clip_report["camera_name"]))
    write_json_report(hourly_summary, hourly_destination)
    print(f"Wrote demo clip report: {clip_destination}")
    print(f"Wrote demo hourly summary: {hourly_destination}")
    return 0


def run_camtrap_dp_export(config_path: Path, reports_dir: Path | None, export_dir: Path | None) -> int:
    """Export existing clip reports as a Camtrap DP package."""
    from utils.config import read_settings_file

    settings = read_settings_file(config_path)
    output_dir = Path(str(settings.get("OUTPUT_DIR", "output")))
    source_dir = reports_dir or (output_dir / "clips" if (output_dir / "clips").is_dir() else output_dir)
    destination_dir = export_dir or (output_dir / "camtrap_dp")
    export_camtrap_dp_package(source_dir, destination_dir, settings)
    print(f"Wrote Camtrap DP package: {destination_dir}")
    return 0


def run_validation(config_path: Path, validation_source: Path, reports_dir: Path | None) -> int:
    """Validate clip reports against a labeled CSV/JSON file."""
    from utils.config import read_settings_file

    settings = read_settings_file(config_path)
    output_dir = Path(str(settings.get("OUTPUT_DIR", "output")))
    source_dir = reports_dir or (output_dir / "clips" if (output_dir / "clips").is_dir() else output_dir)
    destination = output_dir / "validation" / "validation_report.json"
    validate_reports(source_dir, validation_source, destination)
    print(f"Wrote validation report: {destination}")
    return 0


def save_state(state_path: Path, payload: dict[str, Any]) -> Path:
    """Persist scheduler state to disk."""
    return write_json_report(payload, state_path)


def hourly_summary_path(output_dir: Path, window_start: datetime, camera_name: str | None) -> Path:
    """Build the hourly summary destination path."""
    date_dir = window_start.strftime("%Y-%m-%d")
    hour_label = window_start.strftime("%H00")
    camera_label = sanitize_name(camera_name or "all")
    return output_dir / "hourly" / date_dir / f"{hour_label}_{camera_label}.json"


def hourly_review_dir(output_dir: Path, window_start: datetime, camera_name: str | None) -> Path:
    """Build the hourly review bundle directory path."""
    date_dir = window_start.strftime("%Y-%m-%d")
    hour_label = window_start.strftime("%H00")
    camera_label = sanitize_name(camera_name or "all")
    return output_dir / "review" / date_dir / f"{hour_label}_{camera_label}"


def run_retention(
    output_dir: Path,
    positive_clips_dir: Path | None,
    retention_days: int,
) -> dict[str, int]:
    """Prune old generated outputs while leaving models and state intact."""
    targets = {
        "clip_reports_deleted": output_dir / "clips",
        "hourly_reports_deleted": output_dir / "hourly",
    }
    if positive_clips_dir is not None:
        targets["positive_clips_deleted"] = positive_clips_dir

    results: dict[str, int] = {}
    for key, path in targets.items():
        results[key] = len(prune_old_files(path, retention_days))
    return results


def attach_classifications(
    detector: Any,
    detection_results: list[dict[str, Any]],
    classifications: list[Any],
    frame_timestamps: list[float],
    clip_start_epoch_seconds: float | None,
) -> list[dict[str, Any]]:
    classification_index = 0
    merged_frames: list[dict[str, Any]] = []

    for frame_index, frame in enumerate(detection_results, start=1):
        normalized_frame = detector.normalize_result(frame)
        merged_detections: list[dict[str, Any]] = []
        for detection in normalized_frame["detections"]:
            item = {
                "label": detection.label,
                "class_id": detection.class_id,
                "confidence": detection.confidence,
                "bbox_xyxy": detection.bbox_xyxy,
                "bbox_normalized": detection.bbox_normalized,
                "classification": None,
            }

            if detection.class_id == 0 and classification_index < len(classifications):
                classification = classifications[classification_index]
                classification_index += 1
                item["classification"] = {
                    "prediction": classification.prediction,
                    "class_id": classification.class_id,
                    "confidence": classification.confidence,
                    "all_confidences": classification.all_confidences,
                }

            merged_detections.append(item)

        absolute_timestamp_utc = None
        absolute_timestamp_local = None
        if clip_start_epoch_seconds is not None:
            absolute_epoch = clip_start_epoch_seconds + frame_timestamps[frame_index - 1]
            absolute_timestamp_utc = isoformat_from_epoch(absolute_epoch, timezone.utc)
            absolute_timestamp_local = isoformat_from_epoch(absolute_epoch, LOCAL_TIMEZONE)

        merged_frames.append(
            {
                "frame_index": frame_index,
                "frame_path": Path(normalized_frame["img_id"]).name,
                "timestamp_seconds": frame_timestamps[frame_index - 1],
                "timestamp_utc": absolute_timestamp_utc,
                "timestamp_local": absolute_timestamp_local,
                "detections": merged_detections,
            }
        )

    return merged_frames


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    species_counter: Counter[str] = Counter()
    detector_class_counter: Counter[str] = Counter()
    species_max_confidence: dict[str, float] = {}
    max_confidence = 0.0
    best_prediction: str | None = None
    animal_detection_count = 0
    human_present = False
    vehicle_present = False
    had_detections = False
    top_frames: list[dict[str, Any]] = []

    for frame in frames:
        for detection in frame["detections"]:
            had_detections = True
            detector_class_counter[detection["label"]] += 1
            if detection["label"] == "person":
                human_present = True
            if detection["label"] == "vehicle":
                vehicle_present = True

            classification = detection["classification"]
            if classification is None:
                continue

            animal_detection_count += 1
            prediction = classification["prediction"]
            confidence = classification["confidence"]
            species_counter[prediction] += 1
            species_max_confidence[prediction] = max(
                confidence,
                species_max_confidence.get(prediction, 0.0),
            )

            if confidence > max_confidence:
                max_confidence = confidence
                best_prediction = prediction

            top_frames.append(
                {
                    "frame_index": frame["frame_index"],
                    "frame_path": frame["frame_path"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "timestamp_utc": frame["timestamp_utc"],
                    "timestamp_local": frame["timestamp_local"],
                    "prediction": prediction,
                    "confidence": confidence,
                }
            )

    top_frames.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "had_detections": had_detections,
        "animal_detection_count": animal_detection_count,
        "detector_class_counts": dict(detector_class_counter),
        "species_counts": dict(species_counter),
        "species_confidence_summary": {
            key: round(value, 6) for key, value in sorted(species_max_confidence.items())
        },
        "top_prediction": best_prediction,
        "top_prediction_confidence": max_confidence,
        "human_present": human_present,
        "vehicle_present": vehicle_present,
        "top_frames": top_frames[:3],
    }


def aggregate_clip_reports(
    clip_reports: list[dict[str, Any]],
    window_start: datetime,
    window_end: datetime,
    camera_name: str | None,
    dry_run: bool,
    selected_clips: list[Path],
    skipped_without_timestamp: list[Path],
) -> dict[str, Any]:
    """Aggregate per-clip results into one hourly summary report."""
    species_counts: Counter[str] = Counter()
    detector_class_counts: Counter[str] = Counter()
    clip_status_counts: Counter[str] = Counter()
    total_animal_detections = 0
    clips_with_detections = 0
    human_present = False
    vehicle_present = False
    top_frames: list[dict[str, Any]] = []
    copied_positive_clips: list[str] = []
    clip_errors: list[dict[str, Any]] = []

    for report in clip_reports:
        clip_status_counts[str(report.get("status", "unknown"))] += 1
        if report.get("had_detections"):
            clips_with_detections += 1
        total_animal_detections += int(report.get("animal_detection_count", 0))
        species_counts.update(report.get("species_counts", {}))
        detector_class_counts.update(report.get("detector_class_counts", {}))
        human_present = human_present or bool(report.get("human_present"))
        vehicle_present = vehicle_present or bool(report.get("vehicle_present"))
        if report.get("positive_clip_copied") and report.get("positive_clip_path"):
            copied_positive_clips.append(str(report["positive_clip_path"]))
        if report.get("status") == "error":
            clip_errors.append(
                {
                    "clip_path": report["clip_path"],
                    "relative_clip_path": report["relative_clip_path"],
                    "error_type": report.get("error_type"),
                    "error_message": report.get("error_message"),
                }
            )

        for top_frame in report.get("top_frames", []):
            top_frames.append(
                {
                    "clip_path": report["clip_path"],
                    "relative_clip_path": report["relative_clip_path"],
                    **top_frame,
                }
            )

    top_frames.sort(key=lambda item: item["confidence"], reverse=True)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if dry_run else "success",
        "mode": "hourly",
        "dry_run": dry_run,
        "camera_name": camera_name,
        "window_start_utc": window_start.astimezone(timezone.utc).isoformat(),
        "window_end_utc": window_end.astimezone(timezone.utc).isoformat(),
        "window_start_local": window_start.astimezone(LOCAL_TIMEZONE).isoformat(),
        "window_end_local": window_end.astimezone(LOCAL_TIMEZONE).isoformat(),
        "window_start_epoch_seconds": window_start.astimezone(timezone.utc).timestamp(),
        "window_end_epoch_seconds": window_end.astimezone(timezone.utc).timestamp(),
        "clip_count_selected": len(selected_clips),
        "clip_count_processed": len(clip_reports),
        "clip_count_with_detections": clips_with_detections,
        "clip_status_counts": dict(clip_status_counts),
        "total_animal_detections": total_animal_detections,
        "species_counts": dict(species_counts),
        "detector_class_counts": dict(detector_class_counts),
        "human_present": human_present,
        "vehicle_present": vehicle_present,
        "copied_positive_clips": copied_positive_clips,
        "clip_errors": clip_errors,
        "top_frames": top_frames[:10],
        "selected_clips": [
            {
                "clip_path": str(path),
                "camera_name": get_clip_camera_name(path),
                "clip_start_epoch_seconds": parse_clip_epoch_range(path)[0],
            }
            for path in selected_clips
        ],
        "skipped_without_timestamp": [str(path) for path in skipped_without_timestamp],
        "processed_reports": [
            {
                "clip_path": report["clip_path"],
                "relative_clip_path": report["relative_clip_path"],
                "status": report["status"],
                "report_path": report.get("report_path"),
            }
            for report in clip_reports
        ],
    }


def process_clip(
    clip_path: Path,
    mount_root: Path,
    detector: Any,
    classifier: Any,
    frame_sample_seconds: float,
    confidence_threshold: float,
    detector_version: str,
    classifier_model_path: Path,
    device: str,
) -> dict[str, Any]:
    started_at = perf_counter()
    video_metadata = get_video_metadata(clip_path)
    try:
        relative_clip_path = clip_path.relative_to(mount_root)
    except ValueError:
        relative_clip_path = clip_path.name
    clip_start_epoch_seconds, clip_end_epoch_seconds = parse_clip_epoch_range(clip_path)

    with tempfile.TemporaryDirectory(prefix="dfne_frames_") as temp_dir:
        frame_dir = Path(temp_dir)
        frame_paths = extract_frames(
            video_path=clip_path,
            output_dir=frame_dir,
            sample_every_seconds=frame_sample_seconds,
        )
        frame_timestamps = get_sampled_frame_timestamps(
            video_path=clip_path,
            sample_every_seconds=frame_sample_seconds,
        )
        if len(frame_timestamps) != len(frame_paths):
            raise RuntimeError(
                "Frame timestamp count does not match extracted frame count: "
                f"{len(frame_timestamps)} timestamps vs {len(frame_paths)} frames"
            )

        detection_results = detector.detect_directory(frame_dir)
        classifications = classifier.classify_detections(detection_results)
        merged_frames = attach_classifications(
            detector,
            detection_results,
            classifications,
            frame_timestamps,
            clip_start_epoch_seconds,
        )
        summary = summarize_frames(merged_frames)

    processing_time_seconds = round(perf_counter() - started_at, 3)
    status = "success" if summary["had_detections"] else "no_detections"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "clip_path": str(clip_path),
        "relative_clip_path": str(relative_clip_path),
        "camera_name": clip_path.parent.name,
        "clip_start_epoch_seconds": clip_start_epoch_seconds,
        "clip_end_epoch_seconds": clip_end_epoch_seconds,
        "clip_start_time_utc": (
            isoformat_from_epoch(clip_start_epoch_seconds, timezone.utc)
            if clip_start_epoch_seconds is not None
            else None
        ),
        "clip_end_time_utc": (
            isoformat_from_epoch(clip_end_epoch_seconds, timezone.utc)
            if clip_end_epoch_seconds is not None
            else None
        ),
        "clip_start_time_local": (
            isoformat_from_epoch(clip_start_epoch_seconds, LOCAL_TIMEZONE)
            if clip_start_epoch_seconds is not None
            else None
        ),
        "clip_end_time_local": (
            isoformat_from_epoch(clip_end_epoch_seconds, LOCAL_TIMEZONE)
            if clip_end_epoch_seconds is not None
            else None
        ),
        "clip_duration_seconds": round(float(video_metadata["duration_seconds"]), 3),
        "clip_size_bytes": int(video_metadata["size_bytes"]),
        "frame_sample_seconds": frame_sample_seconds,
        "frame_count": len(frame_paths),
        "processing_time_seconds": processing_time_seconds,
        "device": device,
        "models": {
            "detector": {
                "version": detector_version,
                "weights_path": str(detector.model_path),
                "confidence_threshold": confidence_threshold,
            },
            "classifier": {
                "weights_path": str(classifier_model_path),
            },
        },
        "positive_clip_copied": False,
        "positive_clip_path": None,
        "summary": summary,
        "frames": merged_frames,
        **summary,
    }


def build_error_report(
    clip_path: Path,
    mount_root: Path,
    error: Exception,
    detector_model_path: Path,
    detector_version: str,
    classifier_model_path: Path,
    confidence_threshold: float,
    device: str,
) -> dict[str, Any]:
    """Build a structured error report for a clip that failed to process."""
    clip_start_epoch_seconds, clip_end_epoch_seconds = parse_clip_epoch_range(clip_path)
    try:
        relative_clip_path = str(clip_path.relative_to(mount_root))
    except ValueError:
        relative_clip_path = str(clip_path)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "clip_path": str(clip_path),
        "relative_clip_path": relative_clip_path,
        "camera_name": get_clip_camera_name(clip_path),
        "clip_start_epoch_seconds": clip_start_epoch_seconds,
        "clip_end_epoch_seconds": clip_end_epoch_seconds,
        "clip_start_time_utc": (
            isoformat_from_epoch(clip_start_epoch_seconds, timezone.utc)
            if clip_start_epoch_seconds is not None
            else None
        ),
        "clip_end_time_utc": (
            isoformat_from_epoch(clip_end_epoch_seconds, timezone.utc)
            if clip_end_epoch_seconds is not None
            else None
        ),
        "clip_start_time_local": (
            isoformat_from_epoch(clip_start_epoch_seconds, LOCAL_TIMEZONE)
            if clip_start_epoch_seconds is not None
            else None
        ),
        "clip_end_time_local": (
            isoformat_from_epoch(clip_end_epoch_seconds, LOCAL_TIMEZONE)
            if clip_end_epoch_seconds is not None
            else None
        ),
        "clip_duration_seconds": None,
        "clip_size_bytes": None,
        "frame_sample_seconds": None,
        "frame_count": 0,
        "processing_time_seconds": None,
        "device": device,
        "models": {
            "detector": {
                "version": detector_version,
                "weights_path": str(detector_model_path),
                "confidence_threshold": confidence_threshold,
            },
            "classifier": {
                "weights_path": str(classifier_model_path),
            },
        },
        "positive_clip_copied": False,
        "positive_clip_path": None,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "summary": {
            "had_detections": False,
            "animal_detection_count": 0,
            "detector_class_counts": {},
            "species_counts": {},
            "species_confidence_summary": {},
            "top_prediction": None,
            "top_prediction_confidence": 0.0,
            "human_present": False,
            "vehicle_present": False,
            "top_frames": [],
        },
        "frames": [],
        "had_detections": False,
        "animal_detection_count": 0,
        "detector_class_counts": {},
        "species_counts": {},
        "species_confidence_summary": {},
        "top_prediction": None,
        "top_prediction_confidence": 0.0,
        "human_present": False,
        "vehicle_present": False,
        "top_frames": [],
    }


def maybe_copy_positive_clip(
    report: dict[str, Any],
    clip_path: Path,
    mount_root: Path,
    positive_clips_dir: Path | None,
    copy_positive_clips: bool,
    positive_clip_min_confidence: float,
) -> dict[str, Any]:
    """Copy animal-positive clips into a designated folder and annotate the report."""
    if not copy_positive_clips or positive_clips_dir is None:
        return report

    if int(report.get("animal_detection_count", 0)) <= 0:
        return report

    if float(report.get("top_prediction_confidence", 0.0)) < positive_clip_min_confidence:
        return report

    destination_path = copy_clip_preserving_relative_path(
        clip_path=clip_path,
        root=mount_root,
        destination_root=positive_clips_dir,
    )
    report["positive_clip_copied"] = True
    report["positive_clip_path"] = str(destination_path)
    return report


def process_hourly_window(
    window_start: datetime,
    window_end: datetime,
    clip_paths: list[Path],
    mount_root: Path,
    output_dir: Path,
    detector: Any | None,
    classifier: Any | None,
    frame_sample_seconds: float,
    confidence_threshold: float,
    detector_version: str,
    classifier_model_path: Path,
    device: str,
    camera_name: str | None,
    dry_run: bool,
    copy_positive_clips: bool,
    positive_clips_dir: Path | None,
    positive_clip_min_confidence: float,
) -> Path:
    """Process or preview one hourly window and write the hourly summary JSON."""
    selected_clips, skipped_without_timestamp = select_clips_for_window(
        clip_paths,
        window_start,
        window_end,
        camera_name,
    )

    clip_reports: list[dict[str, Any]] = []
    if not dry_run and selected_clips:
        if detector is None or classifier is None:
            raise RuntimeError("Detector and classifier must be initialized for non-dry-run hourly mode.")

        for clip_path in selected_clips:
            try:
                report = process_clip(
                    clip_path=clip_path,
                    mount_root=mount_root,
                    detector=detector,
                    classifier=classifier,
                    frame_sample_seconds=frame_sample_seconds,
                    confidence_threshold=confidence_threshold,
                    detector_version=detector_version,
                    classifier_model_path=classifier_model_path,
                    device=device,
                )
                report = maybe_copy_positive_clip(
                    report=report,
                    clip_path=clip_path,
                    mount_root=mount_root,
                    positive_clips_dir=positive_clips_dir,
                    copy_positive_clips=copy_positive_clips,
                    positive_clip_min_confidence=positive_clip_min_confidence,
                )
            except Exception as error:
                report = build_error_report(
                    clip_path=clip_path,
                    mount_root=mount_root,
                    error=error,
                    detector_model_path=Path(detector.model_path),
                    detector_version=detector_version,
                    classifier_model_path=classifier_model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                )
            report_name = safe_report_name(clip_path, mount_root)
            clip_destination = output_dir / "clips" / report_name
            write_json_report(report, clip_destination)
            report["report_path"] = str(clip_destination)
            clip_reports.append(report)

    summary = aggregate_clip_reports(
        clip_reports=clip_reports,
        window_start=window_start,
        window_end=window_end,
        camera_name=camera_name,
        dry_run=dry_run,
        selected_clips=selected_clips,
        skipped_without_timestamp=skipped_without_timestamp,
    )
    destination = hourly_summary_path(output_dir, window_start, camera_name)
    write_json_report(summary, destination)
    write_review_bundle(summary, hourly_review_dir(output_dir, window_start, camera_name))
    return destination


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.demo:
        return run_demo(Path("output"))

    if args.export_camtrap_dp:
        return run_camtrap_dp_export(
            config_path=Path(args.config),
            reports_dir=Path(args.reports_dir) if args.reports_dir else None,
            export_dir=Path(args.export_dir) if args.export_dir else None,
        )

    if args.validate:
        return run_validation(
            config_path=Path(args.config),
            validation_source=Path(args.validate),
            reports_dir=Path(args.reports_dir) if args.reports_dir else None,
        )

    from utils.config import load_pipeline_settings

    config_path = Path(args.config)
    settings = load_pipeline_settings(config_path)
    mount_root = Path(settings["mount_root"])
    detector_model_path = Path(settings["detector_model_path"])
    detector_version = str(settings.get("DETECTOR_VERSION", "MDV6-yolov9-c"))
    classifier_model_path = Path(settings["classifier_model_path"])
    confidence_threshold = float(settings["CONFIDENCE_THRESHOLD"])
    frame_sample_seconds = float(settings.get("FRAME_SAMPLE_SECONDS", 1.0))
    if args.frame_sample_seconds is not None:
        frame_sample_seconds = args.frame_sample_seconds
    output_dir = Path(settings.get("OUTPUT_DIR", "output"))
    state_dir = Path(settings.get("STATE_DIR", output_dir / "state"))
    lock_file = Path(settings.get("LOCK_FILE", state_dir / "pipeline.lock"))
    retention_days = int(settings.get("RETENTION_DAYS", 0))
    default_camera_name = str(settings.get("DEFAULT_CAMERA_NAME", "")).strip() or None
    camera_name = (args.camera or default_camera_name or None)
    copy_positive_clips = bool(settings.get("COPY_POSITIVE_CLIPS", False))
    positive_clips_dir_value = str(settings.get("POSITIVE_CLIPS_DIR", "")).strip()
    positive_clips_dir = Path(positive_clips_dir_value) if positive_clips_dir_value else None
    positive_clip_min_confidence = float(settings.get("POSITIVE_CLIP_MIN_CONFIDENCE", 0.0))

    if args.check:
        return run_preflight_check(
            config_path=config_path,
            settings=settings,
            mount_root=mount_root,
            detector_model_path=detector_model_path,
            classifier_model_path=classifier_model_path,
        )

    with file_lock(lock_file):
        retention_result = run_retention(
            output_dir=output_dir,
            positive_clips_dir=positive_clips_dir,
            retention_days=retention_days,
        )
        total_deleted = sum(retention_result.values())
        if total_deleted:
            print(f"Retention cleanup removed {total_deleted} old files: {retention_result}")

        if not mount_exists(mount_root):
            raise SystemExit(f"Mount path does not exist: {mount_root}")
        if not mount_is_ready(mount_root):
            raise SystemExit(f"Mount path exists but is not mounted: {mount_root}")

        if args.hourly:
            if args.clip is not None:
                raise SystemExit("--clip cannot be combined with --hourly")

            if (args.window_start is None) != (args.window_end is None):
                raise SystemExit("--window-start and --window-end must be provided together")

            all_clip_paths = list_mp4_files(mount_root)
            if camera_name:
                all_clip_paths = [
                    path for path in all_clip_paths if get_clip_camera_name(path) == camera_name
                ]

            now_local = datetime.now(LOCAL_TIMEZONE)
            latest_completed_hour = floor_to_hour(now_local)
            windows: list[tuple[datetime, datetime]]

            if args.window_start and args.window_end:
                window_start = parse_datetime_arg(args.window_start)
                window_end = parse_datetime_arg(args.window_end)
                if window_end <= window_start:
                    raise SystemExit("--window-end must be after --window-start")
                windows = [(window_start, window_end)]
            else:
                state_path = state_file_path(state_dir, camera_name)
                state = load_state(state_path)
                if state and state.get("last_successful_window_end_utc"):
                    start = parse_datetime_arg(str(state["last_successful_window_end_utc"])).astimezone(LOCAL_TIMEZONE)
                else:
                    start = latest_completed_hour - timedelta(hours=1)

                if start >= latest_completed_hour:
                    print("No completed hourly windows to process.")
                    return 0

                windows = iterate_hour_windows(start, latest_completed_hour)

            detector: Any | None = None
            classifier: Any | None = None
            device = select_device()
            if not args.dry_run:
                from core.classifier import DeepFauneClassifierRunner
                from core.detector import MegaDetectorRunner

                detector = MegaDetectorRunner(
                    detector_model_path,
                    confidence_threshold,
                    device,
                    detector_version,
                )
                classifier = DeepFauneClassifierRunner(classifier_model_path, device)
                detector.load()
                classifier.load()

            last_summary_path: Path | None = None
            for window_start, window_end in windows:
                last_summary_path = process_hourly_window(
                    window_start=window_start,
                    window_end=window_end,
                    clip_paths=all_clip_paths,
                    mount_root=mount_root,
                    output_dir=output_dir,
                    detector=detector,
                    classifier=classifier,
                    frame_sample_seconds=frame_sample_seconds,
                    confidence_threshold=confidence_threshold,
                    detector_version=detector_version,
                    classifier_model_path=classifier_model_path,
                    device=device,
                    camera_name=camera_name,
                    dry_run=args.dry_run,
                    copy_positive_clips=copy_positive_clips,
                    positive_clips_dir=positive_clips_dir,
                    positive_clip_min_confidence=positive_clip_min_confidence,
                )
                print(f"Wrote hourly summary: {last_summary_path}")

                if not args.dry_run and not args.window_start and not args.window_end:
                    state_payload = {
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                        "camera_name": camera_name,
                        "last_successful_window_start_utc": window_start.astimezone(timezone.utc).isoformat(),
                        "last_successful_window_end_utc": window_end.astimezone(timezone.utc).isoformat(),
                        "last_summary_path": str(last_summary_path),
                        "dry_run": args.dry_run,
                    }
                    save_state(state_file_path(state_dir, camera_name), state_payload)

            return 0

        if args.clip is not None:
            clip_path = Path(args.clip)
            if not clip_path.is_file():
                raise SystemExit(f"Clip not found: {clip_path}")
            clip_paths = [clip_path]
        else:
            clip_paths = list_mp4_files(mount_root)
            if args.limit is not None:
                clip_paths = clip_paths[: args.limit]

        if not clip_paths:
            raise SystemExit(f"No .mp4 clips found under {mount_root}")

        device = select_device()
        from core.classifier import DeepFauneClassifierRunner
        from core.detector import MegaDetectorRunner

        detector = MegaDetectorRunner(
            detector_model_path,
            confidence_threshold,
            device,
            detector_version,
        )
        classifier = DeepFauneClassifierRunner(classifier_model_path, device)
        detector.load()
        classifier.load()

        for clip_path in clip_paths:
            try:
                report = process_clip(
                    clip_path=clip_path,
                    mount_root=mount_root,
                    detector=detector,
                    classifier=classifier,
                    frame_sample_seconds=frame_sample_seconds,
                    confidence_threshold=confidence_threshold,
                    detector_version=detector_version,
                    classifier_model_path=classifier_model_path,
                    device=device,
                )
                report = maybe_copy_positive_clip(
                    report=report,
                    clip_path=clip_path,
                    mount_root=mount_root,
                    positive_clips_dir=positive_clips_dir,
                    copy_positive_clips=copy_positive_clips,
                    positive_clip_min_confidence=positive_clip_min_confidence,
                )
            except Exception as error:
                report = build_error_report(
                    clip_path=clip_path,
                    mount_root=mount_root,
                    error=error,
                    detector_model_path=detector_model_path,
                    detector_version=detector_version,
                    classifier_model_path=classifier_model_path,
                    confidence_threshold=confidence_threshold,
                    device=device,
                )
            report_name = safe_report_name(clip_path, mount_root)
            destination = output_dir / report_name
            write_json_report(report, destination)
            print(f"Wrote report: {destination}")
            if report["status"] == "error":
                return 1

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
