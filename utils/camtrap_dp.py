from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.reporting import read_json_report, write_json_report


CAMTRAP_DP_VERSION = "1.0.2"
PROFILE_URL = (
    f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}/camtrap-dp-profile.json"
)
SCHEMA_BASE_URL = f"https://raw.githubusercontent.com/tdwg/camtrap-dp/{CAMTRAP_DP_VERSION}"

DEPLOYMENTS_FIELDS = [
    "deploymentID",
    "locationID",
    "locationName",
    "latitude",
    "longitude",
    "deploymentStart",
    "deploymentEnd",
    "cameraID",
    "cameraModel",
    "deploymentTags",
    "deploymentComments",
]

MEDIA_FIELDS = [
    "mediaID",
    "deploymentID",
    "captureMethod",
    "timestamp",
    "filePath",
    "filePublic",
    "fileName",
    "fileMediatype",
    "mediaComments",
]

OBSERVATION_FIELDS = [
    "observationID",
    "deploymentID",
    "mediaID",
    "eventID",
    "eventStart",
    "eventEnd",
    "observationLevel",
    "observationType",
    "scientificName",
    "count",
    "bboxX",
    "bboxY",
    "bboxWidth",
    "bboxHeight",
    "classificationMethod",
    "classifiedBy",
    "classificationTimestamp",
    "classificationProbability",
    "observationTags",
]


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-").lower() or "unknown"


def serialize_csv_rows(rows: list[dict[str, Any]], fields: list[str], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)
    return destination


def list_clip_report_files(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        raise ValueError(f"Reports directory not found: {reports_dir}")

    candidates = sorted(reports_dir.rglob("*.json"))
    clip_reports: list[Path] = []
    for path in candidates:
        if any(part in {"hourly", "state", "camtrap_dp"} for part in path.parts):
            continue
        try:
            payload = read_json_report(path)
        except Exception:
            continue
        if {"clip_path", "relative_clip_path", "camera_name", "frames"}.issubset(payload.keys()):
            clip_reports.append(path)
    return clip_reports


def build_media_id(relative_clip_path: str) -> str:
    return slugify(relative_clip_path.replace("/", "-").replace(".mp4", ""))


def get_camera_metadata(settings: dict[str, Any], camera_name: str) -> dict[str, Any]:
    per_camera = settings.get("CAMTRAP_CAMERA_LOCATIONS", {}) or {}
    if camera_name in per_camera:
        camera_settings = dict(per_camera[camera_name] or {})
    else:
        camera_settings = {}

    if "latitude" not in camera_settings and settings.get("CAMTRAP_LATITUDE") is not None:
        camera_settings["latitude"] = settings["CAMTRAP_LATITUDE"]
    if "longitude" not in camera_settings and settings.get("CAMTRAP_LONGITUDE") is not None:
        camera_settings["longitude"] = settings["CAMTRAP_LONGITUDE"]
    if "locationName" not in camera_settings and settings.get("CAMTRAP_LOCATION_NAME"):
        camera_settings["locationName"] = settings["CAMTRAP_LOCATION_NAME"]
    if "locationID" not in camera_settings and settings.get("CAMTRAP_LOCATION_ID"):
        camera_settings["locationID"] = settings["CAMTRAP_LOCATION_ID"]
    if "cameraModel" not in camera_settings and settings.get("CAMTRAP_CAMERA_MODEL"):
        camera_settings["cameraModel"] = settings["CAMTRAP_CAMERA_MODEL"]

    missing = [key for key in ("latitude", "longitude") if camera_settings.get(key) is None]
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(
            f"Missing Camtrap DP camera metadata for '{camera_name}': {missing_keys}. "
            "Set CAMTRAP_CAMERA_LOCATIONS or global CAMTRAP_LATITUDE/CAMTRAP_LONGITUDE in config."
        )

    return camera_settings


def map_prediction_to_scientific_name(prediction: str, settings: dict[str, Any]) -> str:
    taxon_map = settings.get("CAMTRAP_TAXON_MAP", {}) or {}
    return str(taxon_map.get(prediction, prediction.replace("_", " ")))


def build_deployment_rows(
    clip_reports: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_camera: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in clip_reports:
        by_camera[str(report["camera_name"])].append(report)

    rows: list[dict[str, Any]] = []
    deployment_ids: dict[str, str] = {}
    deployment_prefix = str(settings.get("CAMTRAP_DEPLOYMENT_PREFIX", "dep")).strip() or "dep"

    for camera_name, reports in sorted(by_camera.items()):
        camera_settings = get_camera_metadata(settings, camera_name)
        deployment_id = f"{deployment_prefix}-{slugify(camera_name)}"
        deployment_ids[camera_name] = deployment_id

        start_times = [str(report["clip_start_time_utc"]) for report in reports if report.get("clip_start_time_utc")]
        end_times = [str(report["clip_end_time_utc"]) for report in reports if report.get("clip_end_time_utc")]
        if not start_times or not end_times:
            raise ValueError(
                f"Camera '{camera_name}' is missing clip timestamps required for Camtrap DP deployment export."
            )

        rows.append(
            {
                "deploymentID": deployment_id,
                "locationID": str(camera_settings.get("locationID", slugify(camera_name))),
                "locationName": str(camera_settings.get("locationName", camera_name.replace("_", " "))),
                "latitude": float(camera_settings["latitude"]),
                "longitude": float(camera_settings["longitude"]),
                "deploymentStart": min(start_times),
                "deploymentEnd": max(end_times),
                "cameraID": str(camera_settings.get("cameraID", camera_name)),
                "cameraModel": str(camera_settings.get("cameraModel", "")),
                "deploymentTags": str(camera_settings.get("deploymentTags", "")),
                "deploymentComments": str(camera_settings.get("deploymentComments", "")),
            }
        )

    return rows, deployment_ids


def build_media_rows(
    clip_reports: list[dict[str, Any]],
    deployment_ids: dict[str, str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    capture_method = str(settings.get("CAMTRAP_CAPTURE_METHOD", "activityDetection"))
    media_rows: list[dict[str, Any]] = []
    for report in clip_reports:
        relative_path = str(report["relative_clip_path"])
        media_rows.append(
            {
                "mediaID": build_media_id(relative_path),
                "deploymentID": deployment_ids[str(report["camera_name"])],
                "captureMethod": capture_method,
                "timestamp": str(report.get("clip_start_time_utc") or report.get("generated_at")),
                "filePath": relative_path,
                "filePublic": False,
                "fileName": Path(relative_path).name,
                "fileMediatype": "video/mp4",
                "mediaComments": f"status:{report.get('status', 'unknown')}",
            }
        )
    return media_rows


def bbox_from_detection(detection: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None]:
    bbox = detection.get("bbox_normalized")
    if not bbox or len(bbox) != 4:
        return None, None, None, None
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)


def build_observation_rows(
    clip_reports: list[dict[str, Any]],
    deployment_ids: dict[str, str],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    classified_by = str(
        settings.get("CAMTRAP_CLASSIFIED_BY", "MegaDetector + DeepFaune New England pipeline")
    )

    for report in clip_reports:
        camera_name = str(report["camera_name"])
        relative_path = str(report["relative_clip_path"])
        media_id = build_media_id(relative_path)
        event_id = slugify(relative_path.replace("/", "-").replace(".mp4", ""))
        event_start = str(report.get("clip_start_time_utc") or report.get("generated_at"))
        event_end = str(report.get("clip_end_time_utc") or report.get("clip_start_time_utc") or report.get("generated_at"))

        observation_index = 0
        if not report.get("had_detections"):
            rows.append(
                {
                    "observationID": f"{media_id}-blank",
                    "deploymentID": deployment_ids[camera_name],
                    "mediaID": media_id,
                    "eventID": event_id,
                    "eventStart": event_start,
                    "eventEnd": event_end,
                    "observationLevel": "media",
                    "observationType": "blank",
                    "scientificName": "",
                    "count": "",
                    "bboxX": "",
                    "bboxY": "",
                    "bboxWidth": "",
                    "bboxHeight": "",
                    "classificationMethod": "machine",
                    "classifiedBy": classified_by,
                    "classificationTimestamp": str(report.get("generated_at")),
                    "classificationProbability": "",
                    "observationTags": f"camera:{camera_name}|report_status:{report.get('status', 'unknown')}",
                }
            )
            continue

        for frame in report.get("frames", []):
            for detection in frame.get("detections", []):
                observation_index += 1
                bbox_x, bbox_y, bbox_width, bbox_height = bbox_from_detection(detection)
                observation_type = "unknown"
                scientific_name = ""
                probability = detection.get("confidence", "")
                tags = [
                    f"camera:{camera_name}",
                    f"detector_label:{detection.get('label', 'unknown')}",
                    f"frame_index:{frame.get('frame_index', '')}",
                ]

                if detection.get("label") == "person":
                    observation_type = "human"
                elif detection.get("label") == "vehicle":
                    observation_type = "vehicle"
                elif detection.get("classification"):
                    classification = detection["classification"]
                    observation_type = "animal"
                    scientific_name = map_prediction_to_scientific_name(
                        str(classification.get("prediction", "unknown")),
                        settings,
                    )
                    probability = classification.get("confidence", probability)
                    tags.append(f"predicted_label:{classification.get('prediction', 'unknown')}")
                elif detection.get("label") == "animal":
                    observation_type = "unknown"

                rows.append(
                    {
                        "observationID": f"{media_id}-obs-{observation_index}",
                        "deploymentID": deployment_ids[camera_name],
                        "mediaID": media_id,
                        "eventID": event_id,
                        "eventStart": event_start,
                        "eventEnd": event_end,
                        "observationLevel": "media",
                        "observationType": observation_type,
                        "scientificName": scientific_name,
                        "count": 1 if observation_type != "blank" else "",
                        "bboxX": bbox_x if bbox_x is not None else "",
                        "bboxY": bbox_y if bbox_y is not None else "",
                        "bboxWidth": bbox_width if bbox_width is not None else "",
                        "bboxHeight": bbox_height if bbox_height is not None else "",
                        "classificationMethod": "machine",
                        "classifiedBy": classified_by,
                        "classificationTimestamp": str(report.get("generated_at")),
                        "classificationProbability": probability if probability != "" else "",
                        "observationTags": "|".join(tags),
                    }
                )
    return rows


def build_datapackage(
    deployment_rows: list[dict[str, Any]],
    observation_rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    coordinates = [[row["longitude"], row["latitude"]] for row in deployment_rows]
    temporal_start = min(str(row["deploymentStart"])[:10] for row in deployment_rows)
    temporal_end = max(str(row["deploymentEnd"])[:10] for row in deployment_rows)
    scientific_names = sorted({str(row["scientificName"]) for row in observation_rows if row.get("scientificName")})

    contact_name = str(settings.get("CAMTRAP_CONTACT_NAME", "Joel Payne"))
    contact_email = str(settings.get("CAMTRAP_CONTACT_EMAIL", "payne.joel@gmail.com"))
    project_title = str(settings.get("CAMTRAP_PROJECT_TITLE", "DeepFaune New England Export"))
    project_description = str(
        settings.get(
            "CAMTRAP_PROJECT_DESCRIPTION",
            "Independent wildlife video-processing export generated from DeepFaune New England clip reports.",
        )
    )

    return {
        "name": slugify(str(settings.get("CAMTRAP_PACKAGE_NAME", "deepfaune-new-england-camtrap-dp"))),
        "id": str(settings.get("CAMTRAP_PACKAGE_ID", "https://github.com/paynejoel-source/deepfaune-new-england")),
        "profile": PROFILE_URL,
        "created": datetime.now(timezone.utc).isoformat(),
        "title": project_title,
        "description": project_description,
        "contributors": [
            {
                "title": contact_name,
                "email": contact_email,
                "role": "contact",
            }
        ],
        "project": {
            "id": str(settings.get("CAMTRAP_PROJECT_ID", slugify(project_title))),
            "title": project_title,
            "description": project_description,
            "samplingDesign": str(settings.get("CAMTRAP_SAMPLING_DESIGN", "targeted")),
            "captureMethod": [str(settings.get("CAMTRAP_CAPTURE_METHOD", "activityDetection"))],
            "individualAnimals": bool(settings.get("CAMTRAP_INDIVIDUAL_ANIMALS", False)),
            "observationLevel": ["media"],
        },
        "spatial": {
            "type": "Feature",
            "geometry": {
                "type": "MultiPoint",
                "coordinates": coordinates,
            },
            "properties": {},
        },
        "temporal": {
            "start": temporal_start,
            "end": temporal_end,
        },
        "taxonomic": [{"scientificName": name} for name in scientific_names],
        "resources": [
            {
                "name": "deployments",
                "path": "deployments.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "UTF-8",
                "schema": f"{SCHEMA_BASE_URL}/deployments-table-schema.json",
            },
            {
                "name": "media",
                "path": "media.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "UTF-8",
                "schema": f"{SCHEMA_BASE_URL}/media-table-schema.json",
            },
            {
                "name": "observations",
                "path": "observations.csv",
                "profile": "tabular-data-resource",
                "format": "csv",
                "mediatype": "text/csv",
                "encoding": "UTF-8",
                "schema": f"{SCHEMA_BASE_URL}/observations-table-schema.json",
            },
        ],
    }


def export_camtrap_dp_package(
    reports_dir: Path,
    destination_dir: Path,
    settings: dict[str, Any],
) -> Path:
    report_files = list_clip_report_files(reports_dir)
    if not report_files:
        raise ValueError(f"No clip reports found under {reports_dir}")

    clip_reports = [read_json_report(path) for path in report_files]
    deployment_rows, deployment_ids = build_deployment_rows(clip_reports, settings)
    media_rows = build_media_rows(clip_reports, deployment_ids, settings)
    observation_rows = build_observation_rows(clip_reports, deployment_ids, settings)
    datapackage = build_datapackage(deployment_rows, observation_rows, settings)

    destination_dir.mkdir(parents=True, exist_ok=True)
    serialize_csv_rows(deployment_rows, DEPLOYMENTS_FIELDS, destination_dir / "deployments.csv")
    serialize_csv_rows(media_rows, MEDIA_FIELDS, destination_dir / "media.csv")
    serialize_csv_rows(observation_rows, OBSERVATION_FIELDS, destination_dir / "observations.csv")
    write_json_report(datapackage, destination_dir / "datapackage.json")
    return destination_dir
