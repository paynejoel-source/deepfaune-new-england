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

DEFAULT_DFNE_TAXON_MAP = {
    "american_marten": "Martes americana",
    "bird_sp.": "Aves",
    "black_bear": "Ursus americanus",
    "bobcat": "Lynx rufus",
    "coyote": "Canis latrans",
    "domestic_cat": "Felis catus",
    "domestic_cow": "Bos taurus",
    "domestic_dog": "Canis lupus familiaris",
    "fisher": "Pekania pennanti",
    "gray_fox": "Urocyon cinereoargenteus",
    "gray_squirrel": "Sciurus carolinensis",
    "human": "Homo sapiens",
    "moose": "Alces alces",
    "mouse_sp.": "Rodentia",
    "opossum": "Didelphis virginiana",
    "raccoon": "Procyon lotor",
    "red_fox": "Vulpes vulpes",
    "red_squirrel": "Tamiasciurus hudsonicus",
    "skunk": "Mephitidae",
    "snowshoe_hare": "Lepus americanus",
    "white_tailed_deer": "Odocoileus virginianus",
    "wild_boar": "Sus scrofa",
    "wild_turkey": "Meleagris gallopavo",
    "no-species": "",
    "no_species": "",
}


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


def normalize_prediction_label(value: str) -> str:
    return slugify(value).replace("-", "_")


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
    normalized_prediction = normalize_prediction_label(prediction)
    taxon_map = {
        normalize_prediction_label(str(key)): str(value)
        for key, value in (settings.get("CAMTRAP_TAXON_MAP", {}) or {}).items()
    }
    if normalized_prediction in taxon_map:
        return taxon_map[normalized_prediction]
    if normalized_prediction in DEFAULT_DFNE_TAXON_MAP:
        return DEFAULT_DFNE_TAXON_MAP[normalized_prediction]
    return prediction.replace("_", " ")


def get_observation_mode(settings: dict[str, Any]) -> str:
    mode = str(settings.get("CAMTRAP_OBSERVATION_MODE", "detection")).strip().lower() or "detection"
    valid_modes = {"detection", "clip_top", "species_summary"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported CAMTRAP_OBSERVATION_MODE '{mode}'. "
            f"Expected one of: {', '.join(sorted(valid_modes))}."
        )
    return mode


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
    observation_mode = get_observation_mode(settings)

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

        if observation_mode == "clip_top":
            top_prediction = str(report.get("top_prediction") or "")
            rows.append(
                {
                    "observationID": f"{media_id}-top",
                    "deploymentID": deployment_ids[camera_name],
                    "mediaID": media_id,
                    "eventID": event_id,
                    "eventStart": event_start,
                    "eventEnd": event_end,
                    "observationLevel": "media",
                    "observationType": "animal" if top_prediction else "unknown",
                    "scientificName": map_prediction_to_scientific_name(top_prediction, settings) if top_prediction else "",
                    "count": int(report.get("animal_detection_count", 0)) or "",
                    "bboxX": "",
                    "bboxY": "",
                    "bboxWidth": "",
                    "bboxHeight": "",
                    "classificationMethod": "machine",
                    "classifiedBy": classified_by,
                    "classificationTimestamp": str(report.get("generated_at")),
                    "classificationProbability": report.get("top_prediction_confidence", ""),
                    "observationTags": f"camera:{camera_name}|mode:clip_top|predicted_label:{top_prediction or 'unknown'}",
                }
            )
            continue

        if observation_mode == "species_summary":
            species_counts = report.get("species_counts", {}) or {}
            species_confidence_summary = report.get("species_confidence_summary", {}) or {}
            for species_label, count in sorted(species_counts.items()):
                observation_index += 1
                rows.append(
                    {
                        "observationID": f"{media_id}-species-{observation_index}",
                        "deploymentID": deployment_ids[camera_name],
                        "mediaID": media_id,
                        "eventID": event_id,
                        "eventStart": event_start,
                        "eventEnd": event_end,
                        "observationLevel": "media",
                        "observationType": "animal",
                        "scientificName": map_prediction_to_scientific_name(str(species_label), settings),
                        "count": int(count),
                        "bboxX": "",
                        "bboxY": "",
                        "bboxWidth": "",
                        "bboxHeight": "",
                        "classificationMethod": "machine",
                        "classifiedBy": classified_by,
                        "classificationTimestamp": str(report.get("generated_at")),
                        "classificationProbability": species_confidence_summary.get(species_label, ""),
                        "observationTags": (
                            f"camera:{camera_name}|mode:species_summary|predicted_label:{species_label}"
                        ),
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

    contact_name = str(settings.get("CAMTRAP_CONTACT_NAME", "Repository Maintainer"))
    contact_email = str(settings.get("CAMTRAP_CONTACT_EMAIL", "maintainer@example.com"))
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


def build_export_metadata(
    clip_reports: list[dict[str, Any]],
    settings: dict[str, Any],
    destination_dir: Path,
) -> dict[str, Any]:
    species_labels = sorted(
        {
            str(label)
            for report in clip_reports
            for label in (report.get("species_counts", {}) or {}).keys()
            if str(label)
        }
    )
    normalized_custom_taxa = {
        normalize_prediction_label(str(key)): str(value)
        for key, value in (settings.get("CAMTRAP_TAXON_MAP", {}) or {}).items()
    }
    unresolved_labels = [
        label
        for label in species_labels
        if normalize_prediction_label(label) not in normalized_custom_taxa
        and normalize_prediction_label(label) not in DEFAULT_DFNE_TAXON_MAP
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "camtrap_dp_version": CAMTRAP_DP_VERSION,
        "observation_mode": get_observation_mode(settings),
        "source_report_count": len(clip_reports),
        "camera_names": sorted({str(report["camera_name"]) for report in clip_reports}),
        "species_labels_detected": species_labels,
        "unresolved_species_labels": unresolved_labels,
        "output_files": [
            "datapackage.json",
            "deployments.csv",
            "media.csv",
            "observations.csv",
            "export_info.json",
        ],
        "destination_dir": str(destination_dir),
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
    export_metadata = build_export_metadata(clip_reports, settings, destination_dir)

    destination_dir.mkdir(parents=True, exist_ok=True)
    serialize_csv_rows(deployment_rows, DEPLOYMENTS_FIELDS, destination_dir / "deployments.csv")
    serialize_csv_rows(media_rows, MEDIA_FIELDS, destination_dir / "media.csv")
    serialize_csv_rows(observation_rows, OBSERVATION_FIELDS, destination_dir / "observations.csv")
    write_json_report(datapackage, destination_dir / "datapackage.json")
    write_json_report(export_metadata, destination_dir / "export_info.json")
    return destination_dir
