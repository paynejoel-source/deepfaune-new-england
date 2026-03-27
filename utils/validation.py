from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.reporting import read_json_report, write_json_report


def normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def load_validation_cases(source: Path) -> list[dict[str, str]]:
    if not source.is_file():
        raise ValueError(f"Validation file not found: {source}")

    if source.suffix.lower() == ".json":
        payload = read_json_report(source)
        cases = payload.get("cases", payload if isinstance(payload, list) else [])
        if not isinstance(cases, list):
            raise ValueError("Validation JSON must be a list or an object with a 'cases' list.")
        normalized_cases: list[dict[str, str]] = []
        for item in cases:
            if not isinstance(item, dict):
                continue
            normalized_cases.append(
                {
                    "relative_clip_path": str(item.get("relative_clip_path", "")).strip(),
                    "expected_label": str(item.get("expected_label", "")).strip(),
                }
            )
        return normalized_cases

    if source.suffix.lower() == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError("Validation CSV must contain headers.")
            if "relative_clip_path" not in reader.fieldnames or "expected_label" not in reader.fieldnames:
                raise ValueError("Validation CSV must include 'relative_clip_path' and 'expected_label' columns.")
            return [
                {
                    "relative_clip_path": str(row.get("relative_clip_path", "")).strip(),
                    "expected_label": str(row.get("expected_label", "")).strip(),
                }
                for row in reader
            ]

    raise ValueError("Validation file must be .csv or .json")


def list_clip_report_files(reports_dir: Path) -> list[Path]:
    if not reports_dir.exists():
        raise ValueError(f"Reports directory not found: {reports_dir}")
    return sorted(
        path
        for path in reports_dir.rglob("*.json")
        if all(part not in {"hourly", "state", "review", "camtrap_dp"} for part in path.parts)
    )


def build_report_index(reports_dir: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for report_path in list_clip_report_files(reports_dir):
        payload = read_json_report(report_path)
        if "relative_clip_path" not in payload:
            continue
        index[str(payload["relative_clip_path"])] = payload
    return index


def validate_reports(
    reports_dir: Path,
    validation_source: Path,
    destination: Path,
) -> Path:
    cases = load_validation_cases(validation_source)
    report_index = build_report_index(reports_dir)

    results: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for case in cases:
        relative_clip_path = case["relative_clip_path"]
        expected_label = case["expected_label"]
        report = report_index.get(relative_clip_path)

        if report is None:
            counts["missing_report"] += 1
            results.append(
                {
                    "relative_clip_path": relative_clip_path,
                    "expected_label": expected_label,
                    "predicted_label": None,
                    "status": "missing_report",
                }
            )
            continue

        predicted_label = str(report.get("top_prediction") or "")
        normalized_expected = normalize_label(expected_label)
        normalized_predicted = normalize_label(predicted_label)

        if not predicted_label:
            status = "no_prediction"
        elif normalized_expected == normalized_predicted:
            status = "match"
        else:
            status = "mismatch"

        counts[status] += 1
        results.append(
            {
                "relative_clip_path": relative_clip_path,
                "expected_label": expected_label,
                "predicted_label": predicted_label or None,
                "top_prediction_confidence": report.get("top_prediction_confidence"),
                "status": status,
            }
        )

    total_cases = len(cases)
    matched_cases = counts.get("match", 0)
    evaluable_cases = total_cases - counts.get("missing_report", 0)
    match_rate = (matched_cases / evaluable_cases) if evaluable_cases else 0.0

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reports_dir": str(reports_dir),
        "validation_source": str(validation_source),
        "total_cases": total_cases,
        "counts": dict(counts),
        "match_rate": round(match_rate, 6),
        "results": results,
    }
    return write_json_report(payload, destination)
