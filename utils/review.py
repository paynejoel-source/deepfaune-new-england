from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from utils.reporting import write_json_report


def build_review_summary(hourly_summary: dict[str, Any]) -> dict[str, Any]:
    """Extract the human-reviewable slice of an hourly summary."""
    return {
        "generated_at": hourly_summary.get("generated_at"),
        "camera_name": hourly_summary.get("camera_name"),
        "window_start_local": hourly_summary.get("window_start_local"),
        "window_end_local": hourly_summary.get("window_end_local"),
        "status": hourly_summary.get("status"),
        "clip_count_processed": hourly_summary.get("clip_count_processed", 0),
        "clip_count_with_detections": hourly_summary.get("clip_count_with_detections", 0),
        "total_animal_detections": hourly_summary.get("total_animal_detections", 0),
        "species_counts": hourly_summary.get("species_counts", {}),
        "detector_class_counts": hourly_summary.get("detector_class_counts", {}),
        "human_present": bool(hourly_summary.get("human_present")),
        "vehicle_present": bool(hourly_summary.get("vehicle_present")),
        "clip_errors": hourly_summary.get("clip_errors", []),
        "top_frames": hourly_summary.get("top_frames", []),
        "processed_reports": hourly_summary.get("processed_reports", []),
        "copied_positive_clips": hourly_summary.get("copied_positive_clips", []),
    }


def build_review_html(hourly_summary: dict[str, Any]) -> str:
    """Render a lightweight HTML review page for an hourly summary."""
    summary = build_review_summary(hourly_summary)
    species_items = "".join(
        f"<li><strong>{escape(str(species))}</strong>: {count}</li>"
        for species, count in sorted(summary["species_counts"].items())
    ) or "<li>No species classifications</li>"

    detector_items = "".join(
        f"<li><strong>{escape(str(label))}</strong>: {count}</li>"
        for label, count in sorted(summary["detector_class_counts"].items())
    ) or "<li>No detections</li>"

    top_frame_rows = "".join(
        (
            "<tr>"
            f"<td>{escape(str(frame.get('prediction', '')))}</td>"
            f"<td>{frame.get('confidence', '')}</td>"
            f"<td>{escape(str(frame.get('timestamp_local', '')))}</td>"
            f"<td>{escape(str(frame.get('relative_clip_path', '')))}</td>"
            "</tr>"
        )
        for frame in summary["top_frames"]
    ) or "<tr><td colspan='4'>No top frames</td></tr>"

    report_items = "".join(
        (
            "<li>"
            f"{escape(str(report.get('relative_clip_path', '')))}"
            f" [{escape(str(report.get('status', 'unknown')))}]"
            f"{' - ' + escape(str(report.get('report_path'))) if report.get('report_path') else ''}"
            "</li>"
        )
        for report in summary["processed_reports"]
    ) or "<li>No processed reports</li>"

    error_items = "".join(
        (
            "<li>"
            f"{escape(str(error.get('relative_clip_path', '')))}"
            f" - {escape(str(error.get('error_type', 'Error')))}"
            f": {escape(str(error.get('error_message', '')))}"
            "</li>"
        )
        for error in summary["clip_errors"]
    ) or "<li>No clip errors</li>"

    positive_clip_items = "".join(
        f"<li>{escape(str(path))}</li>" for path in summary["copied_positive_clips"]
    ) or "<li>No copied positive clips</li>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>DeepFaune Review Summary</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 2rem; background: #f8f5ef; color: #1f2328; }}
    h1, h2 {{ color: #243b2f; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }}
    .card {{ background: #fff; border: 1px solid #d8d2c4; border-radius: 10px; padding: 0.9rem 1rem; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #d8d2c4; padding: 0.55rem; text-align: left; }}
    th {{ background: #ebe5d8; }}
    ul {{ margin-top: 0.4rem; }}
    code {{ background: #f0eadf; padding: 0.1rem 0.3rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>DeepFaune Review Summary</h1>
  <div class="meta">
    <div class="card"><strong>Camera</strong><br>{escape(str(summary['camera_name']))}</div>
    <div class="card"><strong>Window Start</strong><br>{escape(str(summary['window_start_local']))}</div>
    <div class="card"><strong>Window End</strong><br>{escape(str(summary['window_end_local']))}</div>
    <div class="card"><strong>Status</strong><br>{escape(str(summary['status']))}</div>
    <div class="card"><strong>Processed Clips</strong><br>{summary['clip_count_processed']}</div>
    <div class="card"><strong>Animal Detections</strong><br>{summary['total_animal_detections']}</div>
  </div>

  <h2>Species Counts</h2>
  <ul>{species_items}</ul>

  <h2>Detector Class Counts</h2>
  <ul>{detector_items}</ul>

  <h2>Top Frames</h2>
  <table>
    <thead>
      <tr><th>Prediction</th><th>Confidence</th><th>Local Time</th><th>Clip</th></tr>
    </thead>
    <tbody>
      {top_frame_rows}
    </tbody>
  </table>

  <h2>Processed Reports</h2>
  <ul>{report_items}</ul>

  <h2>Copied Positive Clips</h2>
  <ul>{positive_clip_items}</ul>

  <h2>Clip Errors</h2>
  <ul>{error_items}</ul>
</body>
</html>
"""


def write_review_bundle(hourly_summary: dict[str, Any], destination_root: Path) -> tuple[Path, Path]:
    """Write JSON and HTML review artifacts for an hourly summary."""
    destination_root.mkdir(parents=True, exist_ok=True)
    summary_path = write_json_report(build_review_summary(hourly_summary), destination_root / "review_summary.json")
    html_path = destination_root / "review_summary.html"
    html_path.write_text(build_review_html(hourly_summary), encoding="utf-8")
    return summary_path, html_path
