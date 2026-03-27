from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest import mock

import run_pipeline
from utils.config import load_pipeline_settings
from utils.locking import file_lock
from utils.reporting import read_json_report


class PipelineTests(unittest.TestCase):
    def test_parse_clip_epoch_range(self) -> None:
        start, end = run_pipeline.parse_clip_epoch_range(
            Path("1774126221.93046-1774126800.07907.mp4")
        )
        self.assertEqual(start, 1774126221.93046)
        self.assertEqual(end, 1774126800.07907)

    def test_parse_clip_epoch_range_invalid(self) -> None:
        start, end = run_pipeline.parse_clip_epoch_range(Path("not_a_frigate_name.mp4"))
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_select_clips_for_window_filters_by_camera_and_time(self) -> None:
        clips = [
            Path("/mnt/clips/previews/front_yard/1774126221.93046-1774126800.07907.mp4"),
            Path("/mnt/clips/previews/back_yard/1774126400.0-1774127000.0.mp4"),
            Path("/mnt/clips/previews/front_yard/notimestamp.mp4"),
        ]
        window_start = datetime(2026, 3, 21, 20, 0, tzinfo=timezone.utc)
        window_end = datetime(2026, 3, 21, 21, 0, tzinfo=timezone.utc)

        selected, skipped = run_pipeline.select_clips_for_window(
            clips,
            window_start,
            window_end,
            "front_yard",
        )

        self.assertEqual(selected, [clips[0]])
        self.assertEqual(skipped, [clips[2]])

    def test_build_error_report(self) -> None:
        report = run_pipeline.build_error_report(
            clip_path=Path("/tmp/test.mp4"),
            mount_root=Path("/tmp"),
            error=RuntimeError("boom"),
            detector_model_path=Path("models/det.pt"),
            detector_version="MDV6-yolov9-c",
            classifier_model_path=Path("models/cls.pth"),
            confidence_threshold=0.5,
            device="cpu",
        )

        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertEqual(report["error_message"], "boom")
        self.assertEqual(report["relative_clip_path"], "test.mp4")

    def test_positive_clip_copy_respects_confidence_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            dest_root = temp_root / "dest"
            clip_path = source_root / "cam" / "clip.mp4"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_text("clip", encoding="utf-8")

            report = {
                "animal_detection_count": 1,
                "top_prediction_confidence": 0.69,
                "positive_clip_copied": False,
                "positive_clip_path": None,
            }

            updated = run_pipeline.maybe_copy_positive_clip(
                report=report,
                clip_path=clip_path,
                mount_root=source_root,
                positive_clips_dir=dest_root,
                copy_positive_clips=True,
                positive_clip_min_confidence=0.7,
            )

            self.assertFalse(updated["positive_clip_copied"])
            self.assertIsNone(updated["positive_clip_path"])
            self.assertFalse((dest_root / "cam" / "clip.mp4").exists())

    def test_positive_clip_copy_preserves_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            source_root = temp_root / "source"
            dest_root = temp_root / "dest"
            clip_path = source_root / "previews" / "back_yard" / "clip.mp4"
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            clip_path.write_text("clip", encoding="utf-8")

            report = {
                "animal_detection_count": 1,
                "top_prediction_confidence": 0.95,
                "positive_clip_copied": False,
                "positive_clip_path": None,
            }

            updated = run_pipeline.maybe_copy_positive_clip(
                report=report,
                clip_path=clip_path,
                mount_root=source_root,
                positive_clips_dir=dest_root,
                copy_positive_clips=True,
                positive_clip_min_confidence=0.7,
            )

            expected_path = dest_root / "previews" / "back_yard" / "clip.mp4"
            self.assertTrue(updated["positive_clip_copied"])
            self.assertEqual(updated["positive_clip_path"], str(expected_path))
            self.assertTrue(expected_path.is_file())

    def test_file_lock_blocks_second_acquire(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "pipeline.lock"
            with file_lock(lock_path):
                with self.assertRaises(RuntimeError):
                    with file_lock(lock_path):
                        pass

    def test_load_pipeline_settings_reports_missing_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            mount_root = root / "clips"
            mount_root.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "pipeline_settings.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f'BEAST_MOUNT_PATH: "{mount_root}"',
                        'DETECTOR_MODEL_PATH: "models/MDV6-yolov9-c.pt"',
                        'CLASSIFIER_MODEL_PATH: "models/dfne_weights_v1_0.pth"',
                        "CONFIDENCE_THRESHOLD: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as context:
                load_pipeline_settings(config_path)

            self.assertIn("Detector model not found", str(context.exception))
            self.assertIn("Classifier model not found", str(context.exception))

    def test_load_pipeline_settings_returns_resolved_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / "config"
            models_dir = root / "models"
            mount_root = root / "clips"
            config_dir.mkdir(parents=True, exist_ok=True)
            models_dir.mkdir(parents=True, exist_ok=True)
            mount_root.mkdir(parents=True, exist_ok=True)
            (models_dir / "MDV6-yolov9-c.pt").write_text("detector", encoding="utf-8")
            (models_dir / "dfne_weights_v1_0.pth").write_text("classifier", encoding="utf-8")
            config_path = config_dir / "pipeline_settings.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f'BEAST_MOUNT_PATH: "{mount_root}"',
                        'DETECTOR_MODEL_PATH: "models/MDV6-yolov9-c.pt"',
                        'CLASSIFIER_MODEL_PATH: "models/dfne_weights_v1_0.pth"',
                        "CONFIDENCE_THRESHOLD: 0.5",
                    ]
                ),
                encoding="utf-8",
            )

            settings = load_pipeline_settings(config_path)

            self.assertEqual(settings["mount_root"], mount_root)
            self.assertEqual(settings["detector_model_path"], models_dir / "MDV6-yolov9-c.pt")
            self.assertEqual(
                settings["classifier_model_path"],
                models_dir / "dfne_weights_v1_0.pth",
            )

    def test_run_preflight_check_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mount_root = root / "clips"
            detector_model_path = root / "models" / "detector.pt"
            classifier_model_path = root / "models" / "classifier.pth"
            detector_model_path.parent.mkdir(parents=True, exist_ok=True)
            mount_root.mkdir(parents=True, exist_ok=True)
            detector_model_path.write_text("detector", encoding="utf-8")
            classifier_model_path.write_text("classifier", encoding="utf-8")

            result = run_pipeline.run_preflight_check(
                config_path=root / "config" / "pipeline_settings.yaml",
                settings={"DEFAULT_CAMERA_NAME": "back_yard"},
                mount_root=mount_root,
                detector_model_path=detector_model_path,
                classifier_model_path=classifier_model_path,
            )

            self.assertEqual(result, 0)

    def test_process_hourly_window_dry_run_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            mount_root = root / "clips"
            clip_start_epoch = 1774126221.93046
            window_start = datetime.fromtimestamp(clip_start_epoch, tz=timezone.utc).replace(
                minute=0,
                second=0,
                microsecond=0,
            )
            window_end = window_start.replace(minute=0, second=0, microsecond=0)
            window_end = window_end.replace(hour=window_start.hour) + run_pipeline.timedelta(hours=1)
            clip_paths = [
                mount_root / "previews" / "front_yard" / "1774126221.93046-1774126800.07907.mp4",
                mount_root / "previews" / "front_yard" / "notimestamp.mp4",
            ]

            summary_path = run_pipeline.process_hourly_window(
                window_start=window_start,
                window_end=window_end,
                clip_paths=clip_paths,
                mount_root=mount_root,
                output_dir=output_dir,
                detector=None,
                classifier=None,
                frame_sample_seconds=1.0,
                confidence_threshold=0.5,
                detector_version="MDV6-yolov9-c",
                classifier_model_path=root / "models" / "classifier.pth",
                device="cpu",
                camera_name="front_yard",
                dry_run=True,
                copy_positive_clips=False,
                positive_clips_dir=None,
                positive_clip_min_confidence=0.7,
            )

            summary = read_json_report(summary_path)
            self.assertEqual(summary["status"], "dry_run")
            self.assertEqual(summary["clip_count_selected"], 1)
            self.assertEqual(summary["clip_count_processed"], 0)
            self.assertEqual(len(summary["selected_clips"]), 1)
            self.assertEqual(len(summary["skipped_without_timestamp"]), 1)

    def test_run_demo_writes_example_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output"

            result = run_pipeline.run_demo(output_dir)

            self.assertEqual(result, 0)
            clip_report_path = output_dir / "demo" / "clips" / "demo_clip_report.json"
            self.assertTrue(clip_report_path.is_file())
            clip_report = read_json_report(clip_report_path)
            self.assertEqual(clip_report["status"], "success")
            self.assertEqual(clip_report["top_prediction"], "white_tailed_deer")

            hourly_reports = list((output_dir / "demo" / "hourly").rglob("*.json"))
            self.assertEqual(len(hourly_reports), 1)
            hourly_report = read_json_report(hourly_reports[0])
            self.assertEqual(hourly_report["clip_count_processed"], 1)
            self.assertEqual(hourly_report["species_counts"], {"white_tailed_deer": 2})

    def test_main_demo_mode_bypasses_runtime_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            current_dir = Path.cwd()
            capture = StringIO()
            try:
                os.chdir(temp_dir)
                with redirect_stdout(capture):
                    with mock.patch("sys.argv", ["run_pipeline.py", "--demo"]):
                        result = run_pipeline.main()
            finally:
                os.chdir(current_dir)

            self.assertEqual(result, 0)
            self.assertIn("Wrote demo clip report", capture.getvalue())
            self.assertTrue((Path(temp_dir) / "output" / "demo" / "clips" / "demo_clip_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
