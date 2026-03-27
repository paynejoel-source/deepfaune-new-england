from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import run_pipeline
from utils.locking import file_lock


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


if __name__ == "__main__":
    unittest.main()
