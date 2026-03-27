from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DetectionRecord:
    label: str
    class_id: int
    confidence: float
    bbox_xyxy: list[float]
    bbox_normalized: list[float]


class MegaDetectorRunner:
    """Thin wrapper around PytorchWildlife's MegaDetector model."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float,
        device: str,
        version: str,
    ) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.version = version
        self._model: Any | None = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"MegaDetector weights not found at {self.model_path}. "
                "Place the .pt file in models/ and update DETECTOR_MODEL_PATH."
            )

        from PytorchWildlife.models import detection as pw_detection

        self._model = pw_detection.MegaDetectorV6(
            weights=str(self.model_path),
            device=self.device,
            version=self.version,
        )

    def detect_directory(self, image_dir: str | Path) -> list[dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Detector not loaded. Call load() first.")

        return self._model.batch_image_detection(
            str(image_dir),
            det_conf_thres=self.confidence_threshold,
        )

    def normalize_result(self, result: dict[str, Any]) -> dict[str, Any]:
        detections = result["detections"]
        normalized_coords = result.get("normalized_coords", [])

        records: list[DetectionRecord] = []
        for index, xyxy in enumerate(detections.xyxy.tolist()):
            class_id = int(detections.class_id[index])
            confidence = float(detections.confidence[index])
            bbox_normalized = (
                [float(value) for value in normalized_coords[index]]
                if index < len(normalized_coords)
                else []
            )
            records.append(
                DetectionRecord(
                    label=self._model.CLASS_NAMES[class_id],
                    class_id=class_id,
                    confidence=confidence,
                    bbox_xyxy=[float(value) for value in xyxy],
                    bbox_normalized=bbox_normalized,
                )
            )

        return {
            "img_id": result["img_id"],
            "labels": list(result.get("labels", [])),
            "detections": records,
        }
