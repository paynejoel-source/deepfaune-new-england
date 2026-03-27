from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ClassificationRecord:
    prediction: str
    class_id: int
    confidence: float
    all_confidences: list[list[Any]]


class DeepFauneClassifierRunner:
    """Wrapper for the DeepFaune New England classifier."""

    def __init__(self, model_path: str | Path, device: str) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._model: Any | None = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Classifier weights not found at {self.model_path}. "
                "Place the .pth file in models/ and update CLASSIFIER_MODEL_PATH."
            )

        from PytorchWildlife.models.classification.timm_base.DFNE import DFNE
        from PytorchWildlife.models.classification.timm_base.base_classifier import (
            TIMM_BaseClassifierInference,
        )

        class LocalDFNE(TIMM_BaseClassifierInference):
            BACKBONE = DFNE.BACKBONE
            MODEL_NAME = DFNE.MODEL_NAME
            IMAGE_SIZE = DFNE.IMAGE_SIZE
            CLASS_NAMES = DFNE.CLASS_NAMES

            def __init__(self, weights: str, device: str) -> None:
                super().__init__(
                    weights=weights,
                    device=device,
                    url=None,
                    transform=None,
                    weights_key="model_state_dict",
                )

        self._model = LocalDFNE(
            weights=str(self.model_path),
            device=self.device,
        )

    def classify_detections(self, detection_results: list[dict[str, Any]]) -> list[ClassificationRecord]:
        if self._model is None:
            raise RuntimeError("Classifier not loaded. Call load() first.")

        has_animal_detections = any(
            any(int(class_id) == 0 for class_id in result["detections"].class_id)
            for result in detection_results
        )
        if not has_animal_detections:
            return []

        raw_results = self._model.batch_image_classification(det_results=detection_results)
        classifications: list[ClassificationRecord] = []

        for result in raw_results:
            all_confidences = [
                [str(label), float(score)]
                for label, score in result.get("all_confidences", [])
            ]
            classifications.append(
                ClassificationRecord(
                    prediction=str(result["prediction"]),
                    class_id=int(result["class_id"]),
                    confidence=float(result["confidence"]),
                    all_confidences=all_confidences,
                )
            )

        return classifications
