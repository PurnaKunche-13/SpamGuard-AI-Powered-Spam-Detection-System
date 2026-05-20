from __future__ import annotations

from pathlib import Path

from ml.src.inference import SpamInferenceService


ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "ml" / "artifacts"


class PredictorManager:
    def __init__(self) -> None:
        self._service: SpamInferenceService | None = None

    def load(self) -> SpamInferenceService:
        if self._service is None:
            self._service = SpamInferenceService.load(ARTIFACT_DIR)
        return self._service

    def is_ready(self) -> bool:
        return (ARTIFACT_DIR / "model.pt").exists() and (ARTIFACT_DIR / "vectorizer.joblib").exists()

    def reload(self) -> SpamInferenceService:
        self._service = SpamInferenceService.load(ARTIFACT_DIR)
        return self._service

