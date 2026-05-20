from __future__ import annotations

from pathlib import Path

import joblib
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.src.model import SpamClassifier
from ml.src.preprocess import normalize_text


LABELS = ["ham", "spam"]


class SpamInferenceService:
    def __init__(self, model: SpamClassifier, vectorizer: TfidfVectorizer) -> None:
        self.model = model.eval()
        self.vectorizer = vectorizer

    @classmethod
    def load(cls, artifact_dir: str | Path) -> "SpamInferenceService":
        artifact_dir = Path(artifact_dir)
        vectorizer: TfidfVectorizer = joblib.load(artifact_dir / "vectorizer.joblib")

        model = SpamClassifier(input_dim=len(vectorizer.get_feature_names_out()))
        state_dict = torch.load(artifact_dir / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return cls(model, vectorizer)

    def predict(self, message: str) -> dict[str, float | str]:
        clean_text = normalize_text(message)
        features = self.vectorizer.transform([clean_text]).toarray()
        tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        ham_probability = float(probs[0].item())
        spam_probability = float(probs[1].item())
        label = LABELS[int(torch.argmax(probs).item())]

        return {
            "label": label,
            "ham_probability": round(ham_probability, 4),
            "spam_probability": round(spam_probability, 4),
        }

