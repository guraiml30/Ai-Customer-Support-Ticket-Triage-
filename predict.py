from __future__ import annotations

from pathlib import Path

import joblib

from .schema import Prediction

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class TriageModel:
    """Loads the saved vectorizer + two classifiers once, then predicts cheaply."""

    def __init__(self, models_dir: Path | None = None):
        models_dir = models_dir or MODELS_DIR
        try:
            self.vectorizer = joblib.load(models_dir / "vectorizer.joblib")
            self.category_model = joblib.load(models_dir / "category_model.joblib")
            self.urgency_model = joblib.load(models_dir / "urgency_model.joblib")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"No trained models found in {models_dir}. Run `python -m src.train` "
                "(or run_all.py) to train them first."
            ) from e

    def predict(self, text: str) -> Prediction:
        X = self.vectorizer.transform([text])

        cat_probs = self.category_model.predict_proba(X)[0]
        cat_idx = cat_probs.argmax()
        predicted_category = self.category_model.classes_[cat_idx]
        category_confidence = float(cat_probs[cat_idx])

        urg_probs = self.urgency_model.predict_proba(X)[0]
        urg_idx = urg_probs.argmax()
        predicted_urgency = self.urgency_model.classes_[urg_idx]
        urgency_confidence = float(urg_probs[urg_idx])

        return Prediction(
            text=text,
            predicted_category=predicted_category,
            category_confidence=category_confidence,
            predicted_urgency=predicted_urgency,
            urgency_confidence=urgency_confidence,
        )
