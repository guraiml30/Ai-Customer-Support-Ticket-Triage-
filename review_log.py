from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .schema import Prediction

REVIEW_LOG_PATH = Path(__file__).resolve().parent.parent / "review_logs" / "low_confidence_queue.csv"

FIELDNAMES = [
    "logged_at",
    "text",
    "predicted_category",
    "category_confidence",
    "predicted_urgency",
    "urgency_confidence",
]


def log_if_low_confidence(prediction: Prediction, log_path: Path | None = None) -> bool:
    """Append to the review log if the prediction is below threshold. Returns True if logged."""
    if not prediction.needs_human_review:
        return False

    log_path = log_path or REVIEW_LOG_PATH
    log_path.parent.mkdir(exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "text": prediction.text,
                "predicted_category": prediction.predicted_category,
                "category_confidence": round(prediction.category_confidence, 3),
                "predicted_urgency": prediction.predicted_urgency,
                "urgency_confidence": round(prediction.urgency_confidence, 3),
            }
        )
    return True


def read_review_queue(log_path: Path | None = None) -> list[dict]:
    log_path = log_path or REVIEW_LOG_PATH
    if not log_path.exists():
        return []
    with open(log_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
