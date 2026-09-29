from __future__ import annotations

from dataclasses import dataclass

CATEGORIES = ["billing", "technical", "account", "product"]
URGENCY_LEVELS = ["low", "medium", "high"]


CATEGORY_TO_QUEUE = {
    "billing": "Billing Support Queue",
    "technical": "Technical Support Queue",
    "account": "Account Management Queue",
    "product": "Product Feedback Queue",
}


LOW_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class Ticket:
    text: str
    category: str | None = None  
    urgency: str | None = None  

@dataclass
class Prediction:
    text: str
    predicted_category: str
    category_confidence: float
    predicted_urgency: str
    urgency_confidence: float

    @property
    def needs_human_review(self) -> bool:
        return (
            self.category_confidence < LOW_CONFIDENCE_THRESHOLD
            or self.urgency_confidence < LOW_CONFIDENCE_THRESHOLD
        )

    @property
    def queue(self) -> str:
        return CATEGORY_TO_QUEUE.get(self.predicted_category, "General Queue")

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "predicted_category": self.predicted_category,
            "category_confidence": round(self.category_confidence, 3),
            "predicted_urgency": self.predicted_urgency,
            "urgency_confidence": round(self.urgency_confidence, 3),
            "queue": self.queue,
            "needs_human_review": self.needs_human_review,
        }
