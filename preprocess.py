from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/extra whitespace, keep words and basic punctuation."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.,!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
