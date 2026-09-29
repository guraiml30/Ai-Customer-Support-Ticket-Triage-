from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .features import build_vectorizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_tickets(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"text", "category", "urgency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tickets CSV is missing required column(s): {missing}")
    return df


def train_models(csv_path: str | Path, test_size: float = 0.25, random_state: int = 42) -> dict:
    """Train both classifiers and save them + the vectorizer to MODELS_DIR."""
    df = load_tickets(csv_path)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["category"]
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    category_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    category_model.fit(X_train, train_df["category"])

    urgency_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    urgency_model.fit(X_train, train_df["urgency"])

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    joblib.dump(category_model, MODELS_DIR / "category_model.joblib")
    joblib.dump(urgency_model, MODELS_DIR / "urgency_model.joblib")

    return {
        "vectorizer": vectorizer,
        "category_model": category_model,
        "urgency_model": urgency_model,
        "X_test": X_test,
        "test_df": test_df,
    }


if __name__ == "__main__":
    default_csv = Path(__file__).resolve().parent.parent / "sample_data" / "tickets.csv"
    train_models(default_csv)
    print(f"Models saved to {MODELS_DIR}")
