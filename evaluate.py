from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


@dataclass
class EvalReport:
    label_name: str
    macro_f1: float
    accuracy: float
    report_text: str
    confusion: np.ndarray
    labels: list[str]


def evaluate_classifier(model, X_test, y_true, labels: list[str], label_name: str) -> EvalReport:
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    accuracy = float((y_pred == y_true.values).mean())
    report_text = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return EvalReport(
        label_name=label_name,
        macro_f1=round(macro_f1, 3),
        accuracy=round(accuracy, 3),
        report_text=report_text,
        confusion=cm,
        labels=labels,
    )


def format_confusion(report: EvalReport) -> str:
    header = "        " + " ".join(f"{l[:8]:>8}" for l in report.labels)
    lines = [header]
    for i, label in enumerate(report.labels):
        row = " ".join(f"{v:>8}" for v in report.confusion[i])
        lines.append(f"{label[:8]:>8} {row}")
    return "\n".join(lines)
