import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_classifier, format_confusion
from src.predict import TriageModel
from src.review_log import log_if_low_confidence
from src.schema import CATEGORIES, URGENCY_LEVELS
from src.train import train_models

CSV_PATH = Path(__file__).resolve().parents[1] / "sample_data" / "tickets.csv"


def test_training_and_evaluation():
    artifacts = train_models(CSV_PATH)
    test_df = artifacts["test_df"]
    X_test = artifacts["X_test"]

    cat_report = evaluate_classifier(
        artifacts["category_model"], X_test, test_df["category"], CATEGORIES, "category"
    )
    urg_report = evaluate_classifier(
        artifacts["urgency_model"], X_test, test_df["urgency"], URGENCY_LEVELS, "urgency"
    )

    print(f"\nCategory model - macro F1: {cat_report.macro_f1}, accuracy: {cat_report.accuracy}")
    print(cat_report.report_text)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(format_confusion(cat_report))

    print(f"\nUrgency model - macro F1: {urg_report.macro_f1}, accuracy: {urg_report.accuracy}")
    print(urg_report.report_text)

    assert cat_report.macro_f1 > 0
    assert urg_report.macro_f1 > 0


def test_prediction_and_review_logging():
    model = TriageModel()
    sample_texts = [
        "I was billed twice this month and need a refund immediately.",
        "The app keeps crashing when I open the reports tab.",
        "asdkj qwepo random gibberish text that matches nothing",
    ]
    for text in sample_texts:
        pred = model.predict(text)
        print(f"\n{text!r}\n  -> category={pred.predicted_category} ({pred.category_confidence:.2f}), "
              f"urgency={pred.predicted_urgency} ({pred.urgency_confidence:.2f}), "
              f"queue={pred.queue}, needs_review={pred.needs_human_review}")
        log_if_low_confidence(pred)


if __name__ == "__main__":
    test_training_and_evaluation()
    test_prediction_and_review_logging()
