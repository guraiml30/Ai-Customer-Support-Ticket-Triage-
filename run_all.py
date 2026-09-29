from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from src.evaluate import evaluate_classifier, format_confusion
from src.schema import CATEGORIES, URGENCY_LEVELS
from src.train import train_models

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "sample_data" / "tickets.csv"
APP_PATH = PROJECT_ROOT / "app.py"


def launch_streamlit_app() -> None:
    """Launch `streamlit run app.py` and block until the user stops it (Ctrl+C)."""
    if importlib.util.find_spec("streamlit") is None:
        print(
            "\nStreamlit isn't installed, so the app can't be launched.\n"
            "Install it with:  pip install -r requirements.txt\n"
            "Then run:         streamlit run app.py"
        )
        return

    print(f"\n{'=' * 60}\nLaunching Streamlit app (Ctrl+C to stop)...\n{'=' * 60}")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], cwd=str(PROJECT_ROOT))


def main() -> None:
    launch_app = "--no-app" not in sys.argv

    if not CSV_PATH.exists():
        print(f"No ticket data found at {CSV_PATH}")
        return

    print(f"Training on {CSV_PATH}...")
    artifacts = train_models(CSV_PATH)
    test_df = artifacts["test_df"]
    X_test = artifacts["X_test"]

    cat_report = evaluate_classifier(
        artifacts["category_model"], X_test, test_df["category"], CATEGORIES, "category"
    )
    urg_report = evaluate_classifier(
        artifacts["urgency_model"], X_test, test_df["urgency"], URGENCY_LEVELS, "urgency"
    )

    print(f"\n{'=' * 60}\nCategory model\n{'=' * 60}")
    print(f"macro F1: {cat_report.macro_f1}   accuracy: {cat_report.accuracy}\n")
    print(cat_report.report_text)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(format_confusion(cat_report))

    print(f"\n{'=' * 60}\nUrgency model\n{'=' * 60}")
    print(f"macro F1: {urg_report.macro_f1}   accuracy: {urg_report.accuracy}\n")
    print(urg_report.report_text)

    print(f"\n{'=' * 60}\nModels trained and saved to {PROJECT_ROOT / 'models'}\n{'=' * 60}")

    if launch_app:
        launch_streamlit_app()


if __name__ == "__main__":
    main()
