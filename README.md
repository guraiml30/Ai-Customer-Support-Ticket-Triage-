# AI Customer Support Ticket Triage

Reads incoming support tickets, predicts their **category** and
**urgency**, and routes them to the right support queue. Built with
scikit-learn on TF-IDF features; Streamlit for the dashboard, per the
suggested stack.

## How the six steps map to this repo

| Step | Where |
|---|---|
| 1. Collect historical ticket text + labels | `sample_data/tickets.csv` |
| 2. Clean text, category labels | `src/preprocess.py`, `src/schema.py` |
| 3. TF-IDF features | `src/features.py` |
| 4. Train multiclass + urgency models | `src/train.py` |
| 5. Evaluate: F1, confusion matrix, precision/recall | `src/evaluate.py` |
| 6. Expose via dashboard + log low-confidence for review | `app.py`, `src/predict.py`, `src/review_log.py` |

## Setup

```bash
pip install -r requirements.txt
```

## Run everything

```bash
python run_all.py
```

This trains both classifiers on `sample_data/tickets.csv`, prints
per-class precision/recall/F1 and a confusion matrix for each model,
saves the trained models to `models/`, then launches the Streamlit
dashboard (`localhost:8501`). Use `python run_all.py --no-app` to skip
launching the dashboard.

## The dashboard

- **Triage a ticket** — paste ticket text, get predicted category,
  urgency, confidence, and which queue it's routed to.
- **Low-confidence review queue** — any prediction below the
  confidence threshold (55% by default, see `LOW_CONFIDENCE_THRESHOLD`
  in `src/schema.py`) is logged to `review_logs/low_confidence_queue.csv`
  instead of being silently auto-routed. Browse and download that
  queue from the second tab.

## Models

Two independent `LogisticRegression` classifiers share one TF-IDF
vectorizer (unigrams + bigrams): one predicts `category` (billing,
technical, account, product), the other predicts `urgency` (low,
medium, high). Both are saved with `joblib` to `models/` after
training and loaded by `src/predict.py` for inference.

## Extending

- The sample dataset is intentionally small (50 tickets) so results
  are decent but not polished — add more labeled tickets to
  `sample_data/tickets.csv` (same three columns: `text`, `category`,
  `urgency`) to improve accuracy.
- Swap `LogisticRegression` for a stronger classifier (e.g.
  `RandomForestClassifier`, or a fine-tuned transformer) in
  `src/train.py` without touching the rest of the pipeline.
- `src/predict.py`'s `TriageModel` class is the seam for a REST API
  wrapper (e.g. FastAPI) if you outgrow the Streamlit dashboard.
- Git is already initialized in this folder — push to GitHub with
  `git remote add origin <url> && git push -u origin main`.
