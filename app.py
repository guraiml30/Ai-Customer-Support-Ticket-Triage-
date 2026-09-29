"""
Step 6 in the plan: expose the model through a small dashboard, and log
low-confidence predictions for human review.

Run with:  streamlit run app.py
(Train models first with `python -m src.train` or `python run_all.py`.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import TriageModel
from src.review_log import log_if_low_confidence, read_review_queue
from src.schema import LOW_CONFIDENCE_THRESHOLD

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Support Ticket Triage", layout="wide")
st.title("🎫 AI Customer Support Ticket Triage")
st.caption("Predicts category and urgency for incoming tickets, and routes them to the right queue.")

try:
    model = TriageModel()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

tab_triage, tab_review = st.tabs(["Triage a ticket", "Low-confidence review queue"])

with tab_triage:
    ticket_text = st.text_area(
        "Paste the support ticket text",
        height=140,
        placeholder="e.g. I was charged twice this month and need a refund...",
    )
    if st.button("Predict & route", type="primary"):
        if not ticket_text.strip():
            st.warning("Paste some ticket text first.")
        else:
            pred = model.predict(ticket_text)
            was_logged = log_if_low_confidence(pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted category", pred.predicted_category, f"{pred.category_confidence:.0%} confidence")
            col2.metric("Predicted urgency", pred.predicted_urgency, f"{pred.urgency_confidence:.0%} confidence")
            col3.metric("Routed to", pred.queue)

            if pred.needs_human_review:
                st.warning(
                    f"Confidence is below {LOW_CONFIDENCE_THRESHOLD:.0%} on at least one prediction — "
                    + ("logged to the review queue." if was_logged else "")
                )
            else:
                st.success("High confidence — auto-routed with no review needed.")

with tab_review:
    st.subheader("Tickets flagged for human review")
    st.caption(f"Predictions below {LOW_CONFIDENCE_THRESHOLD:.0%} confidence land here instead of auto-routing.")
    queue = read_review_queue()
    if not queue:
        st.info("No low-confidence predictions logged yet.")
    else:
        df = pd.DataFrame(queue).sort_values("logged_at", ascending=False)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download review queue as CSV",
            df.to_csv(index=False).encode("utf-8"),
            "low_confidence_queue.csv",
            "text/csv",
        )
