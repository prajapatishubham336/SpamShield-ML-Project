import json
import pandas as pd
import streamlit as st

from app.config import TRAINING_METRICS_PATH
from app.monitoring.metrics import (get_drift_status, get_monitoring_metrics,)

st.set_page_config(
    page_title="SpamShield Admin",
    page_icon="🛡️",
    layout="wide",
)

st.title("SpamShield ML — MLOps Dashboard")
st.caption("Training evaluation, experiment results and prediction monitoring")

monitoring = get_monitoring_metrics()
drift = get_drift_status()

col1, col2, col3 = st.columns(3)
col1.metric("Total predictions", monitoring["total_predictions"])
col2.metric("Spam predictions", monitoring["spam_predictions"])
col3.metric("Ham predictions", monitoring["ham_predictions"])

st.subheader("Prediction monitoring")
st.json({
    "spam_rate": monitoring["spam_rate"],
    "average_spam_probability": monitoring["average_spam_probability"],
    "rate_change_status": drift["status"],
})

st.subheader("Recent predictions")
recent = monitoring["recent_predictions"]

if recent:
    st.dataframe(pd.DataFrame(recent), use_container_width=True)
else:
    st.info("No prediction records yet. Make predictions through the API first.")

st.subheader("Training evaluation")

if TRAINING_METRICS_PATH.exists():
    report = json.loads(TRAINING_METRICS_PATH.read_text(encoding="utf-8"))

    st.write("Selected model:", report.get("best_model", "Unknown"))
    st.json(report.get("best_metrics", {}))

    all_models = report.get("all_models", {})
    if all_models:
        st.dataframe(
            pd.DataFrame(all_models).T,
            use_container_width=True,
        )
else:
    st.info("No training report found. Train the model first.")

st.warning(
    "This dashboard has no authentication. Keep it local or protect it "
    "before exposing it to other users."
)