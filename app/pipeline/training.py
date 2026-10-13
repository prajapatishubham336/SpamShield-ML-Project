import json
import logging
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from app.config import MODEL_PATH, TRAINING_METRICS_PATH
from app.mlops.mlflow_utils import log_training_run
from app.pipeline.models import build_models
from app.pipeline.preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)


def train_and_save(
    csv_path=None,
    text_column: str = "text",
    label_column: str = "label",
) -> dict:
    """Train candidate models, compare them, and persist the winner."""
    if csv_path is None:
        from app.config import RAW_DATA_DIR
        csv_path = RAW_DATA_DIR / "spam.csv"

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="latin-1")

    clean_df = preprocess_dataframe(df, text_column=text_column, label_column=label_column,)

    X = clean_df["text"]
    y = clean_df["label"]

    if y.value_counts().min() < 2:
        raise ValueError("At least two examples per class are required for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = {}
    trained_models = {}

    for name, model in build_models().items():
        logger.info("Training model: %s", name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_spam": float(
                precision_score(y_test, predictions, pos_label="spam",
                                zero_division=0)
            ),
            "recall_spam": float(
                recall_score(y_test, predictions, pos_label="spam",
                             zero_division=0)
            ),
            "f1_spam": float(
                f1_score(y_test, predictions, pos_label="spam",
                         zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_test, predictions, average="macro",
                         zero_division=0)
            ),
        }

        results[name] = metrics
        trained_models[name] = model
        log_training_run(name, metrics, {"test_size": 0.2, "random_state": 42})

    # Prioritize spam F1; use spam precision as a tie-breaker.
    # Select using macro F1, then spam F1 and precision.
    best_name = max(
        results,
        key=lambda name: (
            results[name]["macro_f1"],
            results[name]["f1_spam"],
            results[name]["precision_spam"],
            results[name]["accuracy"],
        ),
    )
    best_model = trained_models[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(clean_df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "best_model": best_name,
        "best_metrics": results[best_name],
        "all_models": results,
        "classes": sorted(y.unique().tolist()),
        "classification_report": classification_report(
            y_test,
            best_model.predict(X_test),
            output_dict=True,
            zero_division=0,
        ),
    }

    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_METRICS_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    logger.info("Best model saved: %s", MODEL_PATH)
    return report


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2))