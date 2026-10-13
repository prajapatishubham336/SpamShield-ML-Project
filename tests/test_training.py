import pandas as pd

from app.pipeline.training import train_and_save


def test_training_creates_model_and_report(tmp_path, monkeypatch):
    import app.pipeline.training as training
    import app.config as config

    csv_path = tmp_path / "spam.csv"

    rows = []
    for i in range(30):
        rows.append({
            "text": f"Claim your free cash prize offer {i}",
            "label": "spam",
        })
        rows.append({
            "text": f"Meeting notes and project discussion {i}",
            "label": "ham",
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    model_path = tmp_path / "spam_pipeline.joblib"
    metrics_path = tmp_path / "training_metrics.json"

    monkeypatch.setattr(training, "MODEL_PATH", model_path)
    monkeypatch.setattr(training, "TRAINING_METRICS_PATH", metrics_path)

    # Prevent this test from writing into the real MLflow experiment store.
    monkeypatch.setattr(
        training,
        "log_training_run",
        lambda *args, **kwargs: None,
    )

    report = training.train_and_save(csv_path)

    assert model_path.exists()
    assert metrics_path.exists()
    assert report["best_model"] in {
        "logistic_regression",
        "multinomial_nb",
        "complement_nb",
    }