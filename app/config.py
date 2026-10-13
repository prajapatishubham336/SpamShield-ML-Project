
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "spam_pipeline.joblib"

TRAINING_METRICS_PATH = LOGS_DIR / "training_metrics.json"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions.jsonl"


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(BASE_DIR / 'mlflow.db').as_posix()}",
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "SpamShield-ML",
)


SPAM_THRESHOLD = float(os.getenv("SPAM_THRESHOLD", "0.5"))
MAX_TEXT_LENGTH = 10000
MAX_BATCH_SIZE = 100

for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, ARTIFACT_DIR,):
    directory.mkdir(parents=True, exist_ok=True)