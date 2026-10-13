import json
from datetime import datetime, timezone

from app.config import PREDICTIONS_LOG_PATH


def log_prediction(prediction: str, spam_probability: float, message_length: int,) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        "spam_probability": round(float(spam_probability), 6),
        "message_length": int(message_length),
    }

    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PREDICTIONS_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


def read_predictions() -> list:
    if not PREDICTIONS_LOG_PATH.exists():
        return []

    records = []
    with PREDICTIONS_LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return records