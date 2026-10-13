from collections import Counter
from app.monitoring.predictions import read_predictions


def get_monitoring_metrics() -> dict:
    records = read_predictions()

    counts = Counter(
        record.get("prediction", "unknown")
        for record in records
    )

    probabilities = [
        float(record["spam_probability"])
        for record in records
        if isinstance(record.get("spam_probability"), (int, float))
    ]

    total = len(records)
    spam_count = counts.get("spam", 0)

    return {
        "total_predictions": total,
        "spam_predictions": spam_count,
        "ham_predictions": counts.get("ham", 0),
        "spam_rate": spam_count / total if total else 0.0,
        "average_spam_probability": (
            sum(probabilities) / len(probabilities)
            if probabilities else 0.0
        ),
        "recent_predictions": records[-20:],
    }


def get_drift_status() -> dict:
    """
    Basic operational alert, not statistical data-drift detection.
    Uses a simple recent-window spam-rate comparison.
    """
    records = read_predictions()
    recent = records[-100:]
    previous = records[-200:-100]

    def spam_rate(items):
        if not items:
            return None
        return sum(
            item.get("prediction") == "spam" for item in items
        ) / len(items)

    recent_rate = spam_rate(recent)
    previous_rate = spam_rate(previous)

    if recent_rate is None or previous_rate is None:
        return {
            "status": "insufficient_data",
            "recent_spam_rate": recent_rate,
            "previous_spam_rate": previous_rate,
            "message": "Need more predictions to compare windows.",
        }

    difference = abs(recent_rate - previous_rate)

    return {
        "status": "rate_change_alert" if difference >= 0.20 else "normal",
        "recent_spam_rate": recent_rate,
        "previous_spam_rate": previous_rate,
        "absolute_difference": difference,
        "message": (
            "Spam prediction rate changed substantially."
            if difference >= 0.20
            else "No large spam-rate change detected."
        ),
    }