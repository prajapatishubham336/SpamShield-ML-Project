
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from app.config import (
    MODEL_PATH,
    SPAM_THRESHOLD,
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
    PREDICTIONS_LOG_PATH,
)
from app.logging_config import setup_logging

# PATHS AND LOGGING
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FILE = BASE_DIR / "app" / "templates" / "index.html"

setup_logging()
logger = logging.getLogger(__name__)

MODEL = None


# MODEL LOADING
def load_model():
    """Load the trained sklearn pipeline."""

    global MODEL

    model_path = Path(MODEL_PATH)

    if not model_path.is_file():
        logger.warning("Model file not found: %s", model_path)
        MODEL = None
        return

    try:
        MODEL = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)

    except Exception:
        logger.exception("Unable to load the trained model.")
        MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("SpamShield application startup complete.")
    yield
    logger.info("SpamShield application shutdown complete.")


app = FastAPI(
    title="SpamShield ML",
    description="Spam detection using machine learning, NLP and MLOps.",
    version="1.0.0",
    lifespan=lifespan,
)


# REQUEST SCHEMAS
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


# INPUT VALIDATION
def validate_text(text: str) -> str:
    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Please enter a message.",
        )

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum message length is {MAX_TEXT_LENGTH} characters.",
        )

    return text


def ensure_model_loaded():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained or loaded. Please train the model first.",
        )

# PREDICTION LOGGING
def log_prediction(text: str, result: dict) -> None:
    """Log prediction metadata without storing message contents."""

    try:
        log_path = Path(PREDICTIONS_LOG_PATH)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text_length": len(text),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
        }

        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    except Exception:
        # Logging errors should not interrupt predictions.
        logger.exception("Failed to write prediction log.")


# MODEL INFERENCE
def get_prediction(text: str) -> dict:
    ensure_model_loaded()

    try:
        input_data = [text]

        prediction = str(MODEL.predict(input_data)[0]).strip().lower()

        confidence: Optional[float] = None
        spam_probability: Optional[float] = None

        if hasattr(MODEL, "predict_proba"):
            probabilities = MODEL.predict_proba(input_data)[0]
            classes = [
                str(label).strip().lower()
                for label in MODEL.classes_
            ]

            if "spam" in classes:
                spam_probability = float(
                    probabilities[classes.index("spam")]
                )

                if spam_probability >= SPAM_THRESHOLD:
                    prediction = "spam"
                    confidence = spam_probability
                else:
                    prediction = "ham"
                    confidence = 1.0 - spam_probability

            elif prediction in classes:
                confidence = float(
                    probabilities[classes.index(prediction)]
                )

        result = {
            "label": prediction,
            "prediction": prediction,
            "is_spam": prediction == "spam",
            "confidence": confidence,
            "spam_probability": spam_probability,
            "message": "Prediction completed successfully.",
        }

        log_prediction(text, result)
        return result

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Verify that the saved model "
                "accepts a DataFrame with a 'text' column."
            ),
        ) from exc


# DASHBOARD
@app.get("/", include_in_schema=False)
async def home():
    """Serve the dashboard HTML."""

    if not INDEX_FILE.is_file():
        logger.error("Dashboard file not found: %s", INDEX_FILE)

        raise HTTPException(
            status_code=500,
            detail=f"Dashboard file not found: {INDEX_FILE}",
        )

    return FileResponse(
        path=str(INDEX_FILE),
        media_type="text/html",
        headers={"Cache-Control": "no-cache"},
    )


# FAVICON
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# HEALTH CHECKS
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
    }


@app.get("/health", include_in_schema=False)
async def health_alias():
    return await health()

# SINGLE MESSAGE PREDICTION
@app.post("/predict")
async def predict(request: PredictRequest):
    text = validate_text(request.text)
    return get_prediction(text)


@app.post("/api/predict")
async def api_predict(request: PredictRequest):
    text = validate_text(request.text)
    return get_prediction(text)

# BATCH PREDICTION
@app.post("/predict/batch")
async def predict_batch(request: BatchPredictRequest):
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Maximum batch size is {MAX_BATCH_SIZE}.",
        )

    results = []

    for original_text in request.texts:
        text = validate_text(original_text)
        results.append(get_prediction(text))

    return {
        "count": len(results),
        "results": results,
    }


# MODEL INFORMATION
@app.get("/api/model-info")
async def model_info():
    return {
        "project": "SpamShield ML",
        "model_loaded": MODEL is not None,
        "model_type": type(MODEL).__name__ if MODEL is not None else None,
        "spam_threshold": SPAM_THRESHOLD,
    }
