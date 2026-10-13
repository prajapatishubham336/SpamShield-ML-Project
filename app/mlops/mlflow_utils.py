import logging
from app.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def log_training_run(model_name: str, metrics: dict, params: dict,) -> None:
    """Log model parameters and metrics to local MLflow."""

    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=model_name,):
            mlflow.log_param("model_name", model_name)
            for key, value in params.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))

        logger.info("MLflow run logged successfully for model: %s", model_name,)

    except Exception:
        # Experiment tracking should not interrupt model training.
        logger.exception(
            "MLflow logging failed; continuing without experiment tracking."
        )
