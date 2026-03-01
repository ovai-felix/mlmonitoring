from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config import settings


def setup_mlflow():
    """Configure MLflow tracking URI and create artifact directory."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    Path(settings.mlflow_artifact_root).mkdir(parents=True, exist_ok=True)


def get_or_create_experiment(name: str | None = None) -> str:
    """Get or create an MLflow experiment. Returns experiment_id."""
    name = name or settings.mlflow_experiment_name
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(
            name,
            artifact_location=str(Path(settings.mlflow_artifact_root) / name),
        )
    return experiment.experiment_id


def get_production_model_metrics(model_name: str) -> dict | None:
    """Get metrics from the current Production model, if any."""
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            return None
        version = latest_versions[0]
        run = client.get_run(version.run_id)
        return dict(run.data.metrics)
    except Exception:
        return None


def promote_model(model_name: str, version: int, stage: str = "Staging"):
    """Promote a model version to a registry stage."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage,
    )
