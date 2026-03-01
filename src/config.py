from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    parquet_dir: Path = Path(__file__).resolve().parent.parent / "data" / "parquet"
    features_dir: Path = Path(__file__).resolve().parent.parent / "data" / "features" / "training"
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "data" / "artifacts"
    sqlite_db_path: Path = Path(__file__).resolve().parent.parent / "data" / "mlmonitoring.db"
    raw_data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw"
    config_dir: Path = Path(__file__).resolve().parent.parent / "config"

    models_dir: Path = Path(__file__).resolve().parent.parent / "data" / "models"
    mlflow_tracking_uri: str = "sqlite:///" + str(Path(__file__).resolve().parent.parent / "data" / "mlflow.db")
    mlflow_artifact_root: str = str(Path(__file__).resolve().parent.parent / "data" / "mlflow-artifacts")
    mlflow_experiment_name: str = "fraud-detection"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    batch_size: int = 5000
    training_cooldown_hours: int = 24
    device: str = "auto"

    # Monitoring settings
    drift_schedule_seconds: int = 3600
    drift_n_recent: int = 5000
    performance_update_seconds: int = 60

    # Retraining settings
    retrain_consecutive_drift_required: int = 2
    retrain_accuracy_drop_threshold: float = 0.05
    stale_model_days: int = 7

    # Auto-rollback settings
    auto_rollback_window_seconds: int = 1800
    auto_rollback_accuracy_drop: float = 0.03
    auto_rollback_latency_p99: float = 2.0

    # Alerting settings
    slack_webhook_url: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_pass: str = ""
    alert_email_to: str = ""

    model_config = {"env_prefix": "MLMON_"}


settings = Settings()
