import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import optuna

from src.config import settings
from src.database import (
    get_connection,
    insert_training_run,
    update_training_run,
    get_last_training_run,
)
from src.models.anomaly import AnomalyDetector
from src.models.classifier import TabularTransformer
from src.models.lstm_model import FraudLSTM
from src.services.data_loader import FraudDataset, SequenceDataset, load_and_split
from src.services.evaluation_service import (
    check_promotion_gate,
    evaluate_anomaly,
    evaluate_classification,
)
from src.services.mlflow_service import get_or_create_experiment, setup_mlflow

logger = logging.getLogger(__name__)

# Suppress Optuna trial-level logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_device() -> torch.device:
    if settings.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings.device)


def check_cooldown(model_type: str, db_path: Path | None = None) -> bool:
    """Return True if training is allowed (cooldown expired)."""
    with get_connection(db_path) as conn:
        last_run = get_last_training_run(conn, model_type)
        if last_run is None:
            return True
        last_started = datetime.fromisoformat(last_run["started_at"])
        if last_started.tzinfo is None:
            last_started = last_started.replace(tzinfo=timezone.utc)
        cooldown = timedelta(hours=settings.training_cooldown_hours)
        return datetime.now(timezone.utc) - last_started > cooldown


def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    n_positive = y.sum()
    n_negative = len(y) - n_positive
    pos_weight = n_negative / max(n_positive, 1)
    return torch.tensor([pos_weight], dtype=torch.float32)


def train_classifier(
    data_version: str,
    n_trials: int = 5,
    db_path: Path | None = None,
) -> dict:
    """Train the Transformer classifier with Optuna hyperparameter search."""
    run_id = str(uuid.uuid4())
    device = get_device()

    if not check_cooldown("classification", db_path):
        return {"status": "skipped", "reason": "cooldown"}

    with get_connection(db_path) as conn:
        insert_training_run(conn, run_id, "classification", data_version)

    try:
        data = load_and_split(data_version)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        num_features = data["num_features"]

        pos_weight = _compute_pos_weight(y_train).to(device)

        setup_mlflow()
        experiment_id = get_or_create_experiment()

        def objective(trial):
            d_model = trial.suggest_categorical("d_model", [64, 128])
            nhead = trial.suggest_categorical("nhead", [2, 4])
            num_layers = trial.suggest_int("num_layers", 2, 4)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.05, 0.3)
            batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

            model = TabularTransformer(
                num_features=num_features,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=d_model * 2,
                dropout=dropout,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            train_loader = DataLoader(
                FraudDataset(X_train, y_train), batch_size=batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                FraudDataset(X_val, y_val), batch_size=batch_size,
            )

            best_val_f1 = 0.0
            for epoch in range(20):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    logits = model(X_batch).squeeze(-1)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                metrics = evaluate_classification(model, val_loader, device)
                best_val_f1 = max(best_val_f1, metrics["f1"])

                trial.report(metrics["f1"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_f1

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params

        # Retrain final model with best params
        with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
            mlflow.set_tag("model_type", "classification")
            mlflow.log_params(best_params)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("device", str(device))

            final_model = TabularTransformer(
                num_features=num_features,
                d_model=best_params["d_model"],
                nhead=best_params["nhead"],
                num_layers=best_params["num_layers"],
                dim_feedforward=best_params["d_model"] * 2,
                dropout=best_params["dropout"],
            ).to(device)

            optimizer = torch.optim.AdamW(
                final_model.parameters(), lr=best_params["lr"], weight_decay=1e-4,
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            train_loader = DataLoader(
                FraudDataset(X_train, y_train),
                batch_size=best_params["batch_size"], shuffle=True,
            )

            for epoch in range(30):
                final_model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    logits = final_model(X_batch).squeeze(-1)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

            # Evaluate on test
            test_loader = DataLoader(FraudDataset(X_test, y_test), batch_size=512)
            test_metrics = evaluate_classification(final_model, test_loader, device)
            mlflow.log_metrics(test_metrics)

            # Save model locally
            model_dir = settings.models_dir / "classifier"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(final_model.state_dict(), model_dir / "model.pt")
            torch.save({
                "model_config": {
                    "num_features": num_features,
                    "d_model": best_params["d_model"],
                    "nhead": best_params["nhead"],
                    "num_layers": best_params["num_layers"],
                    "dim_feedforward": best_params["d_model"] * 2,
                    "dropout": best_params["dropout"],
                },
                "class_weights": pos_weight.cpu(),
            }, model_dir / "config.pt")

            # Log model to MLflow and register
            model_info = mlflow.pytorch.log_model(
                final_model.cpu(), artifact_path="model",
                registered_model_name="fraud-classifier",
            )
            result = mlflow.MlflowClient().get_latest_versions("fraud-classifier")[0]

            gate_result = check_promotion_gate(
                "classification", test_metrics, "fraud-classifier", result.version,
            )

        with get_connection(db_path) as conn:
            update_training_run(
                conn, run_id, status="completed", metrics=test_metrics,
                mlflow_run_id=mlflow_run.info.run_id,
                promoted_to=gate_result.get("promoted_to"),
            )

        return {
            "status": "completed",
            "run_id": run_id,
            "mlflow_run_id": mlflow_run.info.run_id,
            "metrics": test_metrics,
            "best_params": best_params,
            "gate_result": gate_result,
        }

    except Exception as e:
        logger.exception("Classification training failed")
        with get_connection(db_path) as conn:
            update_training_run(conn, run_id, status="failed", error_message=str(e))
        raise


def train_lstm(
    data_version: str,
    window_size: int = 32,
    n_trials: int = 5,
    db_path: Path | None = None,
) -> dict:
    """Train the LSTM model with Optuna hyperparameter search."""
    run_id = str(uuid.uuid4())
    device = get_device()

    if not check_cooldown("timeseries", db_path):
        return {"status": "skipped", "reason": "cooldown"}

    with get_connection(db_path) as conn:
        insert_training_run(conn, run_id, "timeseries", data_version)

    try:
        data = load_and_split(data_version)
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        num_features = data["num_features"]

        pos_weight = _compute_pos_weight(y_train).to(device)

        setup_mlflow()
        experiment_id = get_or_create_experiment()

        def objective(trial):
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128])
            num_layers = trial.suggest_int("num_layers", 1, 3)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.4)
            batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

            model = FraudLSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            train_loader = DataLoader(
                SequenceDataset(X_train, y_train, window_size), batch_size=batch_size, shuffle=True,
            )
            val_loader = DataLoader(
                SequenceDataset(X_val, y_val, window_size), batch_size=batch_size,
            )

            best_val_f1 = 0.0
            for epoch in range(15):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    logits = model(X_batch).squeeze(-1)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()

                metrics = evaluate_classification(model, val_loader, device)
                best_val_f1 = max(best_val_f1, metrics["f1"])

                trial.report(metrics["f1"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_f1

        study = optuna.create_study(
            direction="maximize", pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_trial.params

        with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
            mlflow.set_tag("model_type", "timeseries")
            mlflow.log_params(best_params)
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("window_size", window_size)

            final_model = FraudLSTM(
                input_size=num_features,
                hidden_size=best_params["hidden_size"],
                num_layers=best_params["num_layers"],
                dropout=best_params["dropout"],
            ).to(device)

            optimizer = torch.optim.AdamW(
                final_model.parameters(), lr=best_params["lr"], weight_decay=1e-4,
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            train_loader = DataLoader(
                SequenceDataset(X_train, y_train, window_size),
                batch_size=best_params["batch_size"], shuffle=True,
            )

            for epoch in range(20):
                final_model.train()
                epoch_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    logits = final_model(X_batch).squeeze(-1)
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                mlflow.log_metric("train_loss", epoch_loss / len(train_loader), step=epoch)

            test_loader = DataLoader(
                SequenceDataset(X_test, y_test, window_size), batch_size=512,
            )
            test_metrics = evaluate_classification(final_model, test_loader, device)
            mlflow.log_metrics(test_metrics)

            model_dir = settings.models_dir / "lstm"
            model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(final_model.state_dict(), model_dir / "model.pt")
            torch.save({
                "model_config": {
                    "input_size": num_features,
                    "hidden_size": best_params["hidden_size"],
                    "num_layers": best_params["num_layers"],
                    "dropout": best_params["dropout"],
                },
                "window_size": window_size,
                "class_weights": pos_weight.cpu(),
            }, model_dir / "config.pt")

            model_info = mlflow.pytorch.log_model(
                final_model.cpu(), artifact_path="model",
                registered_model_name="fraud-lstm",
            )
            result = mlflow.MlflowClient().get_latest_versions("fraud-lstm")[0]

            gate_result = check_promotion_gate(
                "timeseries", test_metrics, "fraud-lstm", result.version,
            )

        with get_connection(db_path) as conn:
            update_training_run(
                conn, run_id, status="completed", metrics=test_metrics,
                mlflow_run_id=mlflow_run.info.run_id,
                promoted_to=gate_result.get("promoted_to"),
            )

        return {
            "status": "completed",
            "run_id": run_id,
            "mlflow_run_id": mlflow_run.info.run_id,
            "metrics": test_metrics,
            "best_params": best_params,
            "gate_result": gate_result,
        }

    except Exception as e:
        logger.exception("LSTM training failed")
        with get_connection(db_path) as conn:
            update_training_run(conn, run_id, status="failed", error_message=str(e))
        raise


def train_anomaly(
    data_version: str,
    db_path: Path | None = None,
) -> dict:
    """Train the Isolation Forest anomaly detector."""
    run_id = str(uuid.uuid4())

    if not check_cooldown("anomaly", db_path):
        return {"status": "skipped", "reason": "cooldown"}

    with get_connection(db_path) as conn:
        insert_training_run(conn, run_id, "anomaly", data_version)

    try:
        data = load_and_split(data_version)
        X_train_normal = data["X_train"][data["y_train"] == 0]

        setup_mlflow()
        experiment_id = get_or_create_experiment()

        with mlflow.start_run(experiment_id=experiment_id) as mlflow_run:
            mlflow.set_tag("model_type", "anomaly")
            mlflow.log_param("data_version", data_version)
            mlflow.log_param("contamination", 0.002)
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("n_train_normal", len(X_train_normal))

            detector = AnomalyDetector(contamination=0.002, n_estimators=200)
            detector.fit(X_train_normal)

            test_metrics = evaluate_anomaly(detector, data["X_test"], data["y_test"])
            mlflow.log_metrics(test_metrics)

            model_dir = settings.models_dir / "anomaly"
            model_dir.mkdir(parents=True, exist_ok=True)
            detector.save(model_dir / "model.joblib")

            model_info = mlflow.sklearn.log_model(
                detector.model, artifact_path="model",
                registered_model_name="fraud-anomaly",
            )
            result = mlflow.MlflowClient().get_latest_versions("fraud-anomaly")[0]

            gate_result = check_promotion_gate(
                "anomaly", test_metrics, "fraud-anomaly", result.version,
            )

        with get_connection(db_path) as conn:
            update_training_run(
                conn, run_id, status="completed", metrics=test_metrics,
                mlflow_run_id=mlflow_run.info.run_id,
                promoted_to=gate_result.get("promoted_to"),
            )

        return {
            "status": "completed",
            "run_id": run_id,
            "mlflow_run_id": mlflow_run.info.run_id,
            "metrics": test_metrics,
            "gate_result": gate_result,
        }

    except Exception as e:
        logger.exception("Anomaly training failed")
        with get_connection(db_path) as conn:
            update_training_run(conn, run_id, status="failed", error_message=str(e))
        raise
