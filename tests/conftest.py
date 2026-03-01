import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create temporary data directories."""
    dirs = {
        "data_dir": tmp_path / "data",
        "parquet_dir": tmp_path / "data" / "parquet",
        "features_dir": tmp_path / "data" / "features" / "training",
        "artifacts_dir": tmp_path / "data" / "artifacts",
        "raw_data_dir": tmp_path / "data" / "raw",
        "models_dir": tmp_path / "data" / "models",
        "db_path": tmp_path / "data" / "test.db",
    }
    for key, d in dirs.items():
        if key != "db_path":
            d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def sample_record():
    """A valid transaction record dict."""
    return {
        "Time": 0.0,
        "V1": -1.35,
        "V2": -0.07,
        "V3": 2.54,
        "V4": 1.38,
        "V5": -0.34,
        "V6": 0.46,
        "V7": 0.24,
        "V8": 0.10,
        "V9": 0.36,
        "V10": 0.09,
        "V11": -0.55,
        "V12": -0.62,
        "V13": -0.99,
        "V14": -0.31,
        "V15": 1.47,
        "V16": -0.47,
        "V17": 0.21,
        "V18": 0.03,
        "V19": 0.40,
        "V20": 0.25,
        "V21": -0.02,
        "V22": 0.28,
        "V23": -0.11,
        "V24": 0.07,
        "V25": 0.13,
        "V26": -0.19,
        "V27": 0.13,
        "V28": -0.02,
        "Amount": 149.62,
        "Class": 0,
    }


@pytest.fixture
def sample_records(sample_record):
    """A list of 10 sample records with slight variations."""
    import copy
    records = []
    for i in range(10):
        r = copy.deepcopy(sample_record)
        r["Time"] = float(i * 100)
        r["Amount"] = 50.0 + i * 10
        r["V1"] = -1.35 + i * 0.1
        records.append(r)
    return records


@pytest.fixture
def test_client(tmp_data_dir, monkeypatch):
    """FastAPI test client with temporary directories."""
    from src.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_data_dir["data_dir"])
    monkeypatch.setattr(settings, "parquet_dir", tmp_data_dir["parquet_dir"])
    monkeypatch.setattr(settings, "features_dir", tmp_data_dir["features_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", tmp_data_dir["artifacts_dir"])
    monkeypatch.setattr(settings, "raw_data_dir", tmp_data_dir["raw_data_dir"])
    monkeypatch.setattr(settings, "models_dir", tmp_data_dir["models_dir"])
    monkeypatch.setattr(settings, "sqlite_db_path", tmp_data_dir["db_path"])

    from src.app import app
    from src.database import init_db
    init_db(tmp_data_dir["db_path"])

    with TestClient(app) as client:
        yield client


@pytest.fixture
def saved_dummy_models(tmp_data_dir):
    """Create minimal model files on disk for testing."""
    import torch
    import numpy as np
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from src.models.classifier import TabularTransformer
    from src.models.lstm_model import FraudLSTM
    from src.models.anomaly import AnomalyDetector

    models_dir = tmp_data_dir["models_dir"]
    artifacts_dir = tmp_data_dir["artifacts_dir"]

    # Save classifier
    clf_dir = models_dir / "classifier"
    clf_dir.mkdir(parents=True, exist_ok=True)
    clf_config = {"num_features": 33, "d_model": 16, "nhead": 2, "num_layers": 1, "dim_feedforward": 32, "dropout": 0.0}
    clf = TabularTransformer(**clf_config)
    torch.save(clf_config, clf_dir / "config.pt")
    torch.save(clf.state_dict(), clf_dir / "model.pt")

    # Save LSTM
    lstm_dir = models_dir / "lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)
    lstm_config = {"input_size": 33, "hidden_size": 16, "num_layers": 1, "dropout": 0.0}
    lstm = FraudLSTM(**lstm_config)
    torch.save(lstm_config, lstm_dir / "config.pt")
    torch.save(lstm.state_dict(), lstm_dir / "model.pt")

    # Save anomaly detector
    anomaly_dir = models_dir / "anomaly"
    anomaly_dir.mkdir(parents=True, exist_ok=True)
    ad = AnomalyDetector(n_estimators=10)
    X_normal = np.random.randn(100, 33)
    ad.fit(X_normal)
    ad.save(anomaly_dir / "model.joblib")

    # Save a fitted feature pipeline
    from src.services.feature_engineering import build_pipeline, ALL_INPUT_FEATURES, PIPELINE_FILENAME
    pipeline = build_pipeline()
    X_fit = np.random.randn(50, len(ALL_INPUT_FEATURES))
    # Set Time and Amount to reasonable values
    X_fit[:, 28] = np.abs(X_fit[:, 28]) * 1000  # Time
    X_fit[:, 29] = np.abs(X_fit[:, 29]) * 100    # Amount
    pipeline.fit(X_fit)
    joblib.dump(pipeline, artifacts_dir / PIPELINE_FILENAME)

    return {"models_dir": models_dir, "artifacts_dir": artifacts_dir}
