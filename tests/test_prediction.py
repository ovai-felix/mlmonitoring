import pytest
from fastapi.testclient import TestClient

from src.services.model_manager import model_manager


@pytest.fixture
def loaded_client(tmp_data_dir, saved_dummy_models, monkeypatch):
    """Test client with models loaded."""
    from src.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_data_dir["data_dir"])
    monkeypatch.setattr(settings, "parquet_dir", tmp_data_dir["parquet_dir"])
    monkeypatch.setattr(settings, "features_dir", tmp_data_dir["features_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", saved_dummy_models["artifacts_dir"])
    monkeypatch.setattr(settings, "raw_data_dir", tmp_data_dir["raw_data_dir"])
    monkeypatch.setattr(settings, "models_dir", saved_dummy_models["models_dir"])
    monkeypatch.setattr(settings, "sqlite_db_path", tmp_data_dir["db_path"])

    from src.app import app
    from src.database import init_db
    init_db(tmp_data_dir["db_path"])

    # Load models into the singleton
    model_manager.load_slot(model_manager.active,
                            models_dir=saved_dummy_models["models_dir"],
                            artifacts_dir=saved_dummy_models["artifacts_dir"])
    model_manager.warm_up(model_manager.active)

    with TestClient(app) as client:
        yield client

    # Reset manager state after test
    model_manager._blue = type(model_manager._blue)()
    model_manager._green = type(model_manager._green)()
    model_manager._active_color = "blue"


@pytest.fixture
def unloaded_client(tmp_data_dir, monkeypatch):
    """Test client with NO models loaded."""
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

    # Ensure manager is not ready
    model_manager._blue = type(model_manager._blue)()
    model_manager._green = type(model_manager._green)()
    model_manager._active_color = "blue"

    with TestClient(app) as client:
        yield client


@pytest.fixture
def sample_transaction():
    return {
        "Time": 0.0,
        "V1": -1.35, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
        "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
        "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
        "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
        "Amount": 149.62,
    }


def test_predict_503_when_not_ready(unloaded_client, sample_transaction):
    resp = unloaded_client.post("/predict", json=sample_transaction)
    assert resp.status_code == 503


def test_ready_503_when_not_ready(unloaded_client):
    resp = unloaded_client.get("/ready")
    assert resp.status_code == 503


def test_predict_single_success(loaded_client, sample_transaction):
    resp = loaded_client.post("/predict", json=sample_transaction)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction_id" in data
    assert "label" in data
    assert data["label"] in [0, 1]
    assert "confidence" in data
    assert "anomaly_score" in data
    assert "is_anomalous" in data
    assert "model_version" in data
    assert "latency_ms" in data


def test_predict_batch_success(loaded_client, sample_transaction):
    resp = loaded_client.post("/predict/batch", json={"records": [sample_transaction, sample_transaction]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2


def test_predict_with_warnings(loaded_client):
    record = {
        "Time": -999.0,  # out of range → warning
        "V1": -1.35, "V2": -0.07, "V3": 2.54, "V4": 1.38,
        "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
        "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
        "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
        "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
        "Amount": 149.62,
    }
    resp = loaded_client.post("/predict", json=record)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["warnings"]) > 0


def test_model_info_endpoint(loaded_client):
    resp = loaded_client.get("/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_color" in data
    assert "blue" in data
    assert "green" in data


def test_reload_endpoint(loaded_client):
    resp = loaded_client.post("/model/reload")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_rollback_endpoint(loaded_client):
    # First reload so standby becomes ready
    loaded_client.post("/model/reload")
    resp = loaded_client.post("/model/rollback")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_prediction_logged_to_db(loaded_client, sample_transaction):
    resp = loaded_client.post("/predict", json=sample_transaction)
    assert resp.status_code == 200
    prediction_id = resp.json()["prediction_id"]

    from src.database import get_connection
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE ingestion_id = ?",
            (prediction_id,),
        ).fetchone()
    assert row is not None
    assert row["prediction_label"] is not None
