import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.database import get_connection, init_db
from src.services.baseline_stats import compute_baseline, save_baseline
from src.services.feature_engineering import get_output_feature_names


@pytest.fixture
def monitoring_client(tmp_data_dir, monkeypatch):
    """Test client for monitoring endpoints."""
    from src.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_data_dir["data_dir"])
    monkeypatch.setattr(settings, "parquet_dir", tmp_data_dir["parquet_dir"])
    monkeypatch.setattr(settings, "features_dir", tmp_data_dir["features_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", tmp_data_dir["artifacts_dir"])
    monkeypatch.setattr(settings, "raw_data_dir", tmp_data_dir["raw_data_dir"])
    monkeypatch.setattr(settings, "models_dir", tmp_data_dir["models_dir"])
    monkeypatch.setattr(settings, "sqlite_db_path", tmp_data_dir["db_path"])

    from src.app import app
    init_db(tmp_data_dir["db_path"])

    with TestClient(app) as client:
        yield client


class TestMonitoringRoutes:
    def test_drift_status_empty(self, monitoring_client):
        """GET /monitoring/drift/status returns no_reports when empty."""
        resp = monitoring_client.get("/monitoring/drift/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "no_reports"

    def test_drift_run_no_baseline(self, monitoring_client):
        """POST /monitoring/drift/run without baseline returns error."""
        resp = monitoring_client.post("/monitoring/drift/run")
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_drift_run_with_data(self, monitoring_client, tmp_data_dir):
        """POST /monitoring/drift/run with baseline and data returns report."""
        np.random.seed(42)
        feature_names = get_output_feature_names()
        n_features = len(feature_names)

        # Create baseline
        data = np.random.randn(2000, n_features)
        df = pd.DataFrame(data, columns=feature_names)
        stats = compute_baseline(df)
        save_baseline(stats, 2000, artifacts_dir=tmp_data_dir["artifacts_dir"])

        # Insert predictions with transformed features
        with get_connection(tmp_data_dir["db_path"]) as conn:
            for i in range(200):
                feats = np.random.randn(n_features).tolist()
                conn.execute(
                    """INSERT INTO predictions
                       (ingestion_id, raw_features, transformed_features,
                        prediction_label, prediction_confidence)
                       VALUES (?, '{}', ?, 0, 0.9)""",
                    (f"test-{i}", json.dumps(feats)),
                )

        resp = monitoring_client.post("/monitoring/drift/run")
        assert resp.status_code == 200
        body = resp.json()
        assert "features" in body
        assert "drift_detected" in body

    def test_performance_endpoint(self, monitoring_client, tmp_data_dir):
        """GET /monitoring/performance returns metrics."""
        # Insert some predictions with feedback
        with get_connection(tmp_data_dir["db_path"]) as conn:
            for i in range(50):
                conn.execute(
                    """INSERT INTO predictions
                       (ingestion_id, raw_features, prediction_label,
                        prediction_confidence, actual_label)
                       VALUES (?, '{}', ?, 0.9, ?)""",
                    (f"perf-{i}", 0, 0),
                )

        resp = monitoring_client.get("/monitoring/performance")
        assert resp.status_code == 200
        body = resp.json()
        assert "windows" in body

    def test_retrain_status_endpoint(self, monitoring_client):
        """GET /monitoring/retrain/status returns retrain state."""
        resp = monitoring_client.get("/monitoring/retrain/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "consecutive_drift_count" in body
        assert "consecutive_failures" in body
        assert "auto_trigger_disabled" in body

    def test_rollback_status_endpoint(self, monitoring_client):
        """GET /monitoring/rollback/status returns rollback monitor state."""
        resp = monitoring_client.get("/monitoring/rollback/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "is_active" in body
        assert "window_seconds" in body

    def test_retrain_status_reflects_state(self, monitoring_client):
        """Retrain status should reflect module-level state changes."""
        from src.services.retrain_service import reset_state
        reset_state()
        resp = monitoring_client.get("/monitoring/retrain/status")
        body = resp.json()
        assert body["consecutive_drift_count"] == 0
        assert body["consecutive_failures"] == 0
        assert body["auto_trigger_disabled"] is False

    def test_rollback_status_inactive(self, monitoring_client):
        """Rollback status should show inactive when no monitoring started."""
        from src.services.rollback_monitor import rollback_monitor
        rollback_monitor.stop_monitoring()
        resp = monitoring_client.get("/monitoring/rollback/status")
        body = resp.json()
        assert body["is_active"] is False
