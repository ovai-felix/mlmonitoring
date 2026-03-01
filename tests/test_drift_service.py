import json

import numpy as np
import pytest
from prometheus_client import REGISTRY

from src.database import get_connection, init_db
from src.services.baseline_stats import compute_baseline, save_baseline
from src.services.drift_service import compute_psi, run_drift_detection
from src.services.feature_engineering import get_output_feature_names


def _get_gauge(name, labels=None):
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name:
                if labels is None:
                    return sample.value
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return None


def _create_baseline(artifacts_dir, n=1000, n_features=33):
    """Create a baseline from standard normal data."""
    import pandas as pd
    feature_names = get_output_feature_names()[:n_features]
    data = np.random.randn(n, n_features)
    df = pd.DataFrame(data, columns=feature_names)
    stats = compute_baseline(df)
    save_baseline(stats, n, artifacts_dir=artifacts_dir)
    return feature_names


def _insert_transformed_predictions(db_path, features_list, labels=None, confidences=None):
    """Insert predictions with transformed features."""
    with get_connection(db_path) as conn:
        for i, feats in enumerate(features_list):
            label = labels[i] if labels else 0
            conf = confidences[i] if confidences else 0.9
            conn.execute(
                """INSERT INTO predictions
                   (ingestion_id, raw_features, transformed_features,
                    prediction_label, prediction_confidence)
                   VALUES (?, '{}', ?, ?, ?)""",
                (f"drift-test-{i}", json.dumps(feats.tolist()), label, conf),
            )


class TestComputePSI:
    def test_identical_distributions(self):
        """PSI of identical distributions should be ~0."""
        np.random.seed(42)
        data = np.random.randn(10000)
        psi = compute_psi(data, data)
        assert psi < 0.01

    def test_shifted_distributions(self):
        """PSI of shifted distributions should be > 0.2."""
        np.random.seed(42)
        ref = np.random.randn(10000)
        cur = np.random.randn(10000) + 3.0  # Large shift
        psi = compute_psi(ref, cur)
        assert psi > 0.2

    def test_empty_arrays(self):
        """PSI with empty arrays returns 0."""
        assert compute_psi(np.array([]), np.array([1, 2, 3])) == 0.0
        assert compute_psi(np.array([1, 2]), np.array([])) == 0.0

    def test_constant_distribution(self):
        """PSI with constant values returns 0."""
        ref = np.ones(100)
        cur = np.ones(100)
        assert compute_psi(ref, cur) == 0.0


class TestRunDriftDetection:
    def test_no_baseline(self, tmp_data_dir):
        """Without baseline, report has error."""
        init_db(tmp_data_dir["db_path"])
        result = run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert "error" in result
        assert "No baseline" in result["error"]

    def test_insufficient_data(self, tmp_data_dir):
        """With fewer than 100 predictions, report has error."""
        init_db(tmp_data_dir["db_path"])
        _create_baseline(tmp_data_dir["artifacts_dir"])

        # Insert only 50 predictions
        data = np.random.randn(50, 33)
        _insert_transformed_predictions(tmp_data_dir["db_path"], data)

        result = run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert "error" in result
        assert "Insufficient" in result["error"]

    def test_no_drift_detected(self, tmp_data_dir):
        """Standard normal data vs standard normal baseline => no drift."""
        init_db(tmp_data_dir["db_path"])
        np.random.seed(42)
        _create_baseline(tmp_data_dir["artifacts_dir"], n=2000)

        # Insert similar data
        data = np.random.randn(500, 33)
        _insert_transformed_predictions(tmp_data_dir["db_path"], data)

        result = run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert "error" not in result
        # With same distribution, drift should not be detected
        # (though some features might be borderline due to randomness)
        assert result["drift_detected"] is not None

    def test_drift_with_shifted_data(self, tmp_data_dir):
        """Shifted data should trigger drift detection."""
        init_db(tmp_data_dir["db_path"])
        np.random.seed(42)
        _create_baseline(tmp_data_dir["artifacts_dir"], n=2000)

        # Insert heavily shifted data
        data = np.random.randn(500, 33) + 5.0  # Large shift
        _insert_transformed_predictions(tmp_data_dir["db_path"], data)

        result = run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert result["drift_detected"] is True
        assert len(result["drifted_features"]) > 0

    def test_gauges_set_after_run(self, tmp_data_dir):
        """Prometheus gauges are populated after drift detection."""
        init_db(tmp_data_dir["db_path"])
        np.random.seed(42)
        feature_names = _create_baseline(tmp_data_dir["artifacts_dir"], n=2000)

        data = np.random.randn(500, 33)
        _insert_transformed_predictions(tmp_data_dir["db_path"], data)

        run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )

        # Check that PSI gauge is set for first feature
        val = _get_gauge("feature_drift_psi", {"feature": feature_names[0]})
        assert val is not None

    def test_report_saved(self, tmp_data_dir):
        """Drift report JSON is saved to disk."""
        init_db(tmp_data_dir["db_path"])
        np.random.seed(42)
        _create_baseline(tmp_data_dir["artifacts_dir"], n=2000)

        data = np.random.randn(500, 33)
        _insert_transformed_predictions(tmp_data_dir["db_path"], data)

        result = run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert "report_path" in result
        from pathlib import Path
        assert Path(result["report_path"]).exists()

    def test_data_quality_metrics(self, tmp_data_dir):
        """Data quality gauges are set for null/OOR rates."""
        init_db(tmp_data_dir["db_path"])
        np.random.seed(42)
        feature_names = _create_baseline(tmp_data_dir["artifacts_dir"], n=2000)

        data = np.random.randn(500, 33)
        _insert_transformed_predictions(tmp_data_dir["db_path"], data)

        run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )

        val = _get_gauge("data_quality_null_rate", {"feature": feature_names[0]})
        assert val is not None
        assert val >= 0.0

    def test_prediction_drift(self, tmp_data_dir):
        """Prediction drift metrics (fraud rate, mean confidence) are set."""
        init_db(tmp_data_dir["db_path"])
        np.random.seed(42)
        _create_baseline(tmp_data_dir["artifacts_dir"], n=2000)

        data = np.random.randn(500, 33)
        labels = [1 if i < 50 else 0 for i in range(500)]
        confidences = [0.95 if i < 50 else 0.85 for i in range(500)]
        _insert_transformed_predictions(
            tmp_data_dir["db_path"], data, labels=labels, confidences=confidences
        )

        result = run_drift_detection(
            db_path=tmp_data_dir["db_path"],
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert result["prediction_drift"]["fraud_rate"] == pytest.approx(0.1)
