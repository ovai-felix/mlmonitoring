import pytest
from prometheus_client import REGISTRY

from src.database import get_connection, init_db
from src.services.performance_metrics import compute_rolling_metrics


def _get_gauge(name, labels=None):
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name == name:
                if labels is None:
                    return sample.value
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
    return None


def _insert_predictions(db_path, predictions):
    """Insert (prediction_label, actual_label) tuples into DB."""
    with get_connection(db_path) as conn:
        for pred_label, actual_label in predictions:
            conn.execute(
                """INSERT INTO predictions
                   (ingestion_id, raw_features, prediction_label,
                    prediction_confidence, actual_label)
                   VALUES (?, '{}', ?, 0.9, ?)""",
                (f"test-{pred_label}-{actual_label}", pred_label, actual_label),
            )


class TestRollingPerformanceMetrics:
    def test_empty_db(self, tmp_data_dir):
        """Empty DB returns empty result."""
        init_db(tmp_data_dir["db_path"])
        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        assert result["windows"] == {}
        assert result["fraud_detection_rate"] is None

    def test_all_correct_predictions(self, tmp_data_dir):
        """100% accuracy when all predictions match actual."""
        init_db(tmp_data_dir["db_path"])
        preds = [(0, 0)] * 50 + [(1, 1)] * 50
        _insert_predictions(tmp_data_dir["db_path"], preds)

        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        w100 = result["windows"][100]
        assert w100["accuracy"] == 1.0
        assert w100["precision"] == 1.0
        assert w100["recall"] == 1.0
        assert w100["f1"] == 1.0
        assert w100["fpr"] == 0.0

    def test_all_wrong_predictions(self, tmp_data_dir):
        """0% accuracy when all predictions are wrong."""
        init_db(tmp_data_dir["db_path"])
        preds = [(1, 0)] * 50 + [(0, 1)] * 50
        _insert_predictions(tmp_data_dir["db_path"], preds)

        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        w100 = result["windows"][100]
        assert w100["accuracy"] == 0.0
        assert w100["precision"] == 0.0
        assert w100["recall"] == 0.0

    def test_mixed_known_values(self, tmp_data_dir):
        """Known confusion matrix yields expected metrics."""
        init_db(tmp_data_dir["db_path"])
        # TP=10, FP=5, TN=80, FN=5 => 100 total
        preds = (
            [(1, 1)] * 10  # TP
            + [(1, 0)] * 5   # FP
            + [(0, 0)] * 80  # TN
            + [(0, 1)] * 5   # FN
        )
        _insert_predictions(tmp_data_dir["db_path"], preds)

        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        w100 = result["windows"][100]
        assert w100["tp"] == 10
        assert w100["fp"] == 5
        assert w100["tn"] == 80
        assert w100["fn"] == 5
        assert w100["accuracy"] == pytest.approx(0.9)
        assert w100["precision"] == pytest.approx(10 / 15)
        assert w100["recall"] == pytest.approx(10 / 15)

    def test_window_sizes(self, tmp_data_dir):
        """Only windows <= data count have entries."""
        init_db(tmp_data_dir["db_path"])
        preds = [(0, 0)] * 50 + [(1, 1)] * 50
        _insert_predictions(tmp_data_dir["db_path"], preds)

        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        assert 100 in result["windows"]
        # window 1000 should also be present (gets all 100 rows)
        assert 1000 in result["windows"]
        assert result["windows"][1000]["count"] == 100

    def test_gauges_set(self, tmp_data_dir):
        """Prometheus gauges are populated after compute."""
        init_db(tmp_data_dir["db_path"])
        preds = [(0, 0)] * 90 + [(1, 1)] * 10
        _insert_predictions(tmp_data_dir["db_path"], preds)

        compute_rolling_metrics(tmp_data_dir["db_path"])
        val = _get_gauge("rolling_accuracy", {"window_size": "100"})
        assert val is not None and val == 1.0

    def test_fraud_detection_rate(self, tmp_data_dir):
        """Fraud detection rate = TP / (TP + FN)."""
        init_db(tmp_data_dir["db_path"])
        # 8 detected out of 10 actual fraud
        preds = [(1, 1)] * 8 + [(0, 1)] * 2 + [(0, 0)] * 90
        _insert_predictions(tmp_data_dir["db_path"], preds)

        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        assert result["fraud_detection_rate"] == pytest.approx(0.8)

    def test_false_positive_rate(self, tmp_data_dir):
        """FPR = FP / (FP + TN)."""
        init_db(tmp_data_dir["db_path"])
        # FP=10, TN=90 => FPR = 10/100 = 0.1
        preds = [(1, 0)] * 10 + [(0, 0)] * 90
        _insert_predictions(tmp_data_dir["db_path"], preds)

        result = compute_rolling_metrics(tmp_data_dir["db_path"])
        w100 = result["windows"][100]
        assert w100["fpr"] == pytest.approx(0.1)
