import numpy as np
import pytest

from src.models.anomaly import AnomalyDetector


class TestAnomalyDetector:
    def test_fit_predict(self):
        rng = np.random.default_rng(42)
        X_normal = rng.standard_normal((100, 33))
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(X_normal)
        scores = detector.predict(X_normal)
        assert scores.shape == (100,)

    def test_labels(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 33))
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(X)
        labels = detector.predict_labels(X)
        assert set(np.unique(labels)).issubset({-1, 1})

    def test_save_load(self, tmp_data_dir):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 33))
        detector = AnomalyDetector()
        detector.fit(X)
        path = tmp_data_dir["artifacts_dir"] / "anomaly.joblib"
        detector.save(path)
        loaded = AnomalyDetector.load(path)
        np.testing.assert_array_almost_equal(
            detector.predict(X), loaded.predict(X),
        )
