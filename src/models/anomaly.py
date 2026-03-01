import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Isolation Forest wrapper for anomaly detection on transaction data."""

    def __init__(
        self,
        contamination: float = 0.002,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_normal: np.ndarray):
        """Train on normal-class data only."""
        self.model.fit(X_normal)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores. Lower = more anomalous."""
        return self.model.decision_function(X)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for normal, -1 for anomaly."""
        return self.model.predict(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "contamination": self.contamination}, path)

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        data = joblib.load(path)
        detector = cls.__new__(cls)
        detector.model = data["model"]
        detector.contamination = data["contamination"]
        return detector
