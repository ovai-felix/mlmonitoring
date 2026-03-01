import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from src.config import settings
from src.services.feature_store import load_training_features


class FraudDataset(Dataset):
    """PyTorch Dataset for tabular fraud classification."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SequenceDataset(Dataset):
    """Sliding-window dataset for LSTM sequence prediction.

    Each sample is a window of consecutive feature vectors.
    The label is the Class of the transaction immediately following the window.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray, window_size: int = 32):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.labels) - self.window_size)

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window_size]
        y = self.labels[idx + self.window_size]
        return x, y


def load_and_split(
    data_version: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    features_dir: Path | None = None,
    raw_data_dir: Path | None = None,
) -> dict:
    """Load features and labels, split chronologically.

    Features come from the feature store (versioned Parquet).
    Labels come from the raw CSV (Class column), aligned by index.
    """
    features_df = load_training_features(data_version, features_dir=features_dir)

    raw_dir = raw_data_dir or settings.raw_data_dir
    csv_path = raw_dir / "creditcard.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data not found at {csv_path}")

    raw_df = pd.read_csv(csv_path, usecols=["Class"])
    assert len(raw_df) == len(features_df), (
        f"Length mismatch: features={len(features_df)}, raw={len(raw_df)}"
    )

    X = features_df.values.astype(np.float32)
    y = raw_df["Class"].values.astype(np.float32)

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return {
        "X_train": X[:train_end],
        "y_train": y[:train_end],
        "X_val": X[train_end:val_end],
        "y_val": y[train_end:val_end],
        "X_test": X[val_end:],
        "y_test": y[val_end:],
        "feature_names": list(features_df.columns),
        "num_features": X.shape[1],
    }
