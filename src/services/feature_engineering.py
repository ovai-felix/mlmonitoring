import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import settings
from src.metrics import feature_transform_latency

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
ORIGINAL_FEATURES = ["Time", "Amount"]
ENGINEERED_FEATURES = ["hour_of_day", "time_since_last", "amount_log"]
ALL_INPUT_FEATURES = PCA_FEATURES + ORIGINAL_FEATURES

PIPELINE_FILENAME = "feature_pipeline.joblib"


def _add_temporal_features(X: np.ndarray) -> np.ndarray:
    """Add engineered features: hour_of_day, time_since_last, amount_log.

    Input columns expected: V1-V28, Time, Amount (30 cols).
    Output columns: V1-V28, Time, Amount, hour_of_day, time_since_last, amount_log (33 cols).
    """
    time_col = X[:, 28]  # Time is at index 28
    amount_col = X[:, 29]  # Amount is at index 29

    hour_of_day = (time_col % 86400) / 3600
    time_since_last = np.concatenate([[0], np.diff(time_col)])
    time_since_last = np.maximum(time_since_last, 0)
    amount_log = np.log1p(np.abs(amount_col))

    return np.column_stack([X, hour_of_day, time_since_last, amount_log])


def get_output_feature_names() -> list[str]:
    return PCA_FEATURES + ORIGINAL_FEATURES + ENGINEERED_FEATURES


def build_pipeline() -> Pipeline:
    """Build the feature engineering pipeline.

    Steps:
    1. Add temporal/amount features (FunctionTransformer)
    2. Scale all numeric features (StandardScaler + SimpleImputer)
    """
    all_output_cols = list(range(len(PCA_FEATURES) + len(ORIGINAL_FEATURES) + len(ENGINEERED_FEATURES)))

    pipeline = Pipeline([
        ("add_features", FunctionTransformer(_add_temporal_features, validate=False)),
        (
            "scale",
            ColumnTransformer(
                [
                    (
                        "numeric",
                        Pipeline([
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]),
                        all_output_cols,
                    )
                ],
                remainder="drop",
            ),
        ),
    ])
    return pipeline


def fit_and_save(
    df: pd.DataFrame,
    artifacts_dir: Path | None = None,
) -> Pipeline:
    """Fit the pipeline on training data and save to disk."""
    artifacts = artifacts_dir or settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    X = df[ALL_INPUT_FEATURES].values
    pipeline = build_pipeline()
    pipeline.fit(X)

    joblib.dump(pipeline, artifacts / PIPELINE_FILENAME)
    return pipeline


def load_pipeline(artifacts_dir: Path | None = None) -> Pipeline:
    """Load a fitted pipeline from disk."""
    artifacts = artifacts_dir or settings.artifacts_dir
    return joblib.load(artifacts / PIPELINE_FILENAME)


def transform(
    df: pd.DataFrame,
    pipeline: Pipeline | None = None,
    artifacts_dir: Path | None = None,
) -> pd.DataFrame:
    """Transform data using the fitted pipeline."""
    if pipeline is None:
        pipeline = load_pipeline(artifacts_dir)

    start = time.time()
    X = df[ALL_INPUT_FEATURES].values
    X_transformed = pipeline.transform(X)
    feature_transform_latency.observe(time.time() - start)

    return pd.DataFrame(X_transformed, columns=get_output_feature_names(), index=df.index)
