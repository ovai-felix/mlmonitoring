import numpy as np
import pandas as pd
import pytest

from src.services.feature_engineering import (
    ALL_INPUT_FEATURES,
    build_pipeline,
    fit_and_save,
    get_output_feature_names,
    load_pipeline,
    transform,
)


def _make_df(n=100):
    """Create a synthetic DataFrame matching expected input schema."""
    rng = np.random.default_rng(42)
    data = {}
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)
    data["Time"] = np.arange(n, dtype=float) * 100
    data["Amount"] = rng.exponential(50, n)
    data["Class"] = rng.choice([0, 1], n, p=[0.99, 0.01])
    return pd.DataFrame(data)


class TestPipeline:
    def test_build_pipeline(self):
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_fit_transform(self):
        df = _make_df(100)
        pipeline = build_pipeline()
        X = df[ALL_INPUT_FEATURES].values
        pipeline.fit(X)
        X_out = pipeline.transform(X)
        expected_cols = len(get_output_feature_names())
        assert X_out.shape == (100, expected_cols)

    def test_output_feature_names(self):
        names = get_output_feature_names()
        assert "V1" in names
        assert "Amount" in names
        assert "hour_of_day" in names
        assert "time_since_last" in names
        assert "amount_log" in names
        assert len(names) == 33  # 28 PCA + Time + Amount + 3 engineered

    def test_fit_and_save_load(self, tmp_data_dir):
        df = _make_df(100)
        pipeline = fit_and_save(df, artifacts_dir=tmp_data_dir["artifacts_dir"])
        assert (tmp_data_dir["artifacts_dir"] / "feature_pipeline.joblib").exists()

        loaded = load_pipeline(artifacts_dir=tmp_data_dir["artifacts_dir"])
        X = df[ALL_INPUT_FEATURES].values
        X_orig = pipeline.transform(X)
        X_loaded = loaded.transform(X)
        np.testing.assert_array_almost_equal(X_orig, X_loaded)

    def test_transform_function(self, tmp_data_dir):
        df = _make_df(50)
        fit_and_save(df, artifacts_dir=tmp_data_dir["artifacts_dir"])
        result = transform(df, artifacts_dir=tmp_data_dir["artifacts_dir"])
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == get_output_feature_names()
        assert len(result) == 50

    def test_scaled_output_near_zero_mean(self, tmp_data_dir):
        df = _make_df(500)
        pipeline = fit_and_save(df, artifacts_dir=tmp_data_dir["artifacts_dir"])
        result = transform(df, pipeline=pipeline)
        # After StandardScaler, means should be near 0
        means = result.mean()
        for col in result.columns:
            assert abs(means[col]) < 0.5, f"{col} mean={means[col]}"
