import numpy as np
import pandas as pd
import pytest

from src.services.baseline_stats import (
    compute_baseline,
    expose_baseline_to_prometheus,
    load_baseline,
    save_baseline,
)


def _make_df():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "feat_a": rng.normal(10.0, 2.0, 1000),
        "feat_b": rng.normal(0.0, 1.0, 1000),
        "feat_c": rng.exponential(5.0, 1000),
    })


class TestBaseline:
    def test_compute_baseline(self):
        df = _make_df()
        stats = compute_baseline(df)
        assert len(stats) == 3
        feat_a = next(s for s in stats if s.feature_name == "feat_a")
        assert 9.0 < feat_a.mean < 11.0
        assert 1.5 < feat_a.std < 2.5
        assert feat_a.null_rate == 0.0
        assert len(feat_a.histogram_bins) == 51  # 50 bins → 51 edges
        assert len(feat_a.histogram_counts) == 50

    def test_save_and_load(self, tmp_data_dir):
        df = _make_df()
        stats = compute_baseline(df)
        baseline = save_baseline(
            stats, num_records=1000, version="v_test",
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        assert baseline.version == "v_test"
        assert baseline.num_records == 1000

        loaded = load_baseline(artifacts_dir=tmp_data_dir["artifacts_dir"])
        assert loaded is not None
        assert loaded.version == "v_test"
        assert len(loaded.features) == 3

    def test_load_nonexistent(self, tmp_data_dir):
        result = load_baseline(artifacts_dir=tmp_data_dir["artifacts_dir"])
        assert result is None

    def test_expose_to_prometheus(self, tmp_data_dir):
        df = _make_df()
        stats = compute_baseline(df)
        baseline = save_baseline(
            stats, num_records=1000, version="v_test",
            artifacts_dir=tmp_data_dir["artifacts_dir"],
        )
        # Should not raise
        expose_baseline_to_prometheus(baseline)

    def test_null_handling(self):
        df = pd.DataFrame({
            "feat_a": [1.0, 2.0, None, 4.0, None],
            "feat_b": [None, None, None, None, None],
        })
        stats = compute_baseline(df)
        feat_a = next(s for s in stats if s.feature_name == "feat_a")
        assert feat_a.null_rate == pytest.approx(0.4)

        feat_b = next(s for s in stats if s.feature_name == "feat_b")
        assert feat_b.null_rate == 1.0
        assert feat_b.mean == 0.0
