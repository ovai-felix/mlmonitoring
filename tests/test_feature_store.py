import numpy as np
import pandas as pd
import pytest

from src.database import init_db
from src.services.feature_store import (
    get_entity_inference_features,
    list_versions,
    load_training_features,
    save_inference_features,
    save_training_features,
)


def _make_features_df(n=50):
    rng = np.random.default_rng(42)
    data = {f"feat_{i}": rng.standard_normal(n) for i in range(10)}
    return pd.DataFrame(data)


class TestTrainingFeatureStore:
    def test_save_and_load(self, tmp_data_dir):
        df = _make_features_df()
        version = save_training_features(df, version="v_test", features_dir=tmp_data_dir["features_dir"])
        assert version == "v_test"

        loaded = load_training_features("v_test", features_dir=tmp_data_dir["features_dir"])
        pd.testing.assert_frame_equal(df, loaded)

    def test_list_versions(self, tmp_data_dir):
        df = _make_features_df()
        save_training_features(df, version="v_2024-01-01", features_dir=tmp_data_dir["features_dir"])
        save_training_features(df, version="v_2024-01-02", features_dir=tmp_data_dir["features_dir"])

        versions = list_versions(features_dir=tmp_data_dir["features_dir"])
        assert len(versions) == 2
        assert versions[0]["version"] == "v_2024-01-01"
        assert versions[1]["version"] == "v_2024-01-02"
        assert versions[0]["num_records"] == 50

    def test_load_nonexistent(self, tmp_data_dir):
        with pytest.raises(FileNotFoundError):
            load_training_features("v_nonexistent", features_dir=tmp_data_dir["features_dir"])


class TestInferenceFeatureStore:
    def test_save_and_get(self, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        features = {"feat_0": 1.0, "feat_1": 2.0}
        save_inference_features("entity_1", features, db_path=tmp_data_dir["db_path"])

        result = get_entity_inference_features("entity_1", db_path=tmp_data_dir["db_path"])
        assert result is not None
        assert result["entity_id"] == "entity_1"
        assert result["features"]["feat_0"] == 1.0

    def test_upsert(self, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        save_inference_features("entity_1", {"a": 1.0}, db_path=tmp_data_dir["db_path"])
        save_inference_features("entity_1", {"a": 2.0}, db_path=tmp_data_dir["db_path"])

        result = get_entity_inference_features("entity_1", db_path=tmp_data_dir["db_path"])
        assert result["features"]["a"] == 2.0

    def test_get_nonexistent(self, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        result = get_entity_inference_features("nonexistent", db_path=tmp_data_dir["db_path"])
        assert result is None
