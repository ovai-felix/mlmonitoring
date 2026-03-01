import pytest

from src.services.model_manager import ModelManager, ModelSlot


@pytest.fixture
def manager():
    return ModelManager()


def test_initial_state_not_ready(manager):
    assert not manager.active.is_ready
    assert not manager.standby.is_ready
    assert manager._active_color == "blue"


def test_load_with_no_models_on_disk(manager, tmp_data_dir, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "models_dir", tmp_data_dir["models_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", tmp_data_dir["artifacts_dir"])

    manager.load_slot(manager.active)
    assert not manager.active.has_any_model()
    assert manager.active.loaded_at is not None


def test_load_classifier_only(manager, saved_dummy_models, monkeypatch):
    from src.config import settings
    import shutil

    models_dir = saved_dummy_models["models_dir"]
    # Remove LSTM and anomaly to test partial load
    shutil.rmtree(models_dir / "lstm")
    shutil.rmtree(models_dir / "anomaly")

    monkeypatch.setattr(settings, "models_dir", models_dir)
    monkeypatch.setattr(settings, "artifacts_dir", saved_dummy_models["artifacts_dir"])

    manager.load_slot(manager.active)
    assert manager.active.classifier is not None
    assert manager.active.lstm is None
    assert manager.active.anomaly is None
    assert manager.active.has_any_model()


def test_load_all_models(manager, saved_dummy_models, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "models_dir", saved_dummy_models["models_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", saved_dummy_models["artifacts_dir"])

    manager.load_slot(manager.active)
    assert manager.active.classifier is not None
    assert manager.active.lstm is not None
    assert manager.active.anomaly is not None
    assert manager.active.pipeline is not None


def test_warm_up_sets_ready(manager, saved_dummy_models, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "models_dir", saved_dummy_models["models_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", saved_dummy_models["artifacts_dir"])

    manager.load_slot(manager.active)
    assert not manager.active.is_ready
    manager.warm_up(manager.active)
    assert manager.active.is_ready


def test_reload_swaps_colors(manager, saved_dummy_models, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "models_dir", saved_dummy_models["models_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", saved_dummy_models["artifacts_dir"])

    manager.initial_load()
    assert manager._active_color == "blue"

    result = manager.reload()
    assert result["status"] == "ok"
    assert manager._active_color == "green"

    result = manager.reload()
    assert result["status"] == "ok"
    assert manager._active_color == "blue"


def test_rollback_to_previous(manager, saved_dummy_models, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "models_dir", saved_dummy_models["models_dir"])
    monkeypatch.setattr(settings, "artifacts_dir", saved_dummy_models["artifacts_dir"])

    manager.initial_load()
    manager.reload()
    assert manager._active_color == "green"

    result = manager.rollback()
    assert result["status"] == "ok"
    assert manager._active_color == "blue"


def test_rollback_fails_when_standby_not_ready(manager):
    result = manager.rollback()
    assert result["status"] == "error"
    assert "not ready" in result["detail"]


def test_get_info_structure(manager):
    info = manager.get_info()
    assert "active_color" in info
    assert "blue" in info
    assert "green" in info
    for color in ["blue", "green"]:
        slot = info[color]
        assert "color" in slot
        assert "is_ready" in slot
        assert "version_tag" in slot
        assert "has_classifier" in slot
        assert "has_lstm" in slot
        assert "has_anomaly" in slot
        assert "has_pipeline" in slot
