import pytest
from unittest.mock import patch


class TestTrainingRoute:
    def test_trigger_unknown_model(self, test_client):
        resp = test_client.post(
            "/training/trigger",
            json={"model_type": "unknown", "data_version": "v_test"},
        )
        assert resp.status_code == 400

    @patch("src.routes.training.check_cooldown", return_value=False)
    def test_trigger_cooldown_active(self, mock_cooldown, test_client):
        resp = test_client.post(
            "/training/trigger",
            json={"model_type": "classification", "data_version": "v_test"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_get_training_status(self, test_client):
        resp = test_client.get("/training/status")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_training_status_with_filter(self, test_client):
        resp = test_client.get("/training/status?model_type=classification")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
