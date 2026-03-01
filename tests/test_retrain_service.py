"""Tests for the automated retraining service."""
from unittest.mock import MagicMock, patch

import pytest

from src.services import retrain_service
from src.services.retrain_service import (
    check_and_trigger_retrain,
    get_latest_data_version,
    get_state,
    reset_state,
    set_baseline_accuracy,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset module state before each test."""
    reset_state()
    yield
    reset_state()


class TestRetrainService:
    def test_no_drift_no_accuracy_drop(self):
        """No triggers → not triggered."""
        result = check_and_trigger_retrain(
            drift_report={"drift_detected": False},
            performance_metrics={"windows": {}},
        )
        assert result["triggered"] is False

    def test_single_drift_not_triggered(self):
        """Single drift detection should not trigger (needs 2 consecutive)."""
        result = check_and_trigger_retrain(
            drift_report={"drift_detected": True},
        )
        assert result["triggered"] is False
        assert get_state()["consecutive_drift_count"] == 1

    @patch("src.services.retrain_service._execute_retrain")
    @patch("src.services.training_service.check_cooldown", return_value=True)
    def test_two_consecutive_drifts_triggers(self, mock_cooldown, mock_retrain):
        """Two consecutive drift detections should trigger retraining."""
        mock_retrain.return_value = {"status": "success", "detail": "ok"}

        # First drift
        check_and_trigger_retrain(drift_report={"drift_detected": True})
        # Second drift
        result = check_and_trigger_retrain(drift_report={"drift_detected": True})

        assert result["triggered"] is True
        assert result["reason"] == "drift"
        assert result["outcome"] == "success"
        mock_retrain.assert_called_once()

    @patch("src.services.retrain_service._execute_retrain")
    @patch("src.services.training_service.check_cooldown", return_value=True)
    def test_accuracy_drop_triggers(self, mock_cooldown, mock_retrain):
        """Accuracy drop below threshold should trigger retraining."""
        mock_retrain.return_value = {"status": "success", "detail": "ok"}
        set_baseline_accuracy(0.95)

        result = check_and_trigger_retrain(
            drift_report={"drift_detected": False},
            performance_metrics={"windows": {1000: {"accuracy": 0.88}}},
        )

        assert result["triggered"] is True
        assert result["reason"] == "accuracy_drop"

    @patch("src.services.training_service.check_cooldown", return_value=False)
    def test_cooldown_active_skips(self, mock_cooldown):
        """Cooldown active should prevent triggering."""
        set_baseline_accuracy(0.95)

        result = check_and_trigger_retrain(
            drift_report={"drift_detected": False},
            performance_metrics={"windows": {1000: {"accuracy": 0.88}}},
        )

        assert result["triggered"] is False
        assert result["details"] == "Training cooldown active"

    @patch("src.services.retrain_service._execute_retrain")
    @patch("src.services.training_service.check_cooldown", return_value=True)
    def test_three_failures_escalation(self, mock_cooldown, mock_retrain):
        """Three consecutive failures should disable auto-triggering and send critical alert."""
        mock_retrain.return_value = {"status": "failed", "detail": "training error"}
        set_baseline_accuracy(0.95)

        with patch("src.services.alert_service.AlertService.send_alert") as mock_send:
            for i in range(3):
                retrain_service._last_trigger_reason = None  # allow re-trigger
                check_and_trigger_retrain(
                    drift_report={"drift_detected": False},
                    performance_metrics={"windows": {1000: {"accuracy": 0.88}}},
                )

            mock_send.assert_called()
            call_args = mock_send.call_args
            assert call_args[1].get("severity") == "critical" or call_args[0][2] == "critical"

        assert get_state()["auto_trigger_disabled"] is True

        # Further triggers should be blocked
        result = check_and_trigger_retrain(
            drift_report={"drift_detected": False},
            performance_metrics={"windows": {1000: {"accuracy": 0.88}}},
        )
        assert result["triggered"] is False

    @patch("src.services.retrain_service._execute_retrain")
    @patch("src.services.training_service.check_cooldown", return_value=True)
    def test_success_resets_failure_counter(self, mock_cooldown, mock_retrain):
        """Successful retrain should reset consecutive failure counter."""
        # First: a failure
        mock_retrain.return_value = {"status": "failed", "detail": "error"}
        set_baseline_accuracy(0.95)
        check_and_trigger_retrain(
            performance_metrics={"windows": {1000: {"accuracy": 0.88}}},
        )
        assert get_state()["consecutive_failures"] == 1

        # Then: a success
        mock_retrain.return_value = {"status": "success", "detail": "ok"}
        retrain_service._last_trigger_reason = None
        check_and_trigger_retrain(
            performance_metrics={"windows": {1000: {"accuracy": 0.88}}},
        )
        assert get_state()["consecutive_failures"] == 0

    @patch("src.services.training_service.check_cooldown", return_value=True)
    def test_same_drift_signal_not_retriggered(self, mock_cooldown):
        """Same drift signal should not re-trigger after first trigger."""
        with patch("src.services.retrain_service._execute_retrain") as mock_retrain:
            mock_retrain.return_value = {"status": "success", "detail": "ok"}
            # Trigger via 2 drifts
            check_and_trigger_retrain(drift_report={"drift_detected": True})
            result1 = check_and_trigger_retrain(drift_report={"drift_detected": True})
            assert result1["triggered"] is True

            # Third drift — same reason, should skip
            result2 = check_and_trigger_retrain(drift_report={"drift_detected": True})
            assert result2["triggered"] is False


class TestGetLatestDataVersion:
    def test_finds_latest_parquet(self, tmp_path):
        """Should return latest version from parquet filenames."""
        (tmp_path / "features_v1.parquet").touch()
        (tmp_path / "features_v2.parquet").touch()
        version = get_latest_data_version(tmp_path)
        assert version == "v2"

    def test_empty_dir_returns_none(self, tmp_path):
        """Empty features dir should return None."""
        version = get_latest_data_version(tmp_path)
        assert version is None
