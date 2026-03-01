"""Tests for the alert notification service."""
from unittest.mock import MagicMock, patch

import pytest

from src.services.alert_service import AlertService


@pytest.fixture
def alert_svc():
    return AlertService()


class TestAlertService:
    def test_info_severity_logs_only(self, alert_svc):
        """Info alerts should not send to Slack or Email."""
        result = alert_svc.send_alert("Test", "msg", severity="info")
        assert result["severity"] == "info"
        assert result["channels"] == []

    def test_warning_sends_slack_only(self, alert_svc, monkeypatch):
        """Warning alerts should route to Slack only."""
        from src.config import settings

        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/test")
        with patch.object(alert_svc, "_send_slack", return_value=True) as mock_slack:
            with patch.object(alert_svc, "_send_email", return_value=True) as mock_email:
                result = alert_svc.send_alert("Drift", "drift detected", severity="warning")
        mock_slack.assert_called_once()
        mock_email.assert_not_called()
        assert "slack" in result["channels"]
        assert "email" not in result["channels"]

    def test_critical_sends_both_channels(self, alert_svc, monkeypatch):
        """Critical alerts should route to both Slack and Email."""
        from src.config import settings

        monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.com/test")
        monkeypatch.setattr(settings, "smtp_host", "smtp.test.com")
        monkeypatch.setattr(settings, "alert_email_to", "ops@test.com")
        with patch.object(alert_svc, "_send_slack", return_value=True) as mock_slack:
            with patch.object(alert_svc, "_send_email", return_value=True) as mock_email:
                result = alert_svc.send_alert(
                    "Service Down", "model server unreachable",
                    severity="critical", context={"host": "prod-1"},
                )
        mock_slack.assert_called_once()
        mock_email.assert_called_once()
        assert "slack" in result["channels"]
        assert "email" in result["channels"]

    def test_graceful_skip_no_config(self, alert_svc, monkeypatch):
        """When no webhook/SMTP configured, channels silently skip."""
        from src.config import settings

        monkeypatch.setattr(settings, "slack_webhook_url", "")
        monkeypatch.setattr(settings, "smtp_host", "")
        result = alert_svc.send_alert("Test", "msg", severity="critical")
        assert result["channels"] == []
