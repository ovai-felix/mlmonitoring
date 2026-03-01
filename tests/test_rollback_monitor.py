"""Tests for the post-swap rollback monitor."""
import time
from unittest.mock import patch

import pytest

from src.services.rollback_monitor import RollbackMonitor


@pytest.fixture
def monitor():
    return RollbackMonitor()


class TestRollbackMonitor:
    def test_inactive_by_default(self, monitor):
        """No monitoring active → check returns False."""
        assert monitor.is_active is False
        assert monitor.check(0.95, 0.5) is False

    def test_no_degradation_returns_false(self, monitor):
        """Within window, no degradation → check returns False."""
        monitor.start_monitoring(baseline_accuracy=0.95, baseline_latency_p99=0.5)
        assert monitor.is_active is True
        assert monitor.check(current_accuracy=0.94, current_latency_p99=0.5) is False

    def test_accuracy_drop_returns_true(self, monitor):
        """Within window, accuracy dropped >3% → check returns True."""
        monitor.start_monitoring(baseline_accuracy=0.95, baseline_latency_p99=0.5)
        # Drop of 0.05 > threshold of 0.03
        assert monitor.check(current_accuracy=0.90, current_latency_p99=0.5) is True

    def test_latency_spike_returns_true(self, monitor):
        """Within window, latency spike → check returns True."""
        monitor.start_monitoring(baseline_accuracy=0.95, baseline_latency_p99=0.5)
        # Latency 2.5s > threshold 2.0s
        assert monitor.check(current_accuracy=0.95, current_latency_p99=2.5) is True

    def test_window_expired_inactive(self, monitor, monkeypatch):
        """Window expired → is_active returns False."""
        monitor.start_monitoring(baseline_accuracy=0.95, baseline_latency_p99=0.5)
        # Simulate time passing beyond the window
        from src.config import settings
        monkeypatch.setattr(settings, "auto_rollback_window_seconds", 1800)
        monitor._swap_time = time.time() - 2000  # 2000s ago, window is 1800s
        assert monitor.is_active is False
        # Check should return False even with bad metrics
        assert monitor.check(0.5, 5.0) is False

    def test_stop_monitoring_clears_state(self, monitor):
        """stop_monitoring clears all state."""
        monitor.start_monitoring(baseline_accuracy=0.95, baseline_latency_p99=0.5)
        assert monitor.is_active is True
        monitor.stop_monitoring()
        assert monitor.is_active is False

    def test_get_state(self, monitor):
        """get_state returns monitoring info."""
        state = monitor.get_state()
        assert state["is_active"] is False
        assert state["baseline_accuracy"] is None

        monitor.start_monitoring(baseline_accuracy=0.95, baseline_latency_p99=0.5)
        state = monitor.get_state()
        assert state["is_active"] is True
        assert state["baseline_accuracy"] == 0.95
