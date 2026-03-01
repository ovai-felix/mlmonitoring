"""Post-swap auto-rollback monitor.

Monitors model health for a configurable window after a blue-green swap.
If accuracy drops or latency spikes, triggers automatic rollback.
"""
import logging
import time

from src.config import settings
from src.metrics import post_swap_monitoring_active

logger = logging.getLogger(__name__)


class RollbackMonitor:
    def __init__(self):
        self._swap_time: float | None = None
        self._baseline_accuracy: float | None = None
        self._baseline_latency_p99: float | None = None

    @property
    def is_active(self) -> bool:
        """True if within the post-swap monitoring window."""
        if self._swap_time is None:
            return False
        elapsed = time.time() - self._swap_time
        return elapsed < settings.auto_rollback_window_seconds

    def start_monitoring(self, baseline_accuracy: float, baseline_latency_p99: float):
        """Begin post-swap monitoring with baseline metrics."""
        self._swap_time = time.time()
        self._baseline_accuracy = baseline_accuracy
        self._baseline_latency_p99 = baseline_latency_p99
        post_swap_monitoring_active.set(1)
        logger.info(
            "Post-swap monitoring started: accuracy=%.4f, p99=%.3fs, window=%ds",
            baseline_accuracy, baseline_latency_p99, settings.auto_rollback_window_seconds,
        )

    def check(self, current_accuracy: float, current_latency_p99: float) -> bool:
        """Check if rollback is needed. Returns True if degradation detected.

        Only checks during the active monitoring window.
        """
        if not self.is_active:
            return False

        # Check accuracy drop
        if self._baseline_accuracy is not None:
            accuracy_drop = self._baseline_accuracy - current_accuracy
            if accuracy_drop > settings.auto_rollback_accuracy_drop:
                logger.warning(
                    "Accuracy degradation detected: %.4f -> %.4f (drop=%.4f > threshold=%.4f)",
                    self._baseline_accuracy, current_accuracy,
                    accuracy_drop, settings.auto_rollback_accuracy_drop,
                )
                return True

        # Check latency spike
        if current_latency_p99 > settings.auto_rollback_latency_p99:
            logger.warning(
                "Latency spike detected: p99=%.3fs > threshold=%.3fs",
                current_latency_p99, settings.auto_rollback_latency_p99,
            )
            return True

        return False

    def stop_monitoring(self):
        """Clear monitoring state."""
        self._swap_time = None
        self._baseline_accuracy = None
        self._baseline_latency_p99 = None
        post_swap_monitoring_active.set(0)
        logger.info("Post-swap monitoring stopped")

    def get_state(self) -> dict:
        """Return current monitoring state for API."""
        return {
            "is_active": self.is_active,
            "swap_time": self._swap_time,
            "baseline_accuracy": self._baseline_accuracy,
            "baseline_latency_p99": self._baseline_latency_p99,
            "window_seconds": settings.auto_rollback_window_seconds,
            "elapsed_seconds": (time.time() - self._swap_time) if self._swap_time else None,
        }


rollback_monitor = RollbackMonitor()
