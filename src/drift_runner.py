"""Standalone drift detection runner.

Runs drift detection and rolling performance metrics on a schedule.
Exposes its own /metrics endpoint on port 8001 for Prometheus scraping.
"""
import logging
import time

from prometheus_client import start_http_server

from src.config import settings
from src.database import init_db
from src.services.drift_service import run_drift_detection
from src.services.performance_metrics import compute_rolling_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    init_db()

    # Start Prometheus metrics server on port 8001
    start_http_server(8001)
    logger.info("Drift detector metrics server started on :8001")

    drift_interval = settings.drift_schedule_seconds
    perf_interval = settings.performance_update_seconds

    last_drift = 0.0
    last_perf = 0.0

    while True:
        now = time.time()

        if now - last_perf >= perf_interval:
            try:
                compute_rolling_metrics()
                logger.info("Rolling performance metrics updated")
            except Exception:
                logger.exception("Failed to update performance metrics")
            last_perf = now

        if now - last_drift >= drift_interval:
            try:
                report = run_drift_detection(n_recent=settings.drift_n_recent)
                logger.info(
                    "Drift detection complete: drift=%s, features=%d",
                    report.get("drift_detected"),
                    len(report.get("drifted_features", [])),
                )
            except Exception:
                logger.exception("Failed to run drift detection")
            last_drift = now

        time.sleep(min(perf_interval, drift_interval, 30))


if __name__ == "__main__":
    main()
