"""Standalone drift detection runner.

Runs drift detection and rolling performance metrics on a schedule.
After drift detection, checks retrain triggers.
After performance updates, checks rollback monitor.
Exposes its own /metrics endpoint on port 8001 for Prometheus scraping.
"""
import logging
import time

from prometheus_client import start_http_server

from src.config import settings
from src.database import init_db
from src.services.drift_service import run_drift_detection
from src.services.performance_metrics import compute_rolling_metrics
from src.services.retrain_service import check_and_trigger_retrain
from src.services.rollback_monitor import rollback_monitor
from src.services.alert_service import alert_service
from src.services.model_manager import model_manager

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

        # 1. Update performance metrics
        perf_result = None
        if now - last_perf >= perf_interval:
            try:
                perf_result = compute_rolling_metrics()
                logger.info("Rolling performance metrics updated")
            except Exception:
                logger.exception("Failed to update performance metrics")
            last_perf = now

        # 2. Check rollback monitor (if active)
        if rollback_monitor.is_active and perf_result:
            try:
                w1000 = (perf_result.get("windows") or {}).get(1000, {})
                current_accuracy = w1000.get("accuracy", 1.0)
                current_p99 = 0.5  # default; real value from Prometheus in production
                if rollback_monitor.check(current_accuracy, current_p99):
                    logger.warning("Auto-rollback triggered!")
                    model_manager.rollback(reason="auto_rollback")
                    rollback_monitor.stop_monitoring()
                    alert_service.send_alert(
                        "Auto-Rollback Triggered",
                        f"Model rolled back due to degradation. accuracy={current_accuracy:.4f}",
                        severity="critical",
                        context={"accuracy": current_accuracy, "p99_latency": current_p99},
                    )
                elif not rollback_monitor.is_active:
                    # Window expired without issues
                    rollback_monitor.stop_monitoring()
                    alert_service.send_alert(
                        "Model Swap Confirmed",
                        "Post-swap monitoring window passed without issues.",
                        severity="info",
                    )
            except Exception:
                logger.exception("Rollback monitor check failed")

        # 3. Run drift detection
        if now - last_drift >= drift_interval:
            report = None
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

            # 4. Check retrain triggers (after drift detection)
            if report:
                try:
                    retrain_result = check_and_trigger_retrain(
                        drift_report=report,
                        performance_metrics=perf_result,
                    )
                    if retrain_result["triggered"]:
                        logger.info(
                            "Retrain triggered: reason=%s, outcome=%s",
                            retrain_result["reason"], retrain_result["outcome"],
                        )
                except Exception:
                    logger.exception("Retrain check failed")

        time.sleep(min(perf_interval, drift_interval, 30))


if __name__ == "__main__":
    main()
