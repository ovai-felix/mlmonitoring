"""Automated retraining service — drift-triggered and accuracy-triggered."""
import glob as globmod
import logging
from pathlib import Path

from src.config import settings
from src.metrics import (
    consecutive_retrain_failures,
    retrain_outcome_total,
    retrain_triggered_total,
)

logger = logging.getLogger(__name__)

# Module-level state
_consecutive_drift_count: int = 0
_consecutive_failures: int = 0
_last_trigger_reason: str | None = None
_baseline_accuracy: float | None = None
_auto_trigger_disabled: bool = False


def reset_state():
    """Reset module state (for testing)."""
    global _consecutive_drift_count, _consecutive_failures
    global _last_trigger_reason, _baseline_accuracy, _auto_trigger_disabled
    _consecutive_drift_count = 0
    _consecutive_failures = 0
    _last_trigger_reason = None
    _baseline_accuracy = None
    _auto_trigger_disabled = False
    consecutive_retrain_failures.set(0)


def set_baseline_accuracy(accuracy: float):
    """Record baseline accuracy after a successful model swap."""
    global _baseline_accuracy
    _baseline_accuracy = accuracy


def get_state() -> dict:
    """Return current retrain service state for monitoring."""
    return {
        "consecutive_drift_count": _consecutive_drift_count,
        "consecutive_failures": _consecutive_failures,
        "last_trigger_reason": _last_trigger_reason,
        "baseline_accuracy": _baseline_accuracy,
        "auto_trigger_disabled": _auto_trigger_disabled,
    }


def get_latest_data_version(features_dir: Path | None = None) -> str | None:
    """Find the latest data version from parquet files in features dir."""
    fdir = features_dir or settings.features_dir
    files = sorted(fdir.glob("*.parquet"))
    if not files:
        return None
    # Filename pattern: features_v{version}.parquet
    name = files[-1].stem
    if name.startswith("features_"):
        return name.replace("features_", "")
    return name


def check_and_trigger_retrain(
    drift_report: dict | None = None,
    performance_metrics: dict | None = None,
    db_path: Path | None = None,
) -> dict:
    """Check drift/accuracy triggers and auto-retrain if needed.

    Returns dict with: triggered, reason, outcome, details.
    """
    global _consecutive_drift_count, _consecutive_failures
    global _last_trigger_reason, _auto_trigger_disabled

    result = {"triggered": False, "reason": None, "outcome": None, "details": None}

    if _auto_trigger_disabled:
        result["details"] = "Auto-triggering disabled due to consecutive failures"
        return result

    reason = None

    # Check drift trigger
    if drift_report and drift_report.get("drift_detected"):
        _consecutive_drift_count += 1
        if _consecutive_drift_count >= settings.retrain_consecutive_drift_required:
            reason = "drift"
    else:
        _consecutive_drift_count = 0

    # Check accuracy trigger
    if reason is None and performance_metrics and _baseline_accuracy is not None:
        windows = performance_metrics.get("windows", {})
        w1000 = windows.get(1000) or windows.get("1000")
        if w1000:
            current_accuracy = w1000.get("accuracy", 1.0)
            drop = _baseline_accuracy - current_accuracy
            if drop > settings.retrain_accuracy_drop_threshold:
                reason = "accuracy_drop"

    if reason is None:
        return result

    # Guard: don't re-trigger for identical reason
    if reason == _last_trigger_reason and reason == "drift":
        result["details"] = "Same drift signal, skipping duplicate trigger"
        return result

    # Guard: check cooldown
    from src.services.training_service import check_cooldown
    if not check_cooldown("classification", db_path):
        result["details"] = "Training cooldown active"
        return result

    # Trigger retraining
    _last_trigger_reason = reason
    retrain_triggered_total.labels(reason=reason).inc()
    result["triggered"] = True
    result["reason"] = reason

    outcome = _execute_retrain(db_path)
    result["outcome"] = outcome["status"]
    result["details"] = outcome.get("detail")

    if outcome["status"] == "success":
        _consecutive_failures = 0
        consecutive_retrain_failures.set(0)
        retrain_outcome_total.labels(outcome="success").inc()
        # Reset drift count after successful retrain
        _consecutive_drift_count = 0
        _last_trigger_reason = None
    else:
        _consecutive_failures += 1
        consecutive_retrain_failures.set(_consecutive_failures)
        retrain_outcome_total.labels(outcome="failed").inc()

        if _consecutive_failures >= 3:
            _auto_trigger_disabled = True
            from src.services.alert_service import alert_service
            alert_service.send_alert(
                "Retraining Escalation",
                f"3 consecutive retrain failures. Auto-retraining disabled.",
                severity="critical",
                context={"failures": _consecutive_failures, "last_reason": reason},
            )

    return result


def _execute_retrain(db_path: Path | None = None) -> dict:
    """Run training and model reload. Returns {status, detail}."""
    try:
        data_version = get_latest_data_version()
        if not data_version:
            return {"status": "failed", "detail": "No data version found"}

        from src.services.training_service import train_classifier, train_anomaly
        clf_result = train_classifier(data_version, n_trials=3, db_path=db_path)
        if clf_result.get("status") == "skipped":
            return {"status": "failed", "detail": "Classifier training skipped (cooldown)"}

        train_anomaly(data_version, db_path=db_path)

        # Reload models via blue-green swap
        from src.services.model_manager import model_manager
        reload_result = model_manager.reload()
        if reload_result.get("status") != "ok":
            return {"status": "failed", "detail": reload_result.get("detail", "Reload failed")}

        # Start post-swap monitoring
        from src.services.rollback_monitor import rollback_monitor
        from src.services.performance_metrics import compute_rolling_metrics
        perf = compute_rolling_metrics(db_path)
        w1000 = (perf.get("windows") or {}).get(1000, {})
        current_acc = w1000.get("accuracy", 0.95)
        rollback_monitor.start_monitoring(baseline_accuracy=current_acc, baseline_latency_p99=0.5)

        # Update baseline accuracy
        set_baseline_accuracy(current_acc)

        from src.services.alert_service import alert_service
        alert_service.send_alert(
            "Model Retrained",
            f"Auto-retrain completed. New model deployed with accuracy={current_acc:.4f}",
            severity="info",
        )

        return {"status": "success", "detail": f"Retrained with data {data_version}"}

    except Exception as e:
        logger.exception("Retrain execution failed")
        return {"status": "failed", "detail": str(e)}
