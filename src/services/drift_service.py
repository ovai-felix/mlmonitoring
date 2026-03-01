import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

from src.config import settings
from src.database import get_connection
from src.metrics import (
    data_quality_null_rate,
    data_quality_out_of_range_rate,
    drift_detected,
    drift_report_generated_total,
    feature_drift_ks_pvalue,
    feature_drift_ks_statistic,
    feature_drift_psi,
    prediction_drift_fraud_rate,
    prediction_drift_mean_confidence,
)
from src.services.baseline_stats import load_baseline

logger = logging.getLogger(__name__)

DRIFT_REPORTS_DIR = "drift_reports"
PSI_THRESHOLD = 0.2


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 50) -> float:
    """Compute Population Stability Index between two distributions.

    Uses shared bin edges derived from the reference distribution.
    Returns PSI value (0 = identical, >0.2 = significant drift).
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Create bins from reference
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    # Normalize to proportions with smoothing to avoid division by zero
    eps = 1e-6
    ref_pct = (ref_counts + eps) / (ref_counts.sum() + eps * bins)
    cur_pct = (cur_counts + eps) / (cur_counts.sum() + eps * bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _reconstruct_reference(histogram_bins: list, histogram_counts: list) -> np.ndarray:
    """Reconstruct samples from baseline histogram bins/counts."""
    if not histogram_bins or not histogram_counts:
        return np.array([])

    bins = np.array(histogram_bins)
    counts = np.array(histogram_counts)
    midpoints = (bins[:-1] + bins[1:]) / 2
    samples = np.repeat(midpoints, counts.astype(int))
    return samples


def run_drift_detection(
    n_recent: int = 5000,
    db_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> dict:
    """Run drift detection comparing recent predictions to baseline.

    Loads baseline stats and recent transformed features from the predictions table.
    Computes PSI and KS-test per feature, sets Prometheus gauges.
    Saves a JSON report to data/artifacts/drift_reports/.

    Returns the drift report dict.
    """
    adir = artifacts_dir or settings.artifacts_dir
    report_dir = adir / DRIFT_REPORTS_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_recent": n_recent,
        "features": {},
        "drift_detected": False,
        "drifted_features": [],
        "prediction_drift": {},
        "data_quality": {},
    }

    # Load baseline
    baseline = load_baseline(artifacts_dir=artifacts_dir)
    if baseline is None:
        report["error"] = "No baseline found"
        drift_report_generated_total.labels(status="no_baseline").inc()
        return report

    # Load recent transformed features from DB
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT transformed_features, prediction_label, prediction_confidence
               FROM predictions
               WHERE transformed_features IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (n_recent,),
        ).fetchall()

    if len(rows) < 100:
        report["error"] = f"Insufficient data: {len(rows)} rows (need >= 100)"
        drift_report_generated_total.labels(status="insufficient_data").inc()
        return report

    # Parse transformed features
    current_features = []
    pred_labels = []
    pred_confidences = []
    for row in rows:
        try:
            feats = json.loads(row["transformed_features"])
            current_features.append(feats)
            pred_labels.append(row["prediction_label"])
            pred_confidences.append(row["prediction_confidence"])
        except (json.JSONDecodeError, TypeError):
            continue

    if not current_features:
        report["error"] = "Could not parse any transformed features"
        drift_report_generated_total.labels(status="parse_error").inc()
        return report

    current_array = np.array(current_features)

    # Per-feature drift analysis
    any_drift = False
    for i, feat_stat in enumerate(baseline.features):
        if i >= current_array.shape[1]:
            break

        feature_name = feat_stat.feature_name
        reference = _reconstruct_reference(
            feat_stat.histogram_bins, feat_stat.histogram_counts
        )
        current_col = current_array[:, i]

        # Compute PSI
        psi_val = compute_psi(reference, current_col)

        # Compute KS test
        if len(reference) > 0 and len(current_col) > 0:
            ks_stat, ks_pval = stats.ks_2samp(reference, current_col)
        else:
            ks_stat, ks_pval = 0.0, 1.0

        feature_drift_psi.labels(feature=feature_name).set(psi_val)
        feature_drift_ks_statistic.labels(feature=feature_name).set(ks_stat)
        feature_drift_ks_pvalue.labels(feature=feature_name).set(ks_pval)

        is_drifted = psi_val > PSI_THRESHOLD
        if is_drifted:
            any_drift = True
            report["drifted_features"].append(feature_name)

        # Data quality: null rate
        null_rate = float(np.isnan(current_col).mean()) if len(current_col) > 0 else 0.0
        data_quality_null_rate.labels(feature=feature_name).set(null_rate)

        # Data quality: out-of-range rate (beyond baseline min/max)
        if len(current_col) > 0 and feat_stat.min != feat_stat.max:
            oor = float(
                ((current_col < feat_stat.min) | (current_col > feat_stat.max)).mean()
            )
        else:
            oor = 0.0
        data_quality_out_of_range_rate.labels(feature=feature_name).set(oor)

        report["features"][feature_name] = {
            "psi": psi_val,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "drifted": is_drifted,
            "null_rate": null_rate,
            "out_of_range_rate": oor,
        }
        report["data_quality"][feature_name] = {
            "null_rate": null_rate,
            "out_of_range_rate": oor,
        }

    report["drift_detected"] = any_drift
    drift_detected.set(1.0 if any_drift else 0.0)

    # Prediction drift
    if pred_labels:
        fraud_rate = sum(1 for l in pred_labels if l == 1) / len(pred_labels)
        mean_conf = np.mean([c for c in pred_confidences if c is not None]) if pred_confidences else 0.0
        prediction_drift_fraud_rate.set(fraud_rate)
        prediction_drift_mean_confidence.set(float(mean_conf))
        report["prediction_drift"] = {
            "fraud_rate": fraud_rate,
            "mean_confidence": float(mean_conf),
        }

    # Save report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = report_dir / f"drift_report_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    report["report_path"] = str(report_path)
    drift_report_generated_total.labels(status="success").inc()

    logger.info(
        "Drift detection complete: drift=%s, drifted_features=%d",
        any_drift,
        len(report["drifted_features"]),
    )
    return report


def run_drift_with_evidently(
    n_recent: int = 5000,
    db_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> dict:
    """Run Evidently-based drift detection with HTML reports.

    Falls back to manual-only if Evidently is not installed.
    Always runs manual PSI/KS to populate Prometheus gauges.
    """
    # Always run manual drift detection for Prometheus gauges
    manual_report = run_drift_detection(
        n_recent=n_recent, db_path=db_path, artifacts_dir=artifacts_dir
    )

    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report
    except ImportError:
        logger.info("Evidently not installed, using manual drift detection only")
        manual_report["evidently_available"] = False
        return manual_report

    adir = artifacts_dir or settings.artifacts_dir
    baseline = load_baseline(artifacts_dir=artifacts_dir)
    if baseline is None:
        return manual_report

    # Load current data
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """SELECT transformed_features FROM predictions
               WHERE transformed_features IS NOT NULL
               ORDER BY id DESC LIMIT ?""",
            (n_recent,),
        ).fetchall()

    if len(rows) < 100:
        return manual_report

    import pandas as pd

    feature_names = [f.feature_name for f in baseline.features]
    current_data = []
    for row in rows:
        try:
            feats = json.loads(row["transformed_features"])
            current_data.append(feats)
        except (json.JSONDecodeError, TypeError):
            continue

    if not current_data:
        return manual_report

    current_df = pd.DataFrame(current_data, columns=feature_names[: len(current_data[0])])

    # Reconstruct reference DataFrame
    ref_data = {}
    for feat_stat in baseline.features:
        samples = _reconstruct_reference(feat_stat.histogram_bins, feat_stat.histogram_counts)
        if len(samples) > 0:
            ref_data[feat_stat.feature_name] = samples

    if not ref_data:
        return manual_report

    min_len = min(len(v) for v in ref_data.values())
    ref_df = pd.DataFrame({k: v[:min_len] for k, v in ref_data.items()})

    # Ensure matching columns
    common_cols = list(set(ref_df.columns) & set(current_df.columns))
    if not common_cols:
        return manual_report

    ref_df = ref_df[common_cols]
    current_df = current_df[common_cols]

    try:
        report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        report.run(reference_data=ref_df, current_data=current_df)

        report_dir = adir / DRIFT_REPORTS_DIR
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        html_path = report_dir / f"evidently_report_{timestamp}.html"
        report.save_html(str(html_path))
        manual_report["evidently_report_path"] = str(html_path)
        manual_report["evidently_available"] = True
    except Exception:
        logger.exception("Evidently report generation failed")
        manual_report["evidently_available"] = False

    return manual_report
