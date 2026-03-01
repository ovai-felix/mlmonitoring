import logging
from pathlib import Path

from src.database import get_connection
from src.metrics import (
    false_positive_rate,
    fraud_detection_rate,
    rolling_accuracy,
    rolling_f1,
    rolling_false_negatives,
    rolling_false_positives,
    rolling_precision,
    rolling_recall,
    rolling_true_negatives,
    rolling_true_positives,
)

logger = logging.getLogger(__name__)

WINDOWS = [100, 1000, 10000]


def compute_rolling_metrics(db_path: Path | None = None) -> dict:
    """Compute rolling performance metrics from predictions with feedback.

    Queries the predictions table for rows where actual_label IS NOT NULL,
    computes metrics for each window size, and updates Prometheus gauges.

    Returns a dict with per-window metrics and overall fraud_detection_rate.
    """
    result = {"windows": {}, "fraud_detection_rate": None}

    with get_connection(db_path) as conn:
        # Get total feedback count
        total_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM predictions WHERE actual_label IS NOT NULL"
        ).fetchone()
        total_with_feedback = total_row["cnt"]

        if total_with_feedback == 0:
            return result

        # Overall fraud detection rate: TP / (TP + FN) across all feedback
        all_rows = conn.execute(
            """SELECT prediction_label, actual_label FROM predictions
               WHERE actual_label IS NOT NULL
               ORDER BY id DESC"""
        ).fetchall()

        total_actual_fraud = sum(1 for r in all_rows if r["actual_label"] == 1)
        total_detected = sum(
            1 for r in all_rows
            if r["actual_label"] == 1 and r["prediction_label"] == 1
        )
        fdr = total_detected / total_actual_fraud if total_actual_fraud > 0 else 0.0
        result["fraud_detection_rate"] = fdr
        fraud_detection_rate.set(fdr)

        for window in WINDOWS:
            rows = conn.execute(
                """SELECT prediction_label, actual_label FROM predictions
                   WHERE actual_label IS NOT NULL
                   ORDER BY id DESC LIMIT ?""",
                (window,),
            ).fetchall()

            if not rows:
                continue

            tp = sum(1 for r in rows if r["prediction_label"] == 1 and r["actual_label"] == 1)
            fp = sum(1 for r in rows if r["prediction_label"] == 1 and r["actual_label"] == 0)
            tn = sum(1 for r in rows if r["prediction_label"] == 0 and r["actual_label"] == 0)
            fn = sum(1 for r in rows if r["prediction_label"] == 0 and r["actual_label"] == 1)

            total = tp + fp + tn + fn
            accuracy = (tp + tn) / total if total > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            ws = str(window)
            rolling_accuracy.labels(window_size=ws).set(accuracy)
            rolling_precision.labels(window_size=ws).set(precision)
            rolling_recall.labels(window_size=ws).set(recall)
            rolling_f1.labels(window_size=ws).set(f1)
            rolling_true_positives.labels(window_size=ws).set(tp)
            rolling_false_positives.labels(window_size=ws).set(fp)
            rolling_true_negatives.labels(window_size=ws).set(tn)
            rolling_false_negatives.labels(window_size=ws).set(fn)
            false_positive_rate.labels(window_size=ws).set(fpr)

            result["windows"][window] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "fpr": fpr,
                "count": len(rows),
            }

    return result
