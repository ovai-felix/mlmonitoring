import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from src.services.mlflow_service import get_production_model_metrics, promote_model


def evaluate_classification(model, data_loader, device) -> dict:
    """Evaluate a classification model. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    auc = 0.0
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "auc_roc": float(auc),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


def evaluate_anomaly(detector, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate Isolation Forest on test data with known anomalies."""
    pred_labels = detector.predict_labels(X_test)
    # IsolationForest: -1 = anomaly, 1 = normal
    # Convert to match y_test: 1 = fraud, 0 = normal
    pred_binary = (pred_labels == -1).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, pred_binary)),
        "f1": float(f1_score(y_test, pred_binary, zero_division=0)),
        "precision": float(precision_score(y_test, pred_binary, zero_division=0)),
        "recall": float(recall_score(y_test, pred_binary, zero_division=0)),
        "n_anomalies_detected": int(pred_binary.sum()),
        "n_actual_anomalies": int(y_test.sum()),
    }


def check_promotion_gate(
    model_type: str,
    new_metrics: dict,
    model_name: str,
    model_version: int,
) -> dict:
    """Check if a new model passes the evaluation gate for promotion."""
    if model_type == "anomaly":
        return {
            "decision": "manual_review",
            "reason": "Anomaly models require manual review",
            "promoted_to": None,
        }

    prod_metrics = get_production_model_metrics(model_name)

    if prod_metrics is None:
        promote_model(model_name, model_version, "Staging")
        return {
            "decision": "promoted",
            "reason": "No existing production model",
            "promoted_to": "Staging",
        }

    prod_f1 = prod_metrics.get("f1", 0)
    new_f1 = new_metrics.get("f1", 0)
    tolerance = 0.01
    threshold = prod_f1 - tolerance

    if new_f1 >= threshold:
        promote_model(model_name, model_version, "Staging")
        return {
            "decision": "promoted",
            "reason": f"New F1 ({new_f1:.4f}) >= threshold ({threshold:.4f})",
            "promoted_to": "Staging",
        }
    else:
        return {
            "decision": "rejected",
            "reason": f"New F1 ({new_f1:.4f}) < threshold ({threshold:.4f})",
            "promoted_to": None,
        }
