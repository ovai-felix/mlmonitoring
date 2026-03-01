import json
import logging
import time
import uuid

import numpy as np
import pandas as pd
import torch

from src.database import get_connection
from src.metrics import predictions_total, anomalies_detected, prediction_latency, prediction_confidence
from src.services.feature_engineering import ALL_INPUT_FEATURES, get_output_feature_names
from src.services.model_manager import model_manager

logger = logging.getLogger(__name__)


def predict_single(raw_features: dict, warnings: list[str]) -> dict:
    start = time.time()
    slot = model_manager.active

    if not slot.is_ready:
        raise RuntimeError("Models not loaded")

    prediction_id = str(uuid.uuid4())
    result = {
        "prediction_id": prediction_id,
        "label": None,
        "confidence": None,
        "anomaly_score": None,
        "is_anomalous": False,
        "model_version": slot.version_tag,
        "warnings": list(warnings),
    }

    # Build DataFrame and transform
    df = pd.DataFrame([raw_features])
    if slot.pipeline is None:
        raise RuntimeError("Feature pipeline not loaded")

    transformed = slot.pipeline.transform(df[ALL_INPUT_FEATURES].values)
    transformed_flat = transformed[0] if transformed.ndim > 1 else transformed

    # Anomaly check
    if slot.anomaly is not None:
        score = slot.anomaly.predict(transformed.reshape(1, -1))[0]
        labels = slot.anomaly.predict_labels(transformed.reshape(1, -1))
        is_anomalous = bool(labels[0] == -1)
        result["anomaly_score"] = float(score)
        result["is_anomalous"] = is_anomalous
        if is_anomalous:
            anomalies_detected.inc()

    # Classification
    if slot.classifier is not None:
        X_tensor = torch.tensor(transformed, dtype=torch.float32, device=slot.device)
        if X_tensor.dim() == 1:
            X_tensor = X_tensor.unsqueeze(0)
        with torch.no_grad():
            logits = slot.classifier(X_tensor)
        prob = torch.sigmoid(logits).item()
        label = 1 if prob >= 0.5 else 0
        result["label"] = label
        result["confidence"] = float(prob) if label == 1 else float(1 - prob)
        predictions_total.labels(label=str(label)).inc()
        prediction_confidence.labels(label=str(label)).observe(result["confidence"])

    # Log to database
    try:
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (ingestion_id, raw_features, transformed_features,
                    prediction_label, prediction_confidence, is_anomalous, warnings)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    json.dumps(raw_features),
                    json.dumps(transformed_flat.tolist() if hasattr(transformed_flat, 'tolist') else list(transformed_flat)),
                    result["label"],
                    result["confidence"],
                    int(result["is_anomalous"]),
                    json.dumps(warnings),
                ),
            )
    except Exception:
        logger.exception("Failed to log prediction to database")

    elapsed = time.time() - start
    result["latency_ms"] = round(elapsed * 1000, 2)
    prediction_latency.observe(elapsed)
    return result


def predict_batch(records_with_warnings: list[tuple[dict, list[str]]]) -> list[dict]:
    results = []
    for raw_features, warnings in records_with_warnings:
        results.append(predict_single(raw_features, warnings))
    return results
