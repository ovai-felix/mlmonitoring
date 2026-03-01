import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.config import settings
from src.database import (
    get_connection,
    insert_ingestion_log,
    insert_prediction,
    get_prediction,
    update_feedback,
)
from src.metrics import (
    feedback_match_errors,
    feedback_received,
    ingestion_latency,
    records_ingested,
    validation_failures,
    validation_warnings,
)
from src.schemas import (
    FeedbackRecord,
    IngestionResponse,
    TransactionRecord,
    ValidationResult,
)


def validate_record(data: dict, index: int) -> tuple[TransactionRecord | None, ValidationResult]:
    try:
        record = TransactionRecord(**data)
        warnings = record.get_warnings()
        for w in warnings:
            feat_name = w.split("=")[0] if "=" in w else "unknown"
            validation_warnings.labels(feature=feat_name).inc()
        return record, ValidationResult(
            record_index=index,
            accepted=True,
            warnings=warnings,
        )
    except ValidationError as e:
        for err in e.errors():
            validation_failures.labels(error_type=err["type"]).inc()
        return None, ValidationResult(
            record_index=index,
            accepted=False,
            errors=[str(err["msg"]) for err in e.errors()],
        )


def ingest_batch(
    records: list[dict],
    source: str = "api",
    db_path: Path | None = None,
) -> IngestionResponse:
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    results: list[ValidationResult] = []
    accepted_records: list[tuple[TransactionRecord, int]] = []

    for i, raw in enumerate(records):
        record, result = validate_record(raw, i)
        if record is not None:
            accepted_records.append((record, i))
        results.append(result)

    # Write to SQLite
    with get_connection(db_path) as conn:
        for record, idx in accepted_records:
            pred_id = insert_prediction(
                conn,
                ingestion_id=batch_id,
                raw_features=record.to_feature_dict(),
                warnings=record.get_warnings(),
            )
            results[idx].prediction_id = pred_id
            records_ingested.labels(source=source, status="accepted").inc()

        rejected = len(records) - len(accepted_records)
        if rejected > 0:
            records_ingested.labels(source=source, status="rejected").inc(rejected)

        duration = time.time() - start_time
        insert_ingestion_log(
            conn,
            batch_id=batch_id,
            source=source,
            total_records=len(records),
            accepted_records=len(accepted_records),
            rejected_records=rejected,
            warnings_count=sum(len(r.warnings) for r in results),
            duration_seconds=duration,
        )

    # Write accepted records to Parquet
    if accepted_records:
        _write_parquet(accepted_records, batch_id)

    ingestion_latency.labels(source=source).observe(time.time() - start_time)

    return IngestionResponse(
        batch_id=batch_id,
        total_records=len(records),
        accepted=len(accepted_records),
        rejected=rejected,
        warnings_count=sum(len(r.warnings) for r in results),
        results=results,
    )


def _write_parquet(
    accepted: list[tuple[TransactionRecord, int]],
    batch_id: str,
):
    rows = []
    for record, _ in accepted:
        d = record.model_dump()
        d["batch_id"] = batch_id
        rows.append(d)

    df = pd.DataFrame(rows)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    partition_dir = settings.parquet_dir / f"date={today}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / f"{batch_id}.parquet"
    df.to_parquet(out_path, index=False)


def process_feedback(
    feedback: FeedbackRecord,
    db_path: Path | None = None,
) -> dict:
    with get_connection(db_path) as conn:
        prediction = get_prediction(conn, feedback.prediction_id)
        if prediction is None:
            feedback_match_errors.inc()
            return {"matched": False, "error": "prediction_id not found"}

        update_feedback(conn, feedback.prediction_id, feedback.actual_label)
        feedback_received.inc()

    # Update rolling performance metrics after feedback
    try:
        from src.services.performance_metrics import compute_rolling_metrics
        compute_rolling_metrics(db_path)
    except Exception:
        pass  # Don't fail feedback on metrics update error

    return {"matched": True, "prediction_id": feedback.prediction_id}
