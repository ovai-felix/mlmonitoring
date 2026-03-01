import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_id TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    raw_features TEXT NOT NULL,
    transformed_features TEXT,
    prediction_label INTEGER,
    prediction_confidence REAL,
    actual_label INTEGER,
    feedback_received_at TEXT,
    warnings TEXT,
    is_anomalous INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS inference_features (
    entity_id TEXT PRIMARY KEY,
    features TEXT NOT NULL,
    computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'api',
    total_records INTEGER NOT NULL,
    accepted_records INTEGER NOT NULL,
    rejected_records INTEGER NOT NULL,
    warnings_count INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    duration_seconds REAL
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    data_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    metrics TEXT,
    mlflow_run_id TEXT,
    promoted_to TEXT,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_ingestion_id ON predictions(ingestion_id);
CREATE INDEX IF NOT EXISTS idx_predictions_ingested_at ON predictions(ingested_at);
CREATE INDEX IF NOT EXISTS idx_inference_features_computed_at ON inference_features(computed_at);
CREATE INDEX IF NOT EXISTS idx_training_runs_model_type ON training_runs(model_type);
"""


def get_db_path(db_path: Path | None = None) -> Path:
    return db_path or settings.sqlite_db_path


@contextmanager
def get_connection(db_path: Path | None = None):
    path = get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | None = None):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def insert_prediction(
    conn: sqlite3.Connection,
    ingestion_id: str,
    raw_features: dict,
    warnings: list[str] | None = None,
) -> int:
    cursor = conn.execute(
        """INSERT INTO predictions (ingestion_id, raw_features, warnings)
           VALUES (?, ?, ?)""",
        (ingestion_id, json.dumps(raw_features), json.dumps(warnings or [])),
    )
    return cursor.lastrowid


def get_prediction(conn: sqlite3.Connection, prediction_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def update_feedback(conn: sqlite3.Connection, prediction_id: int, actual_label: int) -> bool:
    cursor = conn.execute(
        """UPDATE predictions
           SET actual_label = ?, feedback_received_at = datetime('now')
           WHERE id = ?""",
        (actual_label, prediction_id),
    )
    return cursor.rowcount > 0


def insert_ingestion_log(
    conn: sqlite3.Connection,
    batch_id: str,
    source: str,
    total_records: int,
    accepted_records: int,
    rejected_records: int,
    warnings_count: int,
    duration_seconds: float,
):
    conn.execute(
        """INSERT INTO ingestion_log
           (batch_id, source, total_records, accepted_records, rejected_records, warnings_count, duration_seconds)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (batch_id, source, total_records, accepted_records, rejected_records, warnings_count, duration_seconds),
    )


def upsert_inference_features(conn: sqlite3.Connection, entity_id: str, features: dict):
    conn.execute(
        """INSERT INTO inference_features (entity_id, features)
           VALUES (?, ?)
           ON CONFLICT(entity_id) DO UPDATE SET features = excluded.features, computed_at = datetime('now')""",
        (entity_id, json.dumps(features)),
    )


def get_inference_features(conn: sqlite3.Connection, entity_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM inference_features WHERE entity_id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["features"] = json.loads(result["features"])
    return result


def insert_training_run(
    conn: sqlite3.Connection,
    run_id: str,
    model_type: str,
    data_version: str,
):
    conn.execute(
        """INSERT INTO training_runs (run_id, model_type, data_version, status)
           VALUES (?, ?, ?, 'running')""",
        (run_id, model_type, data_version),
    )


def update_training_run(
    conn: sqlite3.Connection,
    run_id: str,
    status: str,
    metrics: dict | None = None,
    mlflow_run_id: str | None = None,
    promoted_to: str | None = None,
    error_message: str | None = None,
):
    conn.execute(
        """UPDATE training_runs
           SET status = ?, completed_at = datetime('now'),
               metrics = ?, mlflow_run_id = ?, promoted_to = ?, error_message = ?
           WHERE run_id = ?""",
        (status, json.dumps(metrics) if metrics else None,
         mlflow_run_id, promoted_to, error_message, run_id),
    )


def get_last_training_run(
    conn: sqlite3.Connection,
    model_type: str,
) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT * FROM training_runs
           WHERE model_type = ? ORDER BY started_at DESC LIMIT 1""",
        (model_type,),
    ).fetchone()
    if row is None:
        return None
    return dict(row)
