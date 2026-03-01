import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import settings
from src.database import get_connection, upsert_inference_features, get_inference_features


def save_training_features(
    df: pd.DataFrame,
    version: str | None = None,
    features_dir: Path | None = None,
) -> str:
    """Save training features as versioned Parquet."""
    fdir = features_dir or settings.features_dir
    if version is None:
        version = f"v_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    version_dir = fdir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    path = version_dir / "features.parquet"
    df.to_parquet(path, index=False)
    return version


def load_training_features(
    version: str,
    features_dir: Path | None = None,
) -> pd.DataFrame:
    """Load training features for a specific version."""
    fdir = features_dir or settings.features_dir
    path = fdir / version / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Training features version '{version}' not found")
    return pd.read_parquet(path)


def list_versions(features_dir: Path | None = None) -> list[dict]:
    """List all available training feature versions."""
    fdir = features_dir or settings.features_dir
    if not fdir.exists():
        return []
    versions = []
    for d in sorted(fdir.iterdir()):
        if d.is_dir() and (d / "features.parquet").exists():
            parquet_path = d / "features.parquet"
            df = pd.read_parquet(parquet_path)
            versions.append({
                "version": d.name,
                "num_records": len(df),
                "num_features": len(df.columns),
                "feature_names": list(df.columns),
                "created_at": datetime.fromtimestamp(
                    parquet_path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            })
    return versions


def save_inference_features(
    entity_id: str,
    features: dict,
    db_path: Path | None = None,
):
    """Save/update inference features for an entity."""
    with get_connection(db_path) as conn:
        upsert_inference_features(conn, entity_id, features)


def get_entity_inference_features(
    entity_id: str,
    db_path: Path | None = None,
) -> dict | None:
    """Get cached inference features for an entity."""
    with get_connection(db_path) as conn:
        return get_inference_features(conn, entity_id)
