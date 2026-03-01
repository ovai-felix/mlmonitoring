import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import settings
from src.metrics import baseline_feature_mean, baseline_feature_null_rate, baseline_feature_std
from src.schemas import BaselineStatsResponse, FeatureStats

BASELINE_FILENAME = "baseline_stats.json"


def compute_baseline(df: pd.DataFrame) -> list[FeatureStats]:
    """Compute per-feature baseline statistics."""
    stats = []
    for col in df.columns:
        series = df[col]
        null_rate = float(series.isna().mean())
        clean = series.dropna()

        if len(clean) == 0:
            hist_counts = []
            hist_bins = []
        else:
            counts, bins = np.histogram(clean.values, bins=50)
            hist_counts = counts.tolist()
            hist_bins = bins.tolist()

        stats.append(FeatureStats(
            feature_name=col,
            mean=float(clean.mean()) if len(clean) > 0 else 0.0,
            std=float(clean.std()) if len(clean) > 1 else 0.0,
            min=float(clean.min()) if len(clean) > 0 else 0.0,
            max=float(clean.max()) if len(clean) > 0 else 0.0,
            median=float(clean.median()) if len(clean) > 0 else 0.0,
            null_rate=null_rate,
            histogram_bins=hist_bins,
            histogram_counts=hist_counts,
        ))
    return stats


def save_baseline(
    stats: list[FeatureStats],
    num_records: int,
    version: str | None = None,
    artifacts_dir: Path | None = None,
) -> BaselineStatsResponse:
    """Save baseline statistics to JSON."""
    artifacts = artifacts_dir or settings.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)

    if version is None:
        version = f"v_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    response = BaselineStatsResponse(
        version=version,
        computed_at=datetime.now(timezone.utc).isoformat(),
        num_records=num_records,
        features=stats,
    )

    path = artifacts / BASELINE_FILENAME
    path.write_text(json.dumps(response.model_dump(), indent=2))
    return response


def load_baseline(artifacts_dir: Path | None = None) -> BaselineStatsResponse | None:
    """Load baseline statistics from JSON."""
    artifacts = artifacts_dir or settings.artifacts_dir
    path = artifacts / BASELINE_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return BaselineStatsResponse(**data)


def expose_baseline_to_prometheus(baseline: BaselineStatsResponse):
    """Set Prometheus gauges from baseline statistics."""
    for feat in baseline.features:
        baseline_feature_mean.labels(feature=feat.feature_name).set(feat.mean)
        baseline_feature_std.labels(feature=feat.feature_name).set(feat.std)
        baseline_feature_null_rate.labels(feature=feat.feature_name).set(feat.null_rate)
