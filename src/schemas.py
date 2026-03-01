from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# Reasonable ranges for key features (PCA features typically within [-50, 50])
FEATURE_RANGES = {
    "Time": (0, 200000),
    "Amount": (0, 30000),
}
PCA_RANGE = (-60, 60)
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]


class TransactionRecord(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float
    Class: Optional[int] = None

    _warnings: list[str] = []

    @model_validator(mode="after")
    def check_ranges(self):
        warnings = []
        data = self.model_dump()
        for feat, (lo, hi) in FEATURE_RANGES.items():
            val = data.get(feat)
            if val is not None and (val < lo or val > hi):
                warnings.append(f"{feat}={val} outside expected range [{lo}, {hi}]")
        for feat in PCA_FEATURES:
            val = data.get(feat)
            if val is not None and (val < PCA_RANGE[0] or val > PCA_RANGE[1]):
                warnings.append(f"{feat}={val} outside expected range [{PCA_RANGE[0]}, {PCA_RANGE[1]}]")
        object.__setattr__(self, "_warnings", warnings)
        return self

    def get_warnings(self) -> list[str]:
        return self._warnings

    def to_feature_dict(self) -> dict:
        d = self.model_dump()
        d.pop("Class", None)
        return d


class TransactionBatch(BaseModel):
    records: list[TransactionRecord]
    source: str = "api"


class FeedbackRecord(BaseModel):
    prediction_id: int
    actual_label: int = Field(ge=0, le=1)


class ValidationResult(BaseModel):
    record_index: int
    accepted: bool
    prediction_id: Optional[int] = None
    warnings: list[str] = []
    errors: list[str] = []


class IngestionResponse(BaseModel):
    batch_id: str
    total_records: int
    accepted: int
    rejected: int
    warnings_count: int
    results: list[ValidationResult]


class FeatureStats(BaseModel):
    feature_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    null_rate: float
    histogram_bins: list[float]
    histogram_counts: list[int]


class BaselineStatsResponse(BaseModel):
    version: str
    computed_at: str
    num_records: int
    features: list[FeatureStats]


class TrainingFeaturesResponse(BaseModel):
    version: str
    num_records: int
    num_features: int
    feature_names: list[str]
    created_at: str


class InferenceFeatureResponse(BaseModel):
    entity_id: str
    features: dict
    computed_at: str
