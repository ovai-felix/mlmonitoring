import pytest
from pydantic import ValidationError

from src.schemas import TransactionRecord, TransactionBatch, FeedbackRecord


class TestTransactionRecord:
    def test_valid_record(self, sample_record):
        record = TransactionRecord(**sample_record)
        assert record.Time == 0.0
        assert record.Amount == 149.62
        assert record.Class == 0
        assert record.get_warnings() == []

    def test_optional_class(self, sample_record):
        sample_record.pop("Class")
        record = TransactionRecord(**sample_record)
        assert record.Class is None

    def test_out_of_range_warning_amount(self, sample_record):
        sample_record["Amount"] = 50000.0
        record = TransactionRecord(**sample_record)
        warnings = record.get_warnings()
        assert len(warnings) >= 1
        assert any("Amount" in w for w in warnings)

    def test_out_of_range_warning_pca(self, sample_record):
        sample_record["V1"] = -100.0
        record = TransactionRecord(**sample_record)
        warnings = record.get_warnings()
        assert any("V1" in w for w in warnings)

    def test_missing_required_field(self, sample_record):
        del sample_record["V1"]
        with pytest.raises(ValidationError):
            TransactionRecord(**sample_record)

    def test_invalid_type(self, sample_record):
        sample_record["V1"] = "not_a_number"
        with pytest.raises(ValidationError):
            TransactionRecord(**sample_record)

    def test_to_feature_dict(self, sample_record):
        record = TransactionRecord(**sample_record)
        d = record.to_feature_dict()
        assert "Class" not in d
        assert "V1" in d
        assert "Amount" in d


class TestTransactionBatch:
    def test_valid_batch(self, sample_records):
        batch = TransactionBatch(records=[TransactionRecord(**r) for r in sample_records])
        assert len(batch.records) == 10
        assert batch.source == "api"


class TestFeedbackRecord:
    def test_valid_feedback(self):
        fb = FeedbackRecord(prediction_id=1, actual_label=0)
        assert fb.prediction_id == 1

    def test_invalid_label(self):
        with pytest.raises(ValidationError):
            FeedbackRecord(prediction_id=1, actual_label=2)
