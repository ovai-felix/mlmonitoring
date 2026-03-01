import pytest

from src.database import init_db, get_connection
from src.services.ingestion_service import ingest_batch, process_feedback
from src.schemas import FeedbackRecord


class TestIngestion:
    def test_batch_ingest(self, sample_records, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        response = ingest_batch(sample_records, source="test", db_path=tmp_data_dir["db_path"])
        assert response.total_records == 10
        assert response.accepted == 10
        assert response.rejected == 0
        assert all(r.accepted for r in response.results)
        assert all(r.prediction_id is not None for r in response.results)

    def test_batch_with_invalid_records(self, sample_records, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        # Add an invalid record
        sample_records.append({"Time": "bad"})
        response = ingest_batch(sample_records, source="test", db_path=tmp_data_dir["db_path"])
        assert response.accepted == 10
        assert response.rejected == 1

    def test_parquet_written(self, sample_records, tmp_data_dir, monkeypatch):
        from src.config import settings
        monkeypatch.setattr(settings, "parquet_dir", tmp_data_dir["parquet_dir"])
        init_db(tmp_data_dir["db_path"])
        ingest_batch(sample_records, source="test", db_path=tmp_data_dir["db_path"])
        # Check that parquet files were written
        parquet_files = list(tmp_data_dir["parquet_dir"].rglob("*.parquet"))
        assert len(parquet_files) == 1

    def test_sqlite_records(self, sample_records, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        ingest_batch(sample_records, source="test", db_path=tmp_data_dir["db_path"])
        with get_connection(tmp_data_dir["db_path"]) as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            assert count == 10

    def test_feedback_success(self, sample_records, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        response = ingest_batch(sample_records, source="test", db_path=tmp_data_dir["db_path"])
        pred_id = response.results[0].prediction_id

        fb = FeedbackRecord(prediction_id=pred_id, actual_label=1)
        result = process_feedback(fb, db_path=tmp_data_dir["db_path"])
        assert result["matched"] is True

        with get_connection(tmp_data_dir["db_path"]) as conn:
            row = conn.execute("SELECT actual_label FROM predictions WHERE id = ?", (pred_id,)).fetchone()
            assert row[0] == 1

    def test_feedback_not_found(self, tmp_data_dir):
        init_db(tmp_data_dir["db_path"])
        fb = FeedbackRecord(prediction_id=99999, actual_label=0)
        result = process_feedback(fb, db_path=tmp_data_dir["db_path"])
        assert result["matched"] is False


class TestAPIIngestion:
    def test_stream_ingest(self, test_client, sample_record):
        response = test_client.post("/data/ingest/stream", json=sample_record)
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] is True
        assert data["prediction_id"] is not None

    def test_batch_ingest_api(self, test_client, sample_records):
        response = test_client.post(
            "/data/ingest",
            json={"records": sample_records, "source": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["accepted"] == 10

    def test_feedback_api(self, test_client, sample_record):
        # First ingest
        resp = test_client.post("/data/ingest/stream", json=sample_record)
        pred_id = resp.json()["prediction_id"]

        # Then feedback
        resp = test_client.post("/feedback", json={"prediction_id": pred_id, "actual_label": 0})
        assert resp.status_code == 200

    def test_feedback_not_found_api(self, test_client):
        resp = test_client.post("/feedback", json={"prediction_id": 99999, "actual_label": 0})
        assert resp.status_code == 404

    def test_health(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
