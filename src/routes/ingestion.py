import csv
import io

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from src.schemas import (
    FeedbackRecord,
    IngestionResponse,
    TransactionBatch,
    TransactionRecord,
    ValidationResult,
)
from src.services.ingestion_service import ingest_batch, process_feedback

router = APIRouter()


@router.post("/data/ingest", response_model=IngestionResponse)
async def ingest_data(request: Request):
    """Ingest a batch of transaction records via JSON or CSV upload."""
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided in multipart upload")
        content = await file.read()
        text = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        records = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            records.append(parsed)
        source = "csv_upload"
    elif "application/json" in content_type or content_type == "":
        body = await request.json()
        batch = TransactionBatch(**body)
        records = [r.model_dump() for r in batch.records]
        source = batch.source
    else:
        raise HTTPException(status_code=400, detail="Provide either JSON batch or CSV file")

    return ingest_batch(records, source=source)


@router.post("/data/ingest/stream", response_model=ValidationResult)
async def ingest_stream(record: TransactionRecord):
    """Ingest a single transaction record."""
    response = ingest_batch([record.model_dump()], source="stream")
    return response.results[0]


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRecord):
    """Submit ground-truth feedback for a prediction."""
    result = process_feedback(feedback)
    if not result["matched"]:
        raise HTTPException(status_code=404, detail=result["error"])
    return result
