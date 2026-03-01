from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.schemas import TransactionRecord
from src.services.model_manager import model_manager
from src.services.prediction_service import predict_single, predict_batch

router = APIRouter()


class PredictionResponse(BaseModel):
    prediction_id: str
    label: int | None
    confidence: float | None
    anomaly_score: float | None
    is_anomalous: bool
    model_version: str
    warnings: list[str]
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    records: list[TransactionRecord]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


@router.post("/predict", response_model=PredictionResponse)
async def predict(record: TransactionRecord):
    if not model_manager.active.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        result = predict_single(record.to_feature_dict(), record.get_warnings())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_endpoint(batch: BatchPredictionRequest):
    if not model_manager.active.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        records_with_warnings = [
            (r.to_feature_dict(), r.get_warnings()) for r in batch.records
        ]
        results = predict_batch(records_with_warnings)
        return {"predictions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ready")
async def ready():
    if not model_manager.active.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {"status": "ready", "active_color": model_manager._active_color}


@router.get("/model/info")
async def model_info():
    return model_manager.get_info()


@router.post("/model/reload")
async def reload():
    result = model_manager.reload()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["detail"])
    return result


@router.post("/model/rollback")
async def rollback():
    result = model_manager.rollback()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["detail"])
    return result
