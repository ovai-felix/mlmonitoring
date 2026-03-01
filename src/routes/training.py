import json

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from src.metrics import retrain_triggered_total
from src.services.training_service import (
    check_cooldown,
    train_anomaly,
    train_classifier,
    train_lstm,
)

router = APIRouter(prefix="/training", tags=["training"])


class TrainingRequest(BaseModel):
    model_type: str
    data_version: str
    n_trials: int = Field(default=5, ge=1, le=20)
    window_size: int = Field(default=32, ge=8, le=96)
    reason: str = Field(default="manual", pattern="^(drift|accuracy_drop|manual)$")


class TrainingResponse(BaseModel):
    status: str
    run_id: Optional[str] = None
    message: str


@router.post("/trigger", response_model=TrainingResponse)
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Trigger model training as a background task."""
    if request.model_type not in ("classification", "timeseries", "anomaly"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model type: {request.model_type}. Use: classification, timeseries, anomaly",
        )

    if not check_cooldown(request.model_type):
        return TrainingResponse(
            status="rejected",
            message=f"Training cooldown active for {request.model_type}. Try again later.",
        )

    retrain_triggered_total.labels(reason=request.reason).inc()

    if request.model_type == "classification":
        background_tasks.add_task(train_classifier, request.data_version, request.n_trials)
    elif request.model_type == "timeseries":
        background_tasks.add_task(
            train_lstm, request.data_version, request.window_size, request.n_trials,
        )
    elif request.model_type == "anomaly":
        background_tasks.add_task(train_anomaly, request.data_version)

    return TrainingResponse(
        status="accepted",
        message=f"Training for {request.model_type} started in background",
    )


@router.get("/status")
async def get_training_status(model_type: Optional[str] = None):
    """Get status of recent training runs."""
    from src.database import get_connection

    with get_connection() as conn:
        if model_type:
            rows = conn.execute(
                "SELECT * FROM training_runs WHERE model_type = ? ORDER BY started_at DESC LIMIT 10",
                (model_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT 10",
            ).fetchall()

        results = []
        for row in rows:
            d = dict(row)
            if d.get("metrics"):
                d["metrics"] = json.loads(d["metrics"])
            results.append(d)
        return results
