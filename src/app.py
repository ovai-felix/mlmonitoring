import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import settings
from src.database import init_db
from src.middleware import PrometheusMiddleware
from src.routes.features import router as features_router
from src.routes.ingestion import router as ingestion_router
from src.routes.monitoring import router as monitoring_router
from src.routes.prediction import router as prediction_router
from src.routes.training import router as training_router
from src.services.baseline_stats import expose_baseline_to_prometheus, load_baseline
from src.services.model_manager import model_manager
from src.services.performance_metrics import compute_rolling_metrics

logger = logging.getLogger(__name__)


async def _periodic_performance_update():
    """Background task to update rolling performance metrics periodically."""
    while True:
        await asyncio.sleep(settings.performance_update_seconds)
        try:
            compute_rolling_metrics()
        except Exception:
            logger.exception("Periodic performance metrics update failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create directories, init DB, load baseline
    for d in [settings.data_dir, settings.parquet_dir, settings.features_dir,
              settings.artifacts_dir, settings.raw_data_dir, settings.models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    init_db()

    baseline = load_baseline()
    if baseline is not None:
        expose_baseline_to_prometheus(baseline)

    try:
        model_manager.initial_load()
    except Exception:
        logger.warning("Failed to load models on startup — /ready will return 503", exc_info=True)

    # Start periodic performance metrics updater
    perf_task = asyncio.create_task(_periodic_performance_update())

    yield

    perf_task.cancel()
    try:
        await perf_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="ML Monitoring System",
    description="Production-grade ML monitoring for credit card fraud detection",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(PrometheusMiddleware)

app.include_router(ingestion_router)
app.include_router(features_router)
app.include_router(training_router)
app.include_router(prediction_router)
app.include_router(monitoring_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
