import json
import logging
from pathlib import Path

from fastapi import APIRouter

from src.config import settings
from src.services.drift_service import run_drift_detection
from src.services.performance_metrics import compute_rolling_metrics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/drift/run")
async def trigger_drift():
    """Manually trigger drift detection."""
    report = run_drift_detection()
    return report


@router.get("/drift/status")
async def drift_status():
    """Return the latest drift report."""
    report_dir = settings.artifacts_dir / "drift_reports"
    if not report_dir.exists():
        return {"status": "no_reports", "latest": None}

    reports = sorted(report_dir.glob("drift_report_*.json"), reverse=True)
    if not reports:
        return {"status": "no_reports", "latest": None}

    latest = json.loads(reports[0].read_text())
    return {"status": "ok", "latest": latest}


@router.get("/performance")
async def performance():
    """Return current rolling performance metrics."""
    result = compute_rolling_metrics()
    return result
