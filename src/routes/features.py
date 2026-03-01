from fastapi import APIRouter, HTTPException

from src.schemas import InferenceFeatureResponse, TrainingFeaturesResponse
from src.services.feature_store import (
    get_entity_inference_features,
    list_versions,
    load_training_features,
)

router = APIRouter()


@router.get("/features/training", response_model=TrainingFeaturesResponse)
async def get_training_features(version: str | None = None):
    """Get metadata about training features for a specific version."""
    versions = list_versions()
    if not versions:
        raise HTTPException(status_code=404, detail="No training features available")

    if version is None:
        info = versions[-1]  # latest
    else:
        info = next((v for v in versions if v["version"] == version), None)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Version '{version}' not found")

    return TrainingFeaturesResponse(**info)


@router.get("/features/training/versions")
async def get_training_versions():
    """List all available training feature versions."""
    return list_versions()


@router.get("/features/inference/{entity_id}", response_model=InferenceFeatureResponse)
async def get_inference_features(entity_id: str):
    """Get cached inference features for an entity."""
    result = get_entity_inference_features(entity_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_id}' not found")
    return InferenceFeatureResponse(
        entity_id=result["entity_id"],
        features=result["features"],
        computed_at=result["computed_at"],
    )
