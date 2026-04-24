"""Health and readiness endpoints."""
import logging
import shutil

from fastapi import APIRouter, HTTPException

import engine
from config import get_app_version

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": get_app_version(),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }


@router.get("/ready")
def readiness_check():
    if engine.MODEL_LOADED:
        return {"status": "ready", "model_loaded": True}
    raise HTTPException(
        status_code=503,
        detail="TTS model is not loaded; synthesis endpoints will return 503.",
    )
