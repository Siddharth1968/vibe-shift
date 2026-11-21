"""
VibeShift - Emotion API Routes
Accepts optional image uploads or live webcam frames to estimate facial expression emotion
using the single startup-loaded YuNet + Phase 12 CleanBTNVit model.
Images are processed purely in-memory and never saved to disk.
"""

import json
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from backend.ai import predict
from backend.ai.live_pipeline import get_active_model_info, predict_live
from backend.schemas.emotion import EmotionResponse
from backend.services.emotion import (
    ALLOWED_EXTENSIONS,
    ALLOWED_IMAGE_TYPES,
    process_prediction_result,
)

router = APIRouter(prefix="/api/emotion", tags=["Emotion"])

MAX_SNAPSHOT_BYTES = 5 * 1024 * 1024  # 5 MB (matches frontend MAX_UPLOAD_BYTES)
MAX_LIVE_FRAME_BYTES = 2 * 1024 * 1024  # 2 MB for continuous webcam frames


@router.get("/model", summary="Inspect active emotion recognition model")
async def active_model_info():
    """Returns read-only metadata for the single active Phase 12 CleanBTNVit model."""
    return await run_in_threadpool(get_active_model_info)


@router.post("/live", summary="Real-time webcam or snapshot emotion estimation")
async def live_emotion(
    file: UploadFile = File(...),
    session: str = Form("single-image-session"),
    selection: str = Form("null"),
    snapshot: bool = Form(False),
    debug: bool = Form(False),
    horizontal_flip: Optional[bool] = Form(None),
):
    """
    POST /api/emotion/live
    Runs YuNet face detection + single-model Phase 12 CleanBTNVit inference in-memory.
    Defaults to fast single-view inference; set `horizontal_flip=true` for optional
    horizontal-flip accuracy mode.
    """
    if not 16 <= len(session) <= 64:
        raise HTTPException(status_code=400, detail="Invalid camera session")

    max_bytes = MAX_SNAPSHOT_BYTES if snapshot else MAX_LIVE_FRAME_BYTES
    image = await file.read(max_bytes + 1)
    if not image:
        raise HTTPException(status_code=400, detail="Uploaded image frame is empty.")
    if len(image) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds the {limit_mb} MB upload limit.",
        )

    try:
        point = json.loads(selection)
        if point is not None and (
            not isinstance(point, list)
            or len(point) != 2
            or any(not isinstance(x, (int, float)) for x in point)
        ):
            raise ValueError("Invalid face selection")
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid face selection")

    return await run_in_threadpool(
        predict_live,
        image,
        session,
        point,
        snapshot,
        debug,
        horizontal_flip,
    )


@router.post(
    "",
    response_model=EmotionResponse,
    summary="Estimate emotion from facial photo",
    description=(
        "Analyzes an uploaded facial photo in-memory using YuNet + Phase 12 CleanBTNVit. "
        "Images are never saved to disk. Returns null emotion if detection fails or expression is unmapped."
    ),
)
async def estimate_emotion(
    file: UploadFile = File(...),
    horizontal_flip: Optional[bool] = Form(None),
):
    """
    POST /api/emotion
    Accepts an uploaded image file, validates format and size (<= 5 MB),
    runs backend.ai.predict via run_in_threadpool in-memory,
    maps facial expression to music mood, and returns the validated response.
    """
    filename = file.filename or ""
    _, ext = os.path.splitext(filename.lower())

    is_valid_mime = file.content_type and file.content_type.lower() in ALLOWED_IMAGE_TYPES
    is_valid_ext = ext in ALLOWED_EXTENSIONS

    if not is_valid_mime and not is_valid_ext:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid image file (JPEG, PNG, or WEBP).",
        )

    try:
        image_bytes = await file.read(MAX_SNAPSHOT_BYTES + 1)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {str(exc)}",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty. Please upload a valid image.",
        )

    if len(image_bytes) > MAX_SNAPSHOT_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image size exceeds the 5 MB upload limit.",
        )

    try:
        raw_result = await run_in_threadpool(
            predict,
            image_bytes,
            horizontal_flip=horizontal_flip,
        )
    except Exception:
        return EmotionResponse(
            emotion=None,
            confidence=0.0,
            status="error",
            note="Emotion estimation encountered a temporary issue. Please choose your mood manually.",
        )

    formatted_response = process_prediction_result(raw_result)
    return EmotionResponse(**formatted_response)
