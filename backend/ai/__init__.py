"""
backend.ai — VibeShift single-model facial-expression recognition runtime.

Production pipeline:
  - YuNetFaceDetector (`backend/ai/models/yunet-2023mar.onnx`)
  - preprocess_face_for_clean_btn (exact Phase 12 112x112 grayscale preprocessing)
  - CleanBTNVitAdapter (`backend/ai/models/clean_btn_vit_phase12.pt`, single-view default + optional horizontal-flip mode)
  - predict (unified end-to-end prediction function)
"""

from backend.ai.emotion_model import (
    CLEAN_BTN_EXPECTED_SHA256,
    CLEAN_BTN_LABELS,
    CLEAN_BTN_MODEL_NAME,
    DISCLAIMER_TEXT,
    EXPRESSION_LABELS,
    CleanBTNVitAdapter,
    get_ai_runtime_status,
    get_default_detector,
    get_default_model,
    initialize_ai_runtime,
    predict,
)
from backend.ai.face_detector import YUNET_EXPECTED_SHA256, YuNetFaceDetector
from backend.ai.preprocess import (
    crop_face,
    decode_image,
    preprocess_face_for_clean_btn,
)

__all__ = [
    "YuNetFaceDetector",
    "YUNET_EXPECTED_SHA256",
    "decode_image",
    "crop_face",
    "preprocess_face_for_clean_btn",
    "CleanBTNVitAdapter",
    "get_default_detector",
    "get_default_model",
    "initialize_ai_runtime",
    "get_ai_runtime_status",
    "predict",
    "CLEAN_BTN_LABELS",
    "CLEAN_BTN_MODEL_NAME",
    "CLEAN_BTN_EXPECTED_SHA256",
    "EXPRESSION_LABELS",
    "DISCLAIMER_TEXT",
]
