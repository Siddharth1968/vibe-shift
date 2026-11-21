"""
VibeShift - Emotion Analysis Schemas
Defines schemas for AI facial emotion estimation results.
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class EmotionResponse(BaseModel):
    """
    Response schema for image emotion estimation (`POST /api/emotion`).
    """

    emotion: Optional[str] = Field(
        None,
        description="Estimated music mood ('happy', 'sad', 'angry', 'neutral'), or null if unmapped / failed",
    )
    confidence: float = Field(
        0.0,
        description="Estimated confidence score (0.0 to 1.0)",
    )
    note: str = Field(
        ...,
        description="Explanation note, disclaimer, or prompt to choose mood manually",
    )
    status: Optional[str] = Field(
        None,
        description="Detection status from AI model (e.g., success, no_face_detected, low_confidence)",
    )
    expression: Optional[str] = Field(
        None,
        description="Detected 7-class facial expression label (angry, disgust, fear, happy, neutral, sad, surprise)",
    )
    expression_scores: Optional[Dict[str, float]] = Field(
        None,
        description="Full 7-class softmax probability map from CleanBTNVit",
    )
    model_name: Optional[str] = Field(
        None,
        description="Active emotion classification model identifier",
    )
    inference_mode: Optional[str] = Field(
        None,
        description="Inference mode used ('single_view' or 'horizontal_flip')",
    )
    inference_ms: Optional[float] = Field(
        None,
        description="Model-only forward pass latency in milliseconds",
    )
    latency_ms: Optional[float] = Field(
        None,
        description="Total server-side processing latency in milliseconds",
    )
