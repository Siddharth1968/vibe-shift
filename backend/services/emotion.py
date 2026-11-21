"""
VibeShift - Emotion Analysis Service
Handles facial expression to music mood mapping and result processing.
Integrates with backend.ai.predict output.
"""

from typing import Dict, Any, Optional

# Mapping of facial expressions to VibeShift music moods.
# Only unambiguous expressions are automatically mapped.
# Surprise, fear, and disgust have NO automatic mood.
EXPRESSION_TO_MOOD: Dict[str, str] = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "neutral": "neutral",
}

# Standard disclaimer note
DISCLAIMER_NOTE = (
    "This is only an estimate of facial expression, not proof of feelings."
)

# Prompt appended when automated detection cannot provide a confident mood
MANUAL_CHOICE_PROMPT = "Please choose your mood manually."

# Allowed image MIME types and extensions for validation
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _metadata_fields(raw_result: Dict[str, Any], expression_clean: Optional[str]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if expression_clean:
        fields["expression"] = expression_clean
    if raw_result.get("expression_scores") is not None:
        fields["expression_scores"] = raw_result["expression_scores"]
    if raw_result.get("model_name") is not None:
        fields["model_name"] = raw_result["model_name"]
    if raw_result.get("inference_mode") is not None:
        fields["inference_mode"] = raw_result["inference_mode"]
    if raw_result.get("inference_ms") is not None:
        fields["inference_ms"] = raw_result["inference_ms"]
    if raw_result.get("latency_ms") is not None:
        fields["latency_ms"] = raw_result["latency_ms"]
    return fields


def process_prediction_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms the raw dictionary from backend.ai.predict into the VibeShift API contract:
    - If status is not 'success', returns emotion=None (except low_confidence mapped suggestions)
      and asks user to choose/confirm mood manually.
    - If expression is surprise, fear, or disgust (or unmapped), returns emotion=None and asks
      user to choose mood manually. Never returns neutral silently.
    - Returns emotion, confidence, note, status, plus optional expression/score/latency diagnostics.
    """
    status = str(raw_result.get("status", "unknown")).lower()
    expression = raw_result.get("expression")
    expression_clean = str(expression).strip().lower() if expression else ""
    raw_confidence = raw_result.get("confidence")
    confidence = round(float(raw_confidence), 2) if raw_confidence is not None else 0.0
    extra = _metadata_fields(raw_result, expression_clean or None)

    # 1. Handle non-success statuses from emotion_model.py
    if status != "success":
        mapped_mood = EXPRESSION_TO_MOOD.get(expression_clean) if expression_clean else None

        if status == "low_confidence":
            if mapped_mood:
                note = (
                    f"Facial expression confidence is low ({int(confidence * 100)}%). "
                    f"AI suggests '{mapped_mood}'. Please confirm or choose your mood manually."
                )
                return {
                    "emotion": mapped_mood,
                    "confidence": confidence,
                    "status": "low_confidence",
                    "note": note,
                    **extra,
                }
            elif expression_clean:
                note = (
                    f"Detected facial expression is '{expression_clean}', which has no direct music mood mapping. "
                    f"{MANUAL_CHOICE_PROMPT}"
                )
            else:
                note = f"Facial expression confidence is low. {MANUAL_CHOICE_PROMPT}"
        elif status == "no_face_detected":
            note = f"No face was detected in the photo. {MANUAL_CHOICE_PROMPT}"
        elif status.startswith("model_weights_not_loaded"):
            note = f"AI model weights are currently not loaded. {MANUAL_CHOICE_PROMPT}"
        elif status == "invalid_image":
            note = f"The uploaded image could not be processed. {MANUAL_CHOICE_PROMPT}"
        elif status == "face_detector_unavailable":
            note = f"Face detector is currently unavailable. {MANUAL_CHOICE_PROMPT}"
        elif status == "preprocessing_failed":
            note = f"Could not process facial region. {MANUAL_CHOICE_PROMPT}"
        else:
            note = f"Could not determine facial expression. {MANUAL_CHOICE_PROMPT}"

        return {
            "emotion": None,
            "confidence": confidence,
            "status": status,
            "note": note,
            **extra,
        }

    # 2. Status is 'success', evaluate the detected expression
    if not expression_clean:
        return {
            "emotion": None,
            "confidence": confidence,
            "status": status,
            "note": f"No expression was identified. {MANUAL_CHOICE_PROMPT}",
            **extra,
        }

    # 3. Map expression to mood
    if expression_clean in EXPRESSION_TO_MOOD:
        mapped_mood = EXPRESSION_TO_MOOD[expression_clean]
        return {
            "emotion": mapped_mood,
            "confidence": confidence,
            "status": status,
            "note": DISCLAIMER_NOTE,
            **extra,
        }

    # Surprise, fear, disgust, or any other unmapped expression:
    # Must return emotion = null and prompt user to choose mood manually.
    # NEVER return neutral silently.
    return {
        "emotion": None,
        "confidence": confidence,
        "status": status,
        "note": (
            f"Detected facial expression is '{expression_clean}', which has no direct music mood mapping. "
            f"{MANUAL_CHOICE_PROMPT}"
        ),
        **extra,
    }
