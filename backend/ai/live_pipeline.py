"""
Production live-webcam & snapshot pipeline for VibeShift.

Unifies continuous webcam tracking, multi-face selection, and photo-upload analysis
on the single startup-loaded YuNetFaceDetector + Phase 12 CleanBTNVitAdapter.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ai.emotion_model import (
    CLEAN_BTN_LABELS,
    CLEAN_BTN_MODEL_NAME,
    DEFAULT_CONFIDENCE_THRESHOLD,
    get_ai_runtime_status,
    get_default_detector,
    get_default_model,
)
from backend.ai.preprocess import crop_face, decode_image, preprocess_face_for_clean_btn
from backend.services.emotion import process_prediction_result

_SESSION_TTL_SECONDS = 300.0
_REACQUIRE_GRACE_SECONDS = 0.8
_EMA_ALPHA = 0.5
_MAX_SESSIONS = 128


@dataclass
class _LiveSessionState:
    last_seen_ts: float = field(default_factory=time.monotonic)
    tracked_box: Optional[Tuple[int, int, int, int]] = None
    tracked_lost_ts: Optional[float] = None
    smoothed_probs: Optional[np.ndarray] = None
    last_top_expression: Optional[str] = None
    consistent_frames: int = 0


_SESSION_LOCK = threading.Lock()
_SESSIONS: Dict[str, _LiveSessionState] = {}


def _prune_sessions(now: float) -> None:
    stale = [sid for sid, st in _SESSIONS.items() if (now - st.last_seen_ts) > _SESSION_TTL_SECONDS]
    for sid in stale:
        _SESSIONS.pop(sid, None)
    if len(_SESSIONS) > _MAX_SESSIONS:
        oldest = sorted(_SESSIONS.items(), key=lambda kv: kv[1].last_seen_ts)
        for sid, _ in oldest[: len(_SESSIONS) - _MAX_SESSIONS]:
            _SESSIONS.pop(sid, None)


def _reset_smoothing(state: _LiveSessionState) -> None:
    state.smoothed_probs = None
    state.last_top_expression = None
    state.consistent_frames = 0


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _box_center_distance_norm(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    acx, acy = ax + aw * 0.5, ay + ah * 0.5
    bcx, bcy = bx + bw * 0.5, by + bh * 0.5
    diag = max(1.0, float(np.hypot(max(aw, bw), max(ah, bh))))
    return float(np.hypot(acx - bcx, acy - bcy)) / diag


def _select_target_face(
    detections: List[Dict[str, Any]],
    state: _LiveSessionState,
    selection_point: Optional[Sequence[float]],
    snapshot: bool,
    now: float,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    if not detections:
        if not snapshot and state.tracked_box is not None:
            if state.tracked_lost_ts is None:
                state.tracked_lost_ts = now
            if (now - state.tracked_lost_ts) <= _REACQUIRE_GRACE_SECONDS:
                return "reacquiring", None
            state.tracked_box = None
            state.tracked_lost_ts = None
            _reset_smoothing(state)
        return "no_face", None

    if selection_point is not None and len(selection_point) == 2:
        px, py = float(selection_point[0]), float(selection_point[1])
        containing = [
            det
            for det in detections
            if det["bbox"][0] <= px <= det["bbox"][0] + det["bbox"][2]
            and det["bbox"][1] <= py <= det["bbox"][1] + det["bbox"][3]
        ]
        if containing:
            chosen = max(containing, key=lambda d: d["score"])
        else:
            chosen = min(
                detections,
                key=lambda d: np.hypot(
                    (d["bbox"][0] + d["bbox"][2] * 0.5) - px,
                    (d["bbox"][1] + d["bbox"][3] * 0.5) - py,
                ),
            )
        state.tracked_box = chosen["bbox"]
        state.tracked_lost_ts = None
        _reset_smoothing(state)
        return "tracked", chosen

    if len(detections) == 1:
        chosen = detections[0]
        if state.tracked_box is not None:
            iou = _box_iou(state.tracked_box, chosen["bbox"])
            dist = _box_center_distance_norm(state.tracked_box, chosen["bbox"])
            if iou < 0.15 and dist > 0.85:
                _reset_smoothing(state)
        state.tracked_box = chosen["bbox"]
        state.tracked_lost_ts = None
        return "single_face", chosen

    if snapshot:
        return "primary_face", detections[0]

    if state.tracked_box is not None:
        best_det = None
        best_score = -1.0
        for det in detections:
            iou = _box_iou(state.tracked_box, det["bbox"])
            dist = _box_center_distance_norm(state.tracked_box, det["bbox"])
            if iou >= 0.25 or dist <= 0.45:
                match_score = iou - 0.25 * dist
                if match_score > best_score:
                    best_score = match_score
                    best_det = det
        if best_det is not None:
            state.tracked_box = best_det["bbox"]
            state.tracked_lost_ts = None
            return "tracked", best_det

        if state.tracked_lost_ts is None:
            state.tracked_lost_ts = now
        if (now - state.tracked_lost_ts) <= _REACQUIRE_GRACE_SECONDS:
            return "reacquiring", None
        state.tracked_box = None
        state.tracked_lost_ts = None
        _reset_smoothing(state)

    return "select_face", None


def get_active_model_info() -> Dict[str, Any]:
    """Returns metadata for the single active CleanBTNVit model."""
    status = get_ai_runtime_status()
    return {
        "model": "clean_btn_vit",
        "model_name": CLEAN_BTN_MODEL_NAME,
        "loaded": status["model_loaded"],
        "detector": status["detector"],
        "detector_loaded": status["detector_loaded"],
        "device": status["device"],
        "default_inference_mode": status["default_inference_mode"],
        "checkpoint_sha256": status["checkpoint_sha256"],
    }


def predict_live(
    image_bytes: bytes,
    session: str = "single-image-session",
    selection: Optional[Sequence[float]] = None,
    snapshot: bool = False,
    debug: bool = False,
    horizontal_flip: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Runs real-time webcam or snapshot facial emotion recognition using the single
    startup-loaded YuNetFaceDetector + Phase 12 CleanBTNVitAdapter.
    """
    t0 = time.perf_counter()
    now = time.monotonic()

    detector = get_default_detector()
    model = get_default_model()
    use_flip = model.default_horizontal_flip if horizontal_flip is None else bool(horizontal_flip)
    inference_mode = "horizontal_flip" if use_flip else "single_view"

    def _finish(
        status: str,
        face_status: str,
        expression: Optional[str] = None,
        confidence: float = 0.0,
        raw_expression: Optional[str] = None,
        raw_confidence: Optional[float] = None,
        probabilities: Optional[List[float]] = None,
        expression_scores: Optional[Dict[str, float]] = None,
        boxes: Optional[List[List[int]]] = None,
        inference_ms: Optional[float] = None,
        debug_payload: Optional[Dict[str, Any]] = None,
        override_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        formatted = process_prediction_result(
            {
                "status": status,
                "expression": expression,
                "confidence": confidence,
                "expression_scores": expression_scores,
                "model_name": CLEAN_BTN_MODEL_NAME,
                "inference_mode": inference_mode,
                "inference_ms": inference_ms,
                "latency_ms": latency_ms,
            }
        )
        if override_note:
            formatted["note"] = override_note
        return {
            **formatted,
            "expression": expression,
            "expression_scores": expression_scores,
            "model": "clean_btn_vit",
            "model_name": CLEAN_BTN_MODEL_NAME,
            "inference_mode": inference_mode,
            "raw_expression": raw_expression,
            "raw_confidence": raw_confidence,
            "probabilities": probabilities,
            "debug": debug_payload if debug else None,
            "face_status": face_status,
            "boxes": boxes or [],
            "latency_ms": latency_ms,
            "inference_ms": round(inference_ms, 2) if inference_ms is not None else None,
        }

    image = decode_image(image_bytes)
    if image is None:
        return _finish(status="invalid_image", face_status="invalid_image")

    if not detector.is_available:
        return _finish(status="face_detector_unavailable", face_status="face_detector_unavailable")

    if not model.is_weights_loaded:
        return _finish(status="model_weights_not_loaded", face_status="model_unavailable")

    detections, threshold_used = detector.detect_faces_detailed(image)
    boxes = [list(d["bbox"]) for d in detections]

    with _SESSION_LOCK:
        _prune_sessions(now)
        if snapshot:
            state = _LiveSessionState(last_seen_ts=now)
        else:
            state = _SESSIONS.get(session)
            if state is None:
                state = _LiveSessionState(last_seen_ts=now)
                _SESSIONS[session] = state
            state.last_seen_ts = now
        face_status, chosen = _select_target_face(detections, state, selection, snapshot, now)

    if face_status == "no_face":
        return _finish(status="no_face_detected", face_status="no_face", boxes=boxes)

    if face_status == "reacquiring":
        return _finish(
            status="reacquiring",
            face_status="reacquiring",
            boxes=boxes,
            override_note="Looking for the previously selected face.",
        )

    if face_status == "select_face" or chosen is None:
        return _finish(
            status="multiple_faces",
            face_status="select_face",
            boxes=boxes,
            override_note="Click the intended face in the preview to confirm the target, or choose a mood manually.",
        )

    face_crop = crop_face(image, chosen["bbox"], margin_ratio=0.06)
    tensor = preprocess_face_for_clean_btn(face_crop) if face_crop is not None else None
    if tensor is None:
        return _finish(status="preprocessing_failed", face_status=face_status, boxes=boxes)

    raw_probs, inference_ms, inference_mode = model.predict_probabilities(
        tensor, horizontal_flip=use_flip
    )
    if raw_probs is None or len(raw_probs) != len(CLEAN_BTN_LABELS):
        return _finish(status="error", face_status=face_status, boxes=boxes, inference_ms=inference_ms)

    raw_top_idx = int(np.argmax(raw_probs))
    raw_expression = CLEAN_BTN_LABELS[raw_top_idx]
    raw_confidence = float(raw_probs[raw_top_idx])

    if snapshot:
        final_probs = raw_probs
        confidence = raw_confidence
        expression = raw_expression
        status = "success" if confidence >= DEFAULT_CONFIDENCE_THRESHOLD else "low_confidence"
        override_note = None
    else:
        with _SESSION_LOCK:
            if state.smoothed_probs is None:
                state.smoothed_probs = raw_probs.copy()
            else:
                state.smoothed_probs = (
                    (1.0 - _EMA_ALPHA) * state.smoothed_probs + _EMA_ALPHA * raw_probs
                )
            final_probs = state.smoothed_probs.copy()
            top_idx = int(np.argmax(final_probs))
            candidate_expr = CLEAN_BTN_LABELS[top_idx]
            confidence = float(final_probs[top_idx])
            if candidate_expr == state.last_top_expression:
                state.consistent_frames += 1
            else:
                state.last_top_expression = candidate_expr
                state.consistent_frames = 1
            consistent = state.consistent_frames

        if confidence < DEFAULT_CONFIDENCE_THRESHOLD:
            expression = candidate_expr
            status = "low_confidence"
            override_note = None
        elif consistent < 2:
            expression = None
            status = "confirming"
            override_note = "Waiting for a second consistent expression estimate."
        else:
            expression = candidate_expr
            status = "success"
            override_note = None

    expression_scores = {
        label: round(float(final_probs[idx]), 4)
        for idx, label in enumerate(CLEAN_BTN_LABELS)
    }
    prob_list = [round(float(v), 4) for v in final_probs]

    debug_payload = None
    if debug:
        debug_payload = {
            "architecture": "CleanBTNVit",
            "output_head": "output_VIT",
            "inference_mode": inference_mode,
            "device": model.device_name,
            "checkpoint_sha256": model.checkpoint_sha256,
            "yunet_threshold_used": threshold_used,
            "selected_box": list(chosen["bbox"]),
            "face_detection_score": round(float(chosen["score"]), 4),
            "crop_shape": list(face_crop.shape[:2]),
            "input_tensor_shape": list(tensor.shape),
            "labels_order": list(CLEAN_BTN_LABELS),
            "raw_probabilities": {
                label: round(float(raw_probs[idx]), 4)
                for idx, label in enumerate(CLEAN_BTN_LABELS)
            },
            "smoothed_probabilities": expression_scores,
        }

    return _finish(
        status=status,
        face_status=face_status,
        expression=expression,
        confidence=confidence,
        raw_expression=raw_expression,
        raw_confidence=round(raw_confidence, 4),
        probabilities=prob_list,
        expression_scores=expression_scores,
        boxes=boxes,
        inference_ms=inference_ms,
        debug_payload=debug_payload,
        override_note=override_note,
    )
