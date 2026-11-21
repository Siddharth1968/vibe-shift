"""
Single-model production emotion-recognition runtime for VibeShift.

Uses ONLY the selected Phase 12 standalone CleanBTNVit classifier paired with
YuNet face detection. No ensembles and no legacy fallback classifiers.
Untrained/unloaded weights strictly block inference to avoid unvalidated outputs.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from backend.ai.btn_runtime.clean_btn_vit import CleanBTNVit
from backend.ai.face_detector import YuNetFaceDetector
from backend.ai.preprocess import (
    crop_face,
    decode_image,
    preprocess_face_for_clean_btn,
)

logger = logging.getLogger(__name__)

# Canonical 7 facial expression classes (Phase 12 CleanBTNVit output logit order)
CLEAN_BTN_LABELS: Tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)

EXPRESSION_LABELS: Tuple[str, ...] = CLEAN_BTN_LABELS
NUM_CLASSES: int = len(CLEAN_BTN_LABELS)

CLEAN_BTN_MODEL_NAME: str = "CleanBTNVit (Phase 12)"
CLEAN_BTN_EXPECTED_SHA256: str = (
    "0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58"
)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.40

DISCLAIMER_TEXT: str = (
    "Facial-expression predictions are estimates based on visual features and "
    "do not represent a person's actual emotional state."
)


def _file_sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class CleanBTNVitAdapter:
    """
    Production adapter for the selected Phase 12 standalone CleanBTNVit model
    (`backend/ai/models/clean_btn_vit_phase12.pt`).

    - Strictly batch-independent `output_VIT` pathway at 112x112 grayscale-derived input.
    - Canonical 7-class order: angry, disgust, fear, happy, neutral, sad, surprise.
    - Single-view inference by default for real-time speed; optional horizontal-flip
      accuracy mode (`horizontal_flip=True`).
    - Automatic CUDA FP16 inference with clean CPU FP32 fallback.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        expected_sha256: Optional[str] = CLEAN_BTN_EXPECTED_SHA256,
        default_horizontal_flip: Optional[bool] = None,
    ):
        resolved_device = device or os.environ.get("VIBESHIFT_DEVICE")
        if resolved_device:
            if resolved_device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("Requested device '%s' unavailable; falling back to CPU.", resolved_device)
                resolved_device = "cpu"
        else:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(resolved_device)
        self.confidence_threshold = float(confidence_threshold)
        self.expected_sha256 = expected_sha256
        self.default_horizontal_flip = (
            _parse_bool_env("VIBESHIFT_DEFAULT_FLIP", default=False)
            if default_horizontal_flip is None
            else bool(default_horizontal_flip)
        )
        self.model_name = CLEAN_BTN_MODEL_NAME
        self.model: Optional[CleanBTNVit] = None
        self._weights_loaded: bool = False
        self._weights_path: Optional[str] = None
        self._checkpoint_sha256: Optional[str] = None
        self._load_error: Optional[str] = None
        self._infer_lock = threading.Lock()

        if weights_path is not None:
            self.load_weights(weights_path)

    @property
    def is_weights_loaded(self) -> bool:
        return self._weights_loaded

    @property
    def weights_path(self) -> Optional[str]:
        return self._weights_path

    @property
    def checkpoint_sha256(self) -> Optional[str]:
        return self._checkpoint_sha256

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def device_name(self) -> str:
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                return f"cuda ({torch.cuda.get_device_name(self.device)})"
            except Exception:
                return "cuda"
        return "cpu"

    @classmethod
    def resolve_default_weights_path(cls) -> Optional[str]:
        env_path = os.environ.get("VIBESHIFT_BTN_WEIGHTS")
        module_dir = Path(__file__).resolve().parent
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(module_dir / "models" / "clean_btn_vit_phase12.pt")
        for cand in candidates:
            if cand.is_file():
                return str(cand.resolve())
        return None

    def load_weights(self, weights_path: str) -> bool:
        """
        Loads and SHA-256 verifies the Phase 12 CleanBTNVit checkpoint using weights_only=True.
        """
        if not weights_path or not os.path.isfile(weights_path):
            self._weights_loaded = False
            self._weights_path = None
            self._checkpoint_sha256 = None
            self._load_error = (
                f"CleanBTNVit checkpoint not found at '{weights_path}'. "
                "See docs/MODEL_SETUP.md to install authorized Phase 12 weights."
            )
            logger.warning("CleanBTNVitAdapter weights missing: %s", self._load_error)
            return False

        try:
            actual_sha = _file_sha256(weights_path)
            if self.expected_sha256 and actual_sha.lower() != self.expected_sha256.lower():
                self._weights_loaded = False
                self._weights_path = None
                self._checkpoint_sha256 = actual_sha
                self._load_error = (
                    f"Checkpoint SHA-256 mismatch: expected {self.expected_sha256}, got {actual_sha}"
                )
                logger.error("CleanBTNVitAdapter load blocked: %s", self._load_error)
                return False

            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint does not contain a valid state_dict mapping.")

            model = CleanBTNVit(head_drop=0.0)
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.model = model
            self._weights_loaded = True
            self._weights_path = str(Path(weights_path).resolve())
            self._checkpoint_sha256 = actual_sha
            self._load_error = None
            logger.info(
                "CleanBTNVitAdapter loaded verified checkpoint '%s' (SHA-256 %s...) on %s",
                self._weights_path,
                actual_sha[:12],
                self.device_name,
            )
            return True
        except Exception as exc:
            self.model = None
            self._weights_loaded = False
            self._weights_path = None
            self._checkpoint_sha256 = None
            self._load_error = str(exc)
            logger.exception("CleanBTNVitAdapter failed to load weights from '%s':", weights_path)
            return False

    def warmup(self) -> None:
        """Runs a single dummy inference pass at startup to warm GPU/CPU kernels."""
        if not self._weights_loaded or self.model is None:
            return
        dummy = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
        self.predict_probabilities(dummy, horizontal_flip=False)

    def predict_probabilities(
        self,
        tensor: torch.Tensor,
        horizontal_flip: Optional[bool] = None,
    ) -> Tuple[Optional[np.ndarray], float, str]:
        """
        Executes CleanBTNVit forward pass and returns (probs_7, inference_ms, mode_name).
        """
        use_flip = self.default_horizontal_flip if horizontal_flip is None else bool(horizontal_flip)
        mode_name = "horizontal_flip" if use_flip else "single_view"

        if not self._weights_loaded or self.model is None:
            return None, 0.0, mode_name
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None, 0.0, mode_name

        with self._infer_lock:
            t0 = time.perf_counter()
            with torch.inference_mode():
                x = tensor.to(device=self.device, dtype=torch.float32, non_blocking=False).contiguous()
                if self.device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = self.model(x).float()
                        probs = torch.softmax(logits, dim=1)
                        if use_flip:
                            logits_flip = self.model(x.flip(-1).contiguous()).float()
                            probs = 0.5 * (probs + torch.softmax(logits_flip, dim=1))
                    torch.cuda.synchronize(self.device)
                else:
                    logits = self.model(x).float()
                    probs = torch.softmax(logits, dim=1)
                    if use_flip:
                        logits_flip = self.model(x.flip(-1).contiguous()).float()
                        probs = 0.5 * (probs + torch.softmax(logits_flip, dim=1))
                probs_np = probs.squeeze(0).detach().cpu().numpy().astype(np.float32)
            inference_ms = (time.perf_counter() - t0) * 1000.0
            return probs_np, inference_ms, mode_name

    def predict_tensor(
        self,
        tensor: torch.Tensor,
        horizontal_flip: Optional[bool] = None,
    ) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, float]], str]:
        if not self._weights_loaded or self.model is None:
            return (
                None,
                None,
                None,
                "model_weights_not_loaded: Verified CleanBTNVit weights have not been installed. "
                "Inference blocked to avoid unvalidated output.",
            )

        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None, None, None, "invalid_tensor"

        try:
            probs, _, _ = self.predict_probabilities(tensor, horizontal_flip=horizontal_flip)
            if probs is None or len(probs) != len(CLEAN_BTN_LABELS):
                return None, None, None, "inference_error: invalid probability output"

            scores = {label: float(probs[idx]) for idx, label in enumerate(CLEAN_BTN_LABELS)}
            top_idx = int(np.argmax(probs))
            top_label = CLEAN_BTN_LABELS[top_idx]
            top_confidence = float(probs[top_idx])

            status = "low_confidence" if top_confidence < self.confidence_threshold else "success"
            return (
                top_label,
                round(top_confidence, 4),
                {k: round(v, 4) for k, v in scores.items()},
                status,
            )
        except Exception as exc:
            return None, None, None, f"inference_error: {str(exc)}"


_RUNTIME_LOCK = threading.Lock()
_DEFAULT_DETECTOR: Optional[YuNetFaceDetector] = None
_DEFAULT_MODEL: Optional[CleanBTNVitAdapter] = None
_LAST_DETECTOR_INIT_ATTEMPT: float = 0.0
_DETECTOR_INIT_COOLDOWN: float = 2.0
_LAST_DETECTOR_UNAVAILABLE_LOGGED: Optional[str] = None
_STARTUP_INITIALIZED: bool = False


def get_default_detector() -> YuNetFaceDetector:
    """Returns or initializes the shared YuNet face detector singleton."""
    global _DEFAULT_DETECTOR, _LAST_DETECTOR_INIT_ATTEMPT, _LAST_DETECTOR_UNAVAILABLE_LOGGED
    with _RUNTIME_LOCK:
        now = time.perf_counter()
        if _DEFAULT_DETECTOR is None:
            _LAST_DETECTOR_INIT_ATTEMPT = now
            _DEFAULT_DETECTOR = YuNetFaceDetector()
            return _DEFAULT_DETECTOR

        if not _DEFAULT_DETECTOR.is_available:
            if (now - _LAST_DETECTOR_INIT_ATTEMPT) >= _DETECTOR_INIT_COOLDOWN:
                _LAST_DETECTOR_INIT_ATTEMPT = now
                new_detector = YuNetFaceDetector()
                _LAST_DETECTOR_UNAVAILABLE_LOGGED = None
                _DEFAULT_DETECTOR = new_detector

        return _DEFAULT_DETECTOR


def get_default_model(weights_path: Optional[str] = None) -> CleanBTNVitAdapter:
    """Returns or initializes the shared Phase 12 CleanBTNVit singleton."""
    global _DEFAULT_MODEL
    with _RUNTIME_LOCK:
        if weights_path is not None:
            _DEFAULT_MODEL = CleanBTNVitAdapter(weights_path=weights_path, expected_sha256=None)
            return _DEFAULT_MODEL

        if _DEFAULT_MODEL is None:
            resolved_btn = CleanBTNVitAdapter.resolve_default_weights_path()
            _DEFAULT_MODEL = CleanBTNVitAdapter(weights_path=resolved_btn)

        return _DEFAULT_MODEL


def initialize_ai_runtime(warmup: bool = True) -> Dict[str, Any]:
    """Loads YuNet and CleanBTNVit once during FastAPI application startup."""
    global _STARTUP_INITIALIZED
    get_default_detector()
    model = get_default_model()
    if warmup and model.is_weights_loaded:
        try:
            model.warmup()
        except Exception:
            logger.exception("CleanBTNVit warmup failed (non-fatal):")
    _STARTUP_INITIALIZED = True
    return get_ai_runtime_status()


def get_ai_runtime_status() -> Dict[str, Any]:
    """Returns read-only runtime status for health checks and diagnostics."""
    detector = _DEFAULT_DETECTOR
    model = _DEFAULT_MODEL
    return {
        "initialized": _STARTUP_INITIALIZED,
        "model": CLEAN_BTN_MODEL_NAME,
        "model_loaded": bool(model and model.is_weights_loaded),
        "checkpoint_sha256": getattr(model, "checkpoint_sha256", None),
        "device": getattr(model, "device_name", "uninitialized"),
        "default_inference_mode": (
            "horizontal_flip"
            if getattr(model, "default_horizontal_flip", False)
            else "single_view"
        ),
        "detector": "YuNetFaceDetector",
        "detector_loaded": bool(detector and detector.is_available),
    }


def predict(
    image_input: Union[bytes, bytearray, str, np.ndarray],
    detector: Optional[YuNetFaceDetector] = None,
    model: Optional[CleanBTNVitAdapter] = None,
    horizontal_flip: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    End-to-end single-model facial expression estimation:
    decode_image -> YuNetFaceDetector -> crop_face -> preprocess_face_for_clean_btn -> CleanBTNVit.
    """
    start_time = time.perf_counter()
    mode_fallback = "horizontal_flip" if horizontal_flip else "single_view"

    image = decode_image(image_input)
    if image is None:
        return {
            "face_detected": False,
            "expression": None,
            "confidence": None,
            "status": "invalid_image",
            "disclaimer": DISCLAIMER_TEXT,
            "bbox": None,
            "expression_scores": None,
            "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "inference_ms": None,
            "model_name": CLEAN_BTN_MODEL_NAME,
            "inference_mode": mode_fallback,
        }

    global _LAST_DETECTOR_UNAVAILABLE_LOGGED
    if detector is None:
        detector = get_default_detector()

    if not detector.is_available:
        err_msg = str(detector.load_error)
        if _LAST_DETECTOR_UNAVAILABLE_LOGGED != err_msg:
            logger.error("Prediction aborted: Face detector unavailable (%s).", err_msg)
            _LAST_DETECTOR_UNAVAILABLE_LOGGED = err_msg
        return {
            "face_detected": False,
            "expression": None,
            "confidence": None,
            "status": "face_detector_unavailable",
            "disclaimer": DISCLAIMER_TEXT,
            "bbox": None,
            "expression_scores": None,
            "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "inference_ms": None,
            "model_name": CLEAN_BTN_MODEL_NAME,
            "inference_mode": mode_fallback,
        }

    if _LAST_DETECTOR_UNAVAILABLE_LOGGED is not None:
        _LAST_DETECTOR_UNAVAILABLE_LOGGED = None

    faces = detector.detect_faces(image)
    primary_face = detector.get_primary_face(faces)
    if primary_face is None:
        return {
            "face_detected": False,
            "expression": None,
            "confidence": None,
            "status": "no_face_detected",
            "disclaimer": DISCLAIMER_TEXT,
            "bbox": None,
            "expression_scores": None,
            "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "inference_ms": None,
            "model_name": CLEAN_BTN_MODEL_NAME,
            "inference_mode": mode_fallback,
        }

    if model is None:
        model = get_default_model()

    face_crop = crop_face(image, primary_face, margin_ratio=0.06)
    tensor = preprocess_face_for_clean_btn(face_crop) if face_crop is not None else None
    if tensor is None:
        return {
            "face_detected": True,
            "expression": None,
            "confidence": None,
            "status": "preprocessing_failed",
            "disclaimer": DISCLAIMER_TEXT,
            "bbox": list(primary_face),
            "expression_scores": None,
            "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            "inference_ms": None,
            "model_name": CLEAN_BTN_MODEL_NAME,
            "inference_mode": mode_fallback,
        }

    use_flip = model.default_horizontal_flip if horizontal_flip is None else bool(horizontal_flip)
    inference_mode = "horizontal_flip" if use_flip else "single_view"

    infer_start = time.perf_counter()
    expression, confidence, scores, status = model.predict_tensor(
        tensor, horizontal_flip=use_flip
    )
    inference_ms = (time.perf_counter() - infer_start) * 1000.0

    return {
        "face_detected": True,
        "expression": expression,
        "confidence": confidence,
        "status": status,
        "disclaimer": DISCLAIMER_TEXT,
        "bbox": list(primary_face),
        "expression_scores": scores,
        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
        "inference_ms": round(inference_ms, 2),
        "model_name": CLEAN_BTN_MODEL_NAME,
        "inference_mode": inference_mode,
    }
