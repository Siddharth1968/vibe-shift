"""
Thread-safe OpenCV YuNet ONNX face detector for VibeShift.
Uses project-relative `backend/ai/models/yunet-2023mar.onnx` (or `VIBESHIFT_YUNET_MODEL`).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

YUNET_EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


class YuNetFaceDetector:
    """
    Production neural face detector using OpenCV's YuNet ONNX model
    (`yunet-2023mar.onnx`) with two-stage score thresholding (primary 0.9, fallback 0.6).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        primary_threshold: float = 0.9,
        fallback_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        self.detector: Optional[Any] = None
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        self.primary_threshold = float(primary_threshold)
        self.fallback_threshold = float(fallback_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)

        if not hasattr(cv2, "FaceDetectorYN"):
            self._load_error = (
                "OpenCV build lacks cv2.FaceDetectorYN (YuNet) support. "
                "Install opencv-python-headless>=4.8.0."
            )
            logger.error("YuNetFaceDetector initialization failed: %s", self._load_error)
            self.model_path: Optional[str] = None
            return

        resolved_path, candidates = self._resolve_model_path_candidates(model_path)
        self.model_path = resolved_path

        if resolved_path is None or not os.path.isfile(resolved_path):
            self._load_error = f"YuNet ONNX model not found. Candidates checked: {candidates}"
            logger.error("YuNetFaceDetector initialization failed: %s", self._load_error)
            return

        try:
            self.detector = cv2.FaceDetectorYN.create(
                resolved_path,
                "",
                (320, 320),
                self.primary_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._is_loaded = True
            logger.info("YuNetFaceDetector loaded ONNX model from: '%s'", resolved_path)
        except Exception as exc:
            self._load_error = f"FaceDetectorYN error: {str(exc)}"
            logger.exception("YuNetFaceDetector failed to load '%s':", resolved_path)

    @classmethod
    def _resolve_model_path_candidates(
        cls, explicit_path: Optional[str] = None
    ) -> Tuple[Optional[str], List[str]]:
        if explicit_path is not None:
            resolved = str(Path(explicit_path).resolve())
            return (resolved if os.path.isfile(resolved) else None), [resolved]

        candidates: List[str] = []
        env_path = os.environ.get("VIBESHIFT_YUNET_MODEL")
        if env_path:
            candidates.append(str(Path(env_path).resolve()))

        module_dir = Path(__file__).resolve().parent
        candidates.append(str((module_dir / "models" / "yunet-2023mar.onnx").resolve()))

        for path in candidates:
            if os.path.isfile(path):
                return path, candidates
        return None, candidates

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def is_available(self) -> bool:
        return self._is_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _run_yunet(self, image_bgr: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        h, w = image_bgr.shape[:2]
        if w <= 0 or h <= 0 or self.detector is None:
            return []

        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(float(threshold))
        _, raw_faces = self.detector.detect(image_bgr)
        if raw_faces is None or len(raw_faces) == 0:
            return []

        detections: List[Dict[str, Any]] = []
        min_dim = 16 if min(w, h) < 96 else 24
        for row in raw_faces:
            fx, fy, fw, fh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            score = float(row[-1])
            x1 = max(0, int(round(fx)))
            y1 = max(0, int(round(fy)))
            x2 = min(w, int(round(fx + fw)))
            y2 = min(h, int(round(fy + fh)))
            bw = x2 - x1
            bh = y2 - y1
            if bw < min_dim or bh < min_dim:
                continue
            landmarks = [
                (float(row[4]), float(row[5])),
                (float(row[6]), float(row[7])),
                (float(row[8]), float(row[9])),
                (float(row[10]), float(row[11])),
                (float(row[12]), float(row[13])),
            ]
            detections.append(
                {
                    "bbox": (x1, y1, bw, bh),
                    "score": score,
                    "landmarks": landmarks,
                }
            )
        detections.sort(key=lambda d: d["bbox"][2] * d["bbox"][3], reverse=True)
        return detections

    def detect_faces_detailed(
        self,
        image: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Runs YuNet with primary threshold (0.9) and automatic fallback threshold (0.6).
        """
        if not self._is_loaded or self.detector is None:
            return [], self.primary_threshold
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return [], self.primary_threshold

        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            image_bgr = image
        elif image.ndim == 3 and image.shape[2] == 4:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            return [], self.primary_threshold

        with self._lock:
            try:
                detections = self._run_yunet(image_bgr, self.primary_threshold)
                if detections:
                    return detections, self.primary_threshold
                detections = self._run_yunet(image_bgr, self.fallback_threshold)
                return detections, self.fallback_threshold
            except Exception:
                logger.exception("Exception during YuNet face detection:")
                return [], self.primary_threshold

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        detections, _ = self.detect_faces_detailed(image)
        return [d["bbox"] for d in detections]

    def get_primary_face(
        self,
        faces: List[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        if not faces:
            return None
        return max(faces, key=lambda f: f[2] * f[3])
