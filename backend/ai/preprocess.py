"""
Image decoding, bounding-box face cropping, and exact Phase 12 CleanBTNVit preprocessing.
"""

from __future__ import annotations

import base64
import binascii
import os
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
_CLEAN_BTN_MEAN = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
_CLEAN_BTN_STD = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)


def decode_image(image_input: Union[bytes, bytearray, str, np.ndarray]) -> Optional[np.ndarray]:
    """
    Decodes an image from NumPy ndarray, raw bytes, file path, or Base64 string into a
    standard 3-channel BGR NumPy array (H, W, 3). Returns None if invalid.
    """
    if image_input is None:
        return None

    if isinstance(image_input, np.ndarray):
        if image_input.size == 0 or image_input.ndim not in (2, 3):
            return None
        h, w = image_input.shape[:2]
        if h <= 0 or w <= 0:
            return None
        try:
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.shape[2] == 3:
                return image_input.copy()
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None
        return None

    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            return None
        try:
            np_arr = np.frombuffer(image_input, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0 and img.ndim == 3 and img.shape[2] == 3:
                return img
            return None
        except Exception:
            return None

    if isinstance(image_input, str):
        trimmed = image_input.strip()
        if not trimmed:
            return None
        if os.path.isfile(trimmed):
            try:
                img = cv2.imread(trimmed, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    return img
            except Exception:
                pass
        try:
            payload = (
                trimmed.split(",", 1)[1]
                if "," in trimmed and "base64" in trimmed.split(",", 1)[0].lower()
                else trimmed
            )
            decoded_bytes = base64.b64decode(payload, validate=True)
            if not decoded_bytes:
                return None
            np_arr = np.frombuffer(decoded_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return img
            return None
        except (binascii.Error, ValueError, Exception):
            return None

    return None


def crop_face(
    image: np.ndarray,
    bbox: Union[Sequence[int], Tuple[int, int, int, int]],
    margin_ratio: float = 0.06,
) -> Optional[np.ndarray]:
    """
    Crops a face bounding box (x, y, w, h) with proportional margin clamped strictly
    within image bounds.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0 or image.ndim < 2:
        return None
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    except (ValueError, TypeError):
        return None
    if w <= 0 or h <= 0:
        return None

    img_h, img_w = image.shape[:2]
    if img_h <= 0 or img_w <= 0:
        return None

    margin_ratio = max(0.0, float(margin_ratio))
    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    return crop.copy() if crop.size > 0 else None


def preprocess_face_for_clean_btn(
    face_image: np.ndarray,
    target_size: int = 112,
) -> Optional[torch.Tensor]:
    """
    Exact Phase 12 deterministic preprocessing for CleanBTNVit:
      1. Convert BGR crop to luminance uint8 grayscale (preserving exact pixels if
         already grayscale).
      2. Scale uint8 to float32 [0.0, 1.0] as shape (1, 1, H, W).
      3. Resize to (112, 112) via bilinear interpolation (align_corners=False).
      4. Apply .mul(255.0).round().div(255.0) quantization.
      5. Broadcast-normalize across 3 channels with ImageNet mean/std -> (1, 3, 112, 112).
    """
    if face_image is None or not isinstance(face_image, np.ndarray) or face_image.size == 0:
        return None

    try:
        if face_image.ndim == 2:
            gray = face_image
        elif face_image.ndim == 3 and face_image.shape[2] == 3:
            if (
                np.array_equal(face_image[:, :, 0], face_image[:, :, 1])
                and np.array_equal(face_image[:, :, 0], face_image[:, :, 2])
            ):
                gray = face_image[:, :, 0]
            else:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        elif face_image.ndim == 3 and face_image.shape[2] == 4:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGRA2GRAY)
        else:
            return None

        if gray.size == 0 or gray.shape[0] <= 0 or gray.shape[1] <= 0:
            return None

        gray_u8 = np.ascontiguousarray(gray, dtype=np.uint8)
        x = torch.from_numpy(gray_u8).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32) / 255.0
        x = F.interpolate(
            x,
            size=(int(target_size), int(target_size)),
            mode="bilinear",
            align_corners=False,
        )
        x = (x.mul(255.0).round().div(255.0) - _CLEAN_BTN_MEAN) / _CLEAN_BTN_STD
        return x.contiguous()
    except Exception:
        return None
