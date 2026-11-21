"""
Self-contained integration & AI verification test suite for VibeShift.

Runs independently inside the exported repository without requiring external research
datasets or historical experiment directories. Tests requiring the locally installed
Phase 12 CleanBTNVit checkpoint automatically check its presence so a fresh public clone
still verifies API contracts, preprocessing, YuNet loading, database bootstrap,
recommendations, Spotify URL validity, and graceful model-missing fallback.
"""

from __future__ import annotations

import csv
import hashlib
import re
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.ai import (
    CLEAN_BTN_EXPECTED_SHA256,
    CLEAN_BTN_LABELS,
    YUNET_EXPECTED_SHA256,
    CleanBTNVitAdapter,
    YuNetFaceDetector,
    get_default_detector,
    get_default_model,
    initialize_ai_runtime,
    predict,
    preprocess_face_for_clean_btn,
)
from backend.database import db as db_mod
from backend.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "backend" / "ai" / "models" / "clean_btn_vit_phase12.pt"
YUNET_PATH = REPO_ROOT / "backend" / "ai" / "models" / "yunet-2023mar.onnx"
WEIGHTS_PRESENT = MODEL_PATH.is_file()
SPOTIFY_TRACK_RE = re.compile(r"^https://open\.spotify\.com/track/[A-Za-z0-9]{22}(\?.*)?$")


def _create_detectable_portrait() -> np.ndarray:
    """Creates a clean portrait canvas verified to trigger YuNet face detection."""
    canvas = np.full((400, 400, 3), 220, dtype=np.uint8)
    cv2.ellipse(canvas, (200, 200), (90, 120), 0, 0, 360, (180, 150, 130), -1)
    cv2.ellipse(canvas, (165, 170), (18, 10), 0, 0, 360, (50, 40, 30), -1)
    cv2.ellipse(canvas, (235, 170), (18, 10), 0, 0, 360, (50, 40, 30), -1)
    cv2.ellipse(canvas, (165, 150), (25, 5), 0, 0, 360, (40, 30, 20), -1)
    cv2.ellipse(canvas, (235, 150), (25, 5), 0, 0, 360, (40, 30, 20), -1)
    cv2.line(canvas, (200, 175), (200, 220), (140, 110, 90), 3)
    cv2.ellipse(canvas, (200, 225), (15, 6), 0, 0, 360, (140, 110, 90), -1)
    cv2.ellipse(canvas, (200, 260), (35, 12), 0, 0, 360, (70, 50, 150), -1)
    return canvas


def test_yunet_model_integrity_and_detection():
    assert YUNET_PATH.is_file(), f"Missing YuNet ONNX model at {YUNET_PATH}"
    actual_sha = hashlib.sha256(YUNET_PATH.read_bytes()).hexdigest()
    assert actual_sha == YUNET_EXPECTED_SHA256

    detector = get_default_detector()
    assert isinstance(detector, YuNetFaceDetector)
    assert detector.is_available is True
    assert detector is get_default_detector()

    portrait = _create_detectable_portrait()
    faces = detector.detect_faces(portrait)
    assert len(faces) >= 1


def test_preprocessing_exact_phase12_formula():
    sample_gray = np.arange(48 * 48, dtype=np.uint8).reshape(48, 48)
    sample_bgr = cv2.cvtColor(sample_gray, cv2.COLOR_GRAY2BGR)
    t_gray = preprocess_face_for_clean_btn(sample_gray)
    t_bgr = preprocess_face_for_clean_btn(sample_bgr)
    assert t_gray is not None and t_bgr is not None
    assert tuple(t_gray.shape) == (1, 3, 112, 112)
    assert torch.equal(t_gray, t_bgr)

    ref = torch.from_numpy(sample_gray).unsqueeze(0).unsqueeze(0).float() / 255.0
    ref = torch.nn.functional.interpolate(ref, size=(112, 112), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    ref = (ref.mul(255.0).round().div(255.0) - mean) / std
    assert (t_gray - ref).abs().max().item() == 0.0


def test_error_handling_unloaded_corrupt_blank_and_unavailable_detector():
    unloaded = CleanBTNVitAdapter(weights_path=None)
    assert unloaded.is_weights_loaded is False
    dummy = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
    expr, conf, scores, st = unloaded.predict_tensor(dummy)
    assert expr is None and conf is None and scores is None
    assert st.startswith("model_weights_not_loaded")

    res_corrupt = predict(b"not-a-valid-image")
    assert res_corrupt["status"] == "invalid_image"
    assert res_corrupt["face_detected"] is False

    blank = np.zeros((240, 240, 3), dtype=np.uint8)
    res_blank = predict(blank)
    assert res_blank["status"] == "no_face_detected"
    assert res_blank["face_detected"] is False

    broken = YuNetFaceDetector(model_path="nonexistent_yunet.onnx")
    assert broken.is_available is False
    res_unavail = predict(blank, detector=broken)
    assert res_unavail["status"] == "face_detector_unavailable"


@pytest.mark.skipif(
    not WEIGHTS_PRESENT,
    reason="Authorized Phase 12 CleanBTNVit weights not installed locally (see docs/MODEL_SETUP.md).",
)
def test_single_model_startup_integrity_and_batch_independence():
    status = initialize_ai_runtime(warmup=True)
    assert status["initialized"] is True
    assert status["model_loaded"] is True
    assert status["detector_loaded"] is True
    assert status["model"] == "CleanBTNVit (Phase 12)"
    assert status["checkpoint_sha256"] == CLEAN_BTN_EXPECTED_SHA256
    assert status["default_inference_mode"] == "single_view"

    model_adapter = get_default_model()
    assert isinstance(model_adapter, CleanBTNVitAdapter)
    assert model_adapter is get_default_model()

    assert CLEAN_BTN_LABELS == (
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    )

    rng = torch.Generator(device="cpu").manual_seed(42)
    x = torch.randn((3, 3, 112, 112), generator=rng).to(model_adapter.device)
    peers = x.clone()
    peers[1:] = -2.5 * peers[1:]
    with torch.inference_mode():
        out_orig = model_adapter.model(x)[0]
        out_peers = model_adapter.model(peers)[0]
        out_single = model_adapter.model(x[:1])[0]

    assert (out_orig - out_peers).abs().max().item() < 1e-5
    assert (out_orig - out_single).abs().max().item() < 1e-3
    assert out_orig.argmax().item() == out_single.argmax().item()


@pytest.mark.skipif(
    not WEIGHTS_PRESENT,
    reason="Authorized Phase 12 CleanBTNVit weights not installed locally (see docs/MODEL_SETUP.md).",
)
def test_single_view_and_horizontal_flip_inference():
    portrait = _create_detectable_portrait()
    res_single = predict(portrait, horizontal_flip=False)
    res_flip = predict(portrait, horizontal_flip=True)

    for res, expected_mode in ((res_single, "single_view"), (res_flip, "horizontal_flip")):
        assert res["face_detected"] is True
        assert res["expression"] in CLEAN_BTN_LABELS
        assert res["status"] in ("success", "low_confidence")
        assert res["inference_mode"] == expected_mode
        assert res["model_name"] == "CleanBTNVit (Phase 12)"
        assert set(res["expression_scores"].keys()) == set(CLEAN_BTN_LABELS)
        assert abs(sum(res["expression_scores"].values()) - 1.0) < 0.02
        assert res["inference_ms"] is not None and res["inference_ms"] > 0


def test_fastapi_full_workflow_with_fresh_sqlite_database():
    portrait = _create_detectable_portrait()
    ok, jpg = cv2.imencode(".jpg", portrait)
    assert ok
    jpg_bytes = jpg.tobytes()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "fresh_vibeshift.db"
        with patch.object(db_mod, "DB_PATH", tmp_db):
            with TestClient(app) as client:
                # 1. Health & active model endpoints
                h = client.get("/api/health")
                assert h.status_code == 200
                h_json = h.json()
                assert h_json["status"] == "ok"
                assert h_json["ai"]["model"] == "CleanBTNVit (Phase 12)"
                assert h_json["ai"]["detector_loaded"] is True

                m_info = client.get("/api/emotion/model")
                assert m_info.status_code == 200
                assert m_info.json()["model"] == "clean_btn_vit"

                # 2. Emotion endpoints (full inference when weights present; graceful block when absent)
                r_single = client.post(
                    "/api/emotion",
                    files={"file": ("face.jpg", jpg_bytes, "image/jpeg")},
                )
                assert r_single.status_code == 200
                d_single = r_single.json()
                if WEIGHTS_PRESENT:
                    assert d_single["expression"] in CLEAN_BTN_LABELS
                    assert d_single["inference_mode"] == "single_view"
                    r_flip = client.post(
                        "/api/emotion",
                        files={"file": ("face.jpg", jpg_bytes, "image/jpeg")},
                        data={"horizontal_flip": "true"},
                    )
                    assert r_flip.status_code == 200
                    assert r_flip.json()["inference_mode"] == "horizontal_flip"

                    r_snap = client.post(
                        "/api/emotion/live",
                        files={"file": ("face.jpg", jpg_bytes, "image/jpeg")},
                        data={
                            "session": "test-live-session-123456",
                            "snapshot": "true",
                            "debug": "true",
                            "horizontal_flip": "false",
                        },
                    )
                    assert r_snap.status_code == 200
                    assert r_snap.json()["debug"]["architecture"] == "CleanBTNVit"
                else:
                    assert d_single["emotion"] is None
                    assert d_single["status"].startswith("model_weights_not_loaded")

                # 3. Clean fresh SQLite session -> mood -> recommendations -> feedback -> journey
                s_res = client.post(
                    "/api/sessions",
                    json={"language": "English", "genre": "Pop", "context": "relax"},
                )
                assert s_res.status_code == 201
                sid = s_res.json()["session_id"]
                assert sid == 1

                chosen_mood = d_single.get("emotion") or "happy"
                m_res = client.post(
                    f"/api/sessions/{sid}/mood",
                    json={"mood": chosen_mood, "source": "camera_suggested", "confidence": 0.91},
                )
                assert m_res.status_code == 200
                assert m_res.json()["source"] == "camera_suggested"

                rec_res = client.post(
                    "/api/recommendations",
                    json={
                        "current_mood": chosen_mood,
                        "mode": "match",
                        "language": "English",
                        "genre": "Pop",
                        "context": "relax",
                    },
                )
                assert rec_res.status_code == 200
                songs = rec_res.json()
                assert len(songs) > 0
                assert SPOTIFY_TRACK_RE.match(songs[0]["audio_url"])

                fb_res = client.post(
                    "/api/feedback",
                    json={"session_id": sid, "song_id": songs[0]["id"], "action": "like"},
                )
                assert fb_res.status_code == 200

                j_res = client.get(f"/api/sessions/{sid}/journey")
                assert j_res.status_code == 200
                entries = j_res.json()["entries"]
                assert len(entries) == 1
                assert entries[0]["mood"] == chosen_mood
                assert entries[0]["source"] == "camera_suggested"

            # Verify fresh DB contains exactly 137 seeded songs and zero leaked dev sessions
            conn = sqlite3.connect(tmp_db)
            assert conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0] == 137
            conn.close()


def test_all_spotify_track_urls_valid_in_songs_csv():
    csv_path = REPO_ROOT / "backend" / "data" / "songs.csv"
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 137
    bad_csv = [r["title"] for r in rows if not SPOTIFY_TRACK_RE.match((r.get("audio_url") or "").strip())]
    assert bad_csv == [], f"Invalid Spotify URLs in songs.csv: {bad_csv}"


def test_model_installer_rejects_bad_checksum_and_verifies_valid_weights():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vibeshift_install_model", REPO_ROOT / "scripts" / "install_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory() as tmpdir:
        bad_file = Path(tmpdir) / "wrong.pt"
        bad_file.write_bytes(b"corrupted-weights-bytes")
        assert mod.install_from_file(bad_file) == 1

    if WEIGHTS_PRESENT:
        assert mod.main(["--verify-only"]) == 0

