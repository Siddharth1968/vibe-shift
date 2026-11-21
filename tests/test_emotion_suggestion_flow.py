"""
Verification tests for the AI emotion-to-music-mood mapping & suggestion pipeline.
Uses an isolated temporary SQLite database seeded automatically from backend/data/songs.csv.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.database import db as db_mod
from backend.services.emotion import process_prediction_result
from backend.services.recommendation import get_recommendations


def test_high_confidence_prediction():
    raw = {
        "face_detected": True,
        "expression": "happy",
        "confidence": 0.85,
        "status": "success",
    }
    result = process_prediction_result(raw)
    assert result["emotion"] == "happy"
    assert result["confidence"] == 0.85
    assert result["status"] == "success"
    assert "This is only an estimate" in result["note"]


def test_low_confidence_prediction_preserves_suggestion():
    raw = {
        "face_detected": True,
        "expression": "neutral",
        "confidence": 0.28,
        "status": "low_confidence",
    }
    result = process_prediction_result(raw)
    assert result["emotion"] == "neutral"
    assert result["confidence"] == 0.28
    assert result["status"] == "low_confidence"
    assert "AI suggests 'neutral'" in result["note"]


def test_unmapped_expressions_never_default_to_neutral():
    for expr in ("surprise", "fear", "disgust"):
        raw = {
            "face_detected": True,
            "expression": expr,
            "confidence": 0.82,
            "status": "success",
        }
        result = process_prediction_result(raw)
        assert result["emotion"] is None
        assert result["expression"] == expr
        assert "has no direct music mood mapping" in result["note"]


def test_non_success_statuses():
    for status, expected_phrase in (
        ("no_face_detected", "No face was detected"),
        ("invalid_image", "could not be processed"),
        ("face_detector_unavailable", "Face detector is currently unavailable"),
        ("model_weights_not_loaded", "AI model weights are currently not loaded"),
    ):
        result = process_prediction_result(
            {"face_detected": False, "expression": None, "confidence": 0.0, "status": status}
        )
        assert result["emotion"] is None
        assert result["status"] == status
        assert expected_phrase in result["note"]


def test_recommendations_for_all_six_moods():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "test_moods.db"
        with patch.object(db_mod, "DB_PATH", tmp_db):
            db_mod.init_db()
            db_mod.load_songs_csv_if_empty()
            for mood in ("happy", "sad", "angry", "neutral", "calm", "energetic"):
                recs = get_recommendations(current_mood=mood, mode="match", language="all", limit=5)
                assert len(recs) > 0, f"No recommendations returned for mood '{mood}'"
                for rec in recs:
                    assert rec["title"] and rec["artist"] and rec["audio_url"]
