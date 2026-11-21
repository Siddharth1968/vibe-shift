"""
Verification test suite for strict language-based music recommendations in VibeShift.
Uses an isolated temporary SQLite database bootstrapped cleanly from backend/data/songs.csv.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.database import db as db_mod
from backend.services.recommendation import get_recommendations


@pytest.fixture(scope="module")
def seeded_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_db = Path(tmpdir) / "test_lang.db"
        with patch.object(db_mod, "DB_PATH", tmp_db):
            db_mod.init_db()
            db_mod.load_songs_csv_if_empty()
            yield tmp_db


def test_language_distribution(seeded_db):
    conn = db_mod.get_db_connection()
    counts = dict(
        conn.execute(
            "SELECT language, COUNT(*) FROM songs GROUP BY language ORDER BY COUNT(*) DESC"
        ).fetchall()
    )
    conn.close()
    assert sum(counts.values()) == 137
    assert set(counts.keys()) == {"English", "Tamil", "Hindi", "Malayalam"}


def test_strict_malayalam_and_english_filtering(seeded_db):
    for mood in ("happy", "neutral", "calm", "energetic"):
        recs = get_recommendations(current_mood=mood, mode="match", language="Malayalam", limit=10)
        assert len(recs) > 0
        assert all(s["language"].lower() == "malayalam" for s in recs)

    for mood in ("happy", "sad", "angry", "energetic"):
        recs = get_recommendations(current_mood=mood, mode="match", language="English", limit=10)
        assert len(recs) > 0
        assert all(s["language"].lower() == "english" for s in recs)


def test_insufficient_songs_never_silently_backfills(seeded_db):
    recs = get_recommendations(current_mood="angry", mode="match", language="Malayalam", limit=10)
    assert len(recs) == 1
    assert recs[0]["language"].lower() == "malayalam"


def test_language_aliases_and_all_supported(seeded_db):
    for raw_lang, expected in (("ml", "malayalam"), ("MALAYALAM", "malayalam"), ("en", "english"), ("eng", "english")):
        recs = get_recommendations(current_mood="happy", mode="match", language=raw_lang, limit=5)
        assert len(recs) > 0
        assert all(s["language"].lower() == expected for s in recs)

    all_recs = get_recommendations(current_mood="happy", mode="match", language="All Supported", limit=10)
    assert len(all_recs) == 10
    assert len({s["language"] for s in all_recs}) > 1
