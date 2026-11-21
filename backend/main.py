"""
VibeShift - Backend API
Main FastAPI application entry point with automatic SQLite initialization,
one-time startup AI loading (YuNet + Phase 12 CleanBTNVit), and route registration.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

from backend.ai import get_ai_runtime_status, initialize_ai_runtime
from backend.api.emotion import router as emotion_router
from backend.api.feedback import router as feedback_router
from backend.api.recommendations import router as recommendations_router
from backend.api.sessions import router as sessions_router
from backend.database.db import init_db, load_songs_csv_if_empty


def _get_cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]
    extra = os.environ.get("VIBESHIFT_CORS_ORIGINS", "")
    if extra.strip():
        for origin in extra.split(","):
            cleaned = origin.strip()
            if cleaned and cleaned not in default_origins:
                default_origins.append(cleaned)
    return default_origins


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    1. Initializes SQLite tables and seeds songs.csv if empty.
    2. Loads the YuNet face detector and Phase 12 CleanBTNVit emotion model ONCE at startup.
    """
    print("[Startup] Initializing SQLite database...")
    init_db()
    print("[Startup] Checking and loading initial songs catalogue...")
    load_songs_csv_if_empty()
    print("[Startup] Loading YuNet face detector and Phase 12 CleanBTNVit emotion model...")
    ai_status = initialize_ai_runtime(warmup=True)
    print(
        f"[Startup] AI runtime ready: model={ai_status['model']} "
        f"(loaded={ai_status['model_loaded']}, device={ai_status['device']}, "
        f"mode={ai_status['default_inference_mode']}), "
        f"detector={ai_status['detector']} (loaded={ai_status['detector_loaded']})"
    )
    yield
    print("[Shutdown] VibeShift server shutting down.")


app = FastAPI(
    title="VibeShift API",
    description="Backend API for VibeShift AI adaptive music recommendation web app",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommendations_router)
app.include_router(sessions_router)
app.include_router(feedback_router)
app.include_router(emotion_router)


@app.get("/api/health")
def health_check():
    """Health check endpoint reporting server and single-model CleanBTNVit runtime readiness."""
    return {
        "status": "ok",
        "ai": get_ai_runtime_status(),
    }
