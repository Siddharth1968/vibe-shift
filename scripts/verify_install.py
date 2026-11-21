"""
Runtime verification and launcher probe utility for VibeShift Windows setup/launch scripts.

Modes:
  (default)        Full post-setup verification:
                   1. SQLite schema & 137-track Spotify catalogue bootstrap.
                   2. YuNet ONNX SHA-256 verification and live cv2.FaceDetectorYN loading.
                   3. CleanBTNVit Phase 12 SHA-256 verification, actual PyTorch load,
                      and 7-class forward pass (if checkpoint is installed).
                   4. FastAPI application lifespan startup & /api/health verification.
                   Exit codes: 0 = Full AI Ready, 2 = Partial (AI checkpoint missing), 1 = Failure.
  --check-cuda     Inspect nvidia-smi and print CUDA_OK (driver >= 12.1) or NO_CUDA.
  --port-state P   Print FREE, VIBESHIFT, or OCCUPIED for localhost port 8000 or 5173.
  --wait-backend   Poll http://127.0.0.1:8000/api/health until healthy (with --expect-ai check).
  --wait-frontend  Poll http://localhost:5173 until HTTP 200.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_python_version() -> int:
    ver = sys.version_info[:2]
    if (3, 10) <= ver <= (3, 12):
        print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return 0
    return 1


def _parse_cuda_driver_ok(nvidia_smi_output: str) -> bool:
    m = re.search(r"CUDA(?:\s+\w+)*\s+Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    return bool(m and (int(m.group(1)), int(m.group(2))) >= (12, 1))


def check_cuda_driver() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        if _parse_cuda_driver_ok(out):
            print("CUDA_OK")
            return 0
    except Exception:
        pass
    print("NO_CUDA")
    return 0


def check_port_state(port: int) -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        busy = sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()

    if not busy:
        print("FREE")
        return 0

    try:
        if port == 8000:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=2) as r:
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
            if data.get("status") == "ok" and "ai" in data:
                print("VIBESHIFT")
                return 0
        elif port == 5173:
            with urllib.request.urlopen("http://localhost:5173", timeout=2) as r:
                body = r.read().decode("utf-8", errors="ignore")
            if "VibeShift" in body or "vite" in body.lower():
                print("VIBESHIFT")
                return 0
    except Exception:
        pass

    print("OCCUPIED")
    return 0


def wait_for_backend(expect_ai: bool, timeout_s: float = 90.0) -> int:
    deadline = time.time() + timeout_s
    last_err: str = "no response yet"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=3) as r:
                data = json.loads(r.read().decode("utf-8", errors="ignore"))
            if data.get("status") == "ok" and "ai" in data:
                ai = data["ai"]
                print(
                    f"[OK] Backend healthy: {ai.get('model')} | device={ai.get('device')} | "
                    f"model_loaded={ai.get('model_loaded')} | detector_loaded={ai.get('detector_loaded')}"
                )
                if not ai.get("detector_loaded"):
                    print("[ERROR] YuNet face detector failed to load on backend startup.")
                    return 1
                if expect_ai and not ai.get("model_loaded"):
                    print(
                        "[ERROR] CleanBTNVit AI model failed to load on backend startup. "
                        "Inspect the VibeShift Backend terminal window for details."
                    )
                    return 1
                return 0
        except Exception as exc:
            last_err = str(exc)
        time.sleep(1.0)

    print(
        f"[ERROR] Backend did not become healthy within {timeout_s:.0f}s ({last_err}). "
        "Inspect the VibeShift Backend terminal window."
    )
    return 1


def wait_for_frontend(timeout_s: float = 45.0) -> int:
    deadline = time.time() + timeout_s
    last_err: str = "no response yet"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://localhost:5173", timeout=2) as r:
                if r.status == 200:
                    print("[OK] Frontend ready at http://localhost:5173")
                    return 0
        except Exception as exc:
            last_err = str(exc)
        time.sleep(0.8)

    print(
        f"[ERROR] Frontend did not respond on http://localhost:5173 within {timeout_s:.0f}s ({last_err}). "
        "Inspect the VibeShift Frontend terminal window."
    )
    return 1


def run_full_verification() -> int:
    import torch

    from backend.ai import (
        CLEAN_BTN_EXPECTED_SHA256,
        YUNET_EXPECTED_SHA256,
        get_default_detector,
        get_default_model,
        initialize_ai_runtime,
    )
    from backend.database.db import DB_PATH, init_db, load_songs_csv_if_empty
    from backend.main import app, health_check, lifespan

    print("[Verify 1/4] Checking SQLite database and 137-song catalogue...")
    try:
        init_db()
        load_songs_csv_if_empty()
        conn = sqlite3.connect(DB_PATH)
        song_count = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        conn.close()
        if song_count < 137:
            print(f"[ERROR] Expected 137 songs in SQLite database, found {song_count}.")
            return 1
        print(f"[OK] SQLite ready ({song_count} curated Spotify tracks loaded at {DB_PATH.name}).")
    except Exception as exc:
        print(f"[ERROR] SQLite initialization failed: {exc}")
        return 1

    print("[Verify 2/4] Verifying YuNet ONNX face detector...")
    yunet_path = REPO_ROOT / "backend" / "ai" / "models" / "yunet-2023mar.onnx"
    if not yunet_path.is_file():
        print(f"[ERROR] YuNet model missing at {yunet_path}")
        return 1
    yunet_sha = _sha256(yunet_path)
    if yunet_sha != YUNET_EXPECTED_SHA256:
        print(f"[ERROR] YuNet SHA-256 mismatch: {yunet_sha}")
        return 1
    detector = get_default_detector()
    if not detector.is_available:
        print(f"[ERROR] YuNet failed to initialize: {detector.load_error}")
        return 1
    print(f"[OK] YuNet face detector verified and loaded (SHA-256 {yunet_sha[:12]}...).")

    print("[Verify 3/4] Checking Phase 12 CleanBTNVit checkpoint and actual model loading...")
    ckpt_path = REPO_ROOT / "backend" / "ai" / "models" / "clean_btn_vit_phase12.pt"
    weights_present = ckpt_path.is_file()
    if weights_present:
        if ckpt_path.stat().st_size <= 1024 and ckpt_path.read_bytes()[:64].startswith(
            b"version https://git-lfs.github.com/spec/v1"
        ):
            print("[ERROR] backend/ai/models/clean_btn_vit_phase12.pt is a Git LFS pointer, not the 245.3 MiB weights.")
            print("        Run 'git lfs install && git lfs pull' or install the authorized local checkpoint with:")
            print('          .venv\\Scripts\\python.exe scripts\\install_model.py --from-file "PATH\\TO\\clean_btn_vit_phase12.pt"')
            return 1
        ckpt_sha = _sha256(ckpt_path)
        if ckpt_sha != CLEAN_BTN_EXPECTED_SHA256:
            print("[ERROR] CleanBTNVit checkpoint SHA-256 mismatch!")
            print(f"        Expected: {CLEAN_BTN_EXPECTED_SHA256}")
            print(f"        Actual:   {ckpt_sha}")
            return 1

    ai_status = initialize_ai_runtime(warmup=True)
    model = get_default_model()

    if weights_present:
        if not model.is_weights_loaded:
            print(f"[ERROR] CleanBTNVit failed to load despite file presence: {model.load_error}")
            return 1
        dummy = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
        probs, inf_ms, mode = model.predict_probabilities(dummy, horizontal_flip=False)
        if probs is None or len(probs) != 7 or abs(float(probs.sum()) - 1.0) > 1e-2:
            print("[ERROR] CleanBTNVit forward-pass validation failed.")
            return 1
        print(
            f"[OK] CleanBTNVit Phase 12 loaded & verified on {ai_status['device']} "
            f"({mode} test pass: {inf_ms:.1f} ms)."
        )
    else:
        print(
            "[NOTICE] CleanBTNVit checkpoint is NOT installed at "
            "backend/ai/models/clean_btn_vit_phase12.pt."
        )
        print("         AI facial-expression recognition is NOT active yet.")

    print("[Verify 4/4] Verifying FastAPI application startup & /api/health...")
    try:
        async def _probe_lifespan():
            async with lifespan(app):
                return health_check()

        health = asyncio.run(_probe_lifespan())
        if health.get("status") != "ok":
            print(f"[ERROR] FastAPI health check returned unexpected status: {health}")
            return 1
        if not health["ai"].get("detector_loaded"):
            print("[ERROR] FastAPI health reports detector_loaded=false.")
            return 1
        if weights_present and not health["ai"].get("model_loaded"):
            print("[ERROR] FastAPI health reports model_loaded=false.")
            return 1
        print(
            f"[OK] FastAPI startup & /api/health verified "
            f"(model_loaded={health['ai']['model_loaded']}, "
            f"detector_loaded={health['ai']['detector_loaded']})."
        )
    except Exception as exc:
        print(f"[ERROR] FastAPI startup check failed: {exc}")
        return 1

    return 0 if weights_present else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VibeShift runtime verification and launcher helper.")
    parser.add_argument("--check-python", action="store_true", help="Exit 0 and print version if Python is 3.10-3.12.")
    parser.add_argument("--check-cuda", action="store_true", help="Print CUDA_OK if NVIDIA CUDA >= 12.1 driver exists.")
    parser.add_argument("--port-state", type=int, choices=[8000, 5173], help="Check port state (FREE/VIBESHIFT/OCCUPIED).")
    parser.add_argument("--wait-backend", action="store_true", help="Wait for backend /api/health.")
    parser.add_argument("--expect-ai", action="store_true", help="Require model_loaded=true with --wait-backend.")
    parser.add_argument("--wait-frontend", action="store_true", help="Wait for frontend http://localhost:5173.")
    args = parser.parse_args(argv)

    if args.check_python:
        return check_python_version()
    if args.check_cuda:
        return check_cuda_driver()
    if args.port_state is not None:
        return check_port_state(args.port_state)
    if args.wait_backend:
        return wait_for_backend(expect_ai=args.expect_ai)
    if args.wait_frontend:
        return wait_for_frontend()
    return run_full_verification()


if __name__ == "__main__":
    sys.exit(main())
