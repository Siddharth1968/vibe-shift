"""
Verified local installer for the authorized Phase 12 CleanBTNVit checkpoint.

Supports:
  1. Automatic detection & SHA-256 verification of an already-packaged/installed checkpoint.
  2. Installing from a locally transferred authorized file (--from-file / VIBESHIFT_MODEL_PATH).
  3. Installing from a user-supplied private authorized URL (--url / VIBESHIFT_MODEL_URL).
  4. Interactive prompt (--interactive) allowing drag-and-drop / pasted local paths.
  5. Optional actual PyTorch model-loading verification (--check-load).

No public download URL or private token is hardcoded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

EXPECTED_SHA256 = "0316ea2a96e671f562a770ace64b8665e8bc1996be0046cbe94c4674bb0c5d58"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = REPO_ROOT / "backend" / "ai" / "models" / "clean_btn_vit_phase12.pt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_git_lfs_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        head = path.read_bytes()[:64]
        return head.startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def verify_target(path: Path = TARGET_PATH, check_load: bool = False) -> bool:
    if not path.is_file():
        print(f"[MISSING] CleanBTNVit checkpoint not found at: {path}")
        return False
    if is_git_lfs_pointer(path):
        print(f"[ERROR] {path} is a Git LFS pointer file (not the 245.3 MiB model weights).")
        print("        Install Git LFS (https://git-lfs.com/) and run:")
        print("          git lfs install && git lfs pull")
        print("        OR install from a local authorized copy:")
        print('          .venv\\Scripts\\python.exe scripts\\install_model.py --from-file "PATH\\TO\\clean_btn_vit_phase12.pt"')
        return False
    actual = sha256_file(path)
    if actual.lower() != EXPECTED_SHA256.lower():
        print(f"[ERROR] SHA-256 mismatch at {path}")
        print(f"        Expected: {EXPECTED_SHA256}")
        print(f"        Actual:   {actual}")
        return False

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"[OK] Verified Phase 12 CleanBTNVit checkpoint ({size_mb:.1f} MiB)")
    print(f"     Path:    {path}")
    print(f"     SHA-256: {actual}")

    if check_load:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        try:
            from backend.ai import CleanBTNVitAdapter
            import torch

            adapter = CleanBTNVitAdapter(weights_path=str(path))
            if not adapter.is_weights_loaded:
                print(f"[ERROR] Model failed to load: {adapter.load_error}")
                return False
            dummy = torch.zeros((1, 3, 112, 112), dtype=torch.float32)
            probs, ms, mode = adapter.predict_probabilities(dummy, horizontal_flip=False)
            if probs is None or len(probs) != 7:
                print("[ERROR] Model inference check failed to return 7 class probabilities.")
                return False
            print(
                f"[OK] Model load & inference test passed on {adapter.device_name} "
                f"({mode}, {ms:.1f} ms)"
            )
        except Exception as exc:
            print(f"[ERROR] Model load verification raised an exception: {exc}")
            return False

    return True


def install_from_file(src_path: Path, check_load: bool = False) -> int:
    cleaned = Path(str(src_path).strip().strip('"').strip("'"))
    if not cleaned.is_file():
        print(f"[ERROR] Source file not found: {cleaned}")
        return 1

    print(f"[INFO] Verifying SHA-256 of {cleaned} ...")
    actual = sha256_file(cleaned)
    if actual.lower() != EXPECTED_SHA256.lower():
        print("[ERROR] Refusing to install: SHA-256 does not match Phase 12 CleanBTNVit.")
        print(f"        Expected: {EXPECTED_SHA256}")
        print(f"        Actual:   {actual}")
        return 1

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if cleaned.resolve() != TARGET_PATH.resolve():
        tmp_path = TARGET_PATH.with_suffix(".pt.tmp")
        try:
            print(f"[INFO] Copying verified checkpoint to {TARGET_PATH} ...")
            shutil.copy2(cleaned, tmp_path)
            tmp_path.replace(TARGET_PATH)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    return 0 if verify_target(TARGET_PATH, check_load=check_load) else 1


def install_from_url(url: str, check_load: bool = False) -> int:
    clean_url = url.strip()
    if not clean_url.lower().startswith(("https://", "http://")):
        print(f"[ERROR] Unsupported URL scheme: {clean_url}")
        return 1

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix="clean_btn_dl_", suffix=".pt.tmp", dir=str(TARGET_PATH.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        print("[INFO] Downloading authorized checkpoint from user-provided URL...")
        req = urllib.request.Request(
            clean_url, headers={"User-Agent": "VibeShift-Model-Installer/1.1"}
        )
        h = hashlib.sha256()
        total = 0
        last_report_mb = 0
        with urllib.request.urlopen(req, timeout=180) as resp, tmp_path.open("wb") as out:
            content_len = resp.headers.get("Content-Length")
            expected_mb = int(content_len) / (1024 * 1024) if content_len and content_len.isdigit() else None
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                total += len(chunk)
                cur_mb = total // (1024 * 1024)
                if cur_mb >= last_report_mb + 25:
                    last_report_mb = cur_mb
                    if expected_mb:
                        print(f"       Downloaded {cur_mb} / {expected_mb:.0f} MiB...", flush=True)
                    else:
                        print(f"       Downloaded {cur_mb} MiB...", flush=True)

        actual = h.hexdigest()
        print(f"[INFO] Download complete ({total / (1024 * 1024):.1f} MiB). Verifying SHA-256...")
        if actual.lower() != EXPECTED_SHA256.lower():
            print("[ERROR] Downloaded file SHA-256 mismatch; removing temporary file.")
            print(f"        Expected: {EXPECTED_SHA256}")
            print(f"        Actual:   {actual}")
            return 1

        tmp_path.replace(TARGET_PATH)
        return 0 if verify_target(TARGET_PATH, check_load=check_load) else 1
    except Exception as exc:
        print(f"[ERROR] Failed to download checkpoint: {exc}")
        return 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def run_interactive(check_load: bool = False) -> int:
    print()
    print("--------------------------------------------------------")
    print(" Phase 12 CleanBTNVit Checkpoint Installation (Required for AI)")
    print("--------------------------------------------------------")
    print(" If you already have the authorized file 'clean_btn_vit_phase12.pt'")
    print(" on your computer or USB drive, drag and drop it here (or paste its")
    print(" full path or private authorized URL) and press Enter.")
    print(" Press Enter on an empty line to skip for now (Manual Mood Mode only).")
    print()
    try:
        user_input = input("Checkpoint file path or URL (blank to skip): ").strip()
    except EOFError:
        return 2

    if not user_input:
        print("[NOTICE] Skipped checkpoint installation. AI emotion recognition is not yet active.")
        return 2

    cleaned = user_input.strip().strip('"').strip("'")
    if cleaned.lower().startswith(("https://", "http://")):
        return install_from_url(cleaned, check_load=check_load)
    return install_from_file(Path(cleaned), check_load=check_load)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Install and SHA-256 verify the Phase 12 CleanBTNVit checkpoint."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from-file",
        type=Path,
        help="Path to an authorized local copy of clean_btn_vit_phase12.pt",
    )
    group.add_argument(
        "--url",
        type=str,
        help="User-supplied private authorized download URL for clean_btn_vit_phase12.pt",
    )
    group.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify whether backend/ai/models/clean_btn_vit_phase12.pt is present and valid",
    )
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt interactively for a local checkpoint file path or private URL",
    )
    parser.add_argument(
        "--check-load",
        action="store_true",
        help="Also load the CleanBTNVit PyTorch model and verify a 7-class forward pass",
    )
    args = parser.parse_args(argv)

    if args.from_file:
        return install_from_file(args.from_file, check_load=args.check_load)
    if args.url:
        return install_from_url(args.url, check_load=args.check_load)
    if args.verify_only:
        return 0 if verify_target(TARGET_PATH, check_load=args.check_load) else 1
    if args.interactive:
        if TARGET_PATH.is_file():
            return 0 if verify_target(TARGET_PATH, check_load=args.check_load) else 1
        return run_interactive(check_load=args.check_load)

    env_file = os.environ.get("VIBESHIFT_MODEL_PATH", "").strip()
    if env_file:
        return install_from_file(Path(env_file), check_load=args.check_load)
    env_url = os.environ.get("VIBESHIFT_MODEL_URL", "").strip()
    if env_url:
        return install_from_url(env_url, check_load=args.check_load)

    return 0 if verify_target(TARGET_PATH, check_load=args.check_load) else 1


if __name__ == "__main__":
    sys.exit(main())
