"""
Verified local installer for the authorized Phase 12 CleanBTNVit checkpoint.

Supports installing from:
  1. A locally transferred file (--from-file / VIBESHIFT_MODEL_PATH), e.g., USB or
     private university OneDrive/Google Drive download shared among academic collaborators.
  2. A user-supplied private URL (--url / VIBESHIFT_MODEL_URL). No public URL is
     hardcoded because upstream dataset licenses restrict public redistribution.
  3. Verification of an already installed file (--verify-only).

Always validates exact SHA-256 before placing weights into backend/ai/models/.
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


def verify_target(path: Path = TARGET_PATH) -> bool:
    if not path.is_file():
        print(f"[MISSING] Checkpoint not found at: {path}")
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
    return True


def install_from_file(src_path: Path) -> int:
    if not src_path.is_file():
        print(f"[ERROR] Source file not found: {src_path}")
        return 1
    print(f"[INFO] Computing SHA-256 for {src_path} ...")
    actual = sha256_file(src_path)
    if actual.lower() != EXPECTED_SHA256.lower():
        print("[ERROR] Refusing to install: SHA-256 does not match Phase 12 CleanBTNVit.")
        print(f"        Expected: {EXPECTED_SHA256}")
        print(f"        Actual:   {actual}")
        return 1

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != TARGET_PATH.resolve():
        tmp_path = TARGET_PATH.with_suffix(".pt.tmp")
        try:
            shutil.copy2(src_path, tmp_path)
            tmp_path.replace(TARGET_PATH)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
    return 0 if verify_target(TARGET_PATH) else 1


def install_from_url(url: str) -> int:
    if not url.lower().startswith(("https://", "http://")):
        print(f"[ERROR] Unsupported URL scheme: {url}")
        return 1

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="clean_btn_dl_", suffix=".pt.tmp", dir=str(TARGET_PATH.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        print("[INFO] Downloading authorized checkpoint from user-provided URL...")
        req = urllib.request.Request(url, headers={"User-Agent": "VibeShift-Model-Installer/1.0"})
        h = hashlib.sha256()
        total = 0
        with urllib.request.urlopen(req, timeout=120) as resp, tmp_path.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                total += len(chunk)
        actual = h.hexdigest()
        print(f"[INFO] Downloaded {total / (1024 * 1024):.1f} MiB. Verifying SHA-256...")
        if actual.lower() != EXPECTED_SHA256.lower():
            print("[ERROR] Downloaded file SHA-256 mismatch; removing temporary file.")
            print(f"        Expected: {EXPECTED_SHA256}")
            print(f"        Actual:   {actual}")
            return 1
        tmp_path.replace(TARGET_PATH)
        return 0 if verify_target(TARGET_PATH) else 1
    except Exception as exc:
        print(f"[ERROR] Failed to download checkpoint: {exc}")
        return 1
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


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
    args = parser.parse_args(argv)

    if args.from_file:
        return install_from_file(args.from_file)
    if args.url:
        return install_from_url(args.url)
    if args.verify_only:
        return 0 if verify_target(TARGET_PATH) else 1

    # Fallback to environment variables when invoked with no flags
    env_file = os.environ.get("VIBESHIFT_MODEL_PATH", "").strip()
    if env_file:
        return install_from_file(Path(env_file))
    env_url = os.environ.get("VIBESHIFT_MODEL_URL", "").strip()
    if env_url:
        return install_from_url(env_url)

    return 0 if verify_target(TARGET_PATH) else 1


if __name__ == "__main__":
    sys.exit(main())
