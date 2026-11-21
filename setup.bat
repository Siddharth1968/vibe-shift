@echo off
setlocal enabledelayedexpansion
title VibeShift Automated Setup
cd /d "%~dp0"

echo ========================================================
echo  VibeShift - Automated Windows Installation
echo  Single-Model AI: YuNet + CleanBTNVit Phase 12
echo ========================================================
echo.

:: 1. Verify 64-bit Windows architecture
set "ARCH=%PROCESSOR_ARCHITECTURE%"
if defined PROCESSOR_ARCHITEW6432 set "ARCH=%PROCESSOR_ARCHITEW6432%"
if /I not "%ARCH%"=="AMD64" if /I not "%ARCH%"=="ARM64" (
    echo [ERROR] 64-bit Windows is required for PyTorch and OpenCV. Detected: %ARCH%
    exit /b 1
)
echo [OK] Windows %ARCH% environment detected.

:: 2. Detect compatible Python installation (prefer Python 3.12, then 3.11 / 3.10)
set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.12"
    if not defined PYTHON_CMD (
        py -3.11 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.11"
    )
    if not defined PYTHON_CMD (
        py -3.10 -c "import sys" >nul 2>nul && set "PYTHON_CMD=py -3.10"
    )
)
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if not errorlevel 1 (
        python scripts\verify_install.py --check-python >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=python"
    )
)
if not defined PYTHON_CMD (
    echo [ERROR] Python 3.10, 3.11, or 3.12 (64-bit) is required.
    echo         (PyTorch CUDA/CPU wheels do not support Python 3.13/3.14 yet.)
    echo         Download Python 3.12 from:
    echo           https://www.python.org/downloads/release/python-31210/
    echo         Make sure to check "Add python.exe to PATH" during installation.
    exit /b 1
)
for /f "usebackq delims=" %%V in (`%PYTHON_CMD% scripts\verify_install.py --check-python`) do set "PY_VER=%%V"
echo [OK] Using Python %PY_VER% (%PYTHON_CMD%).

:: 3. Detect Node.js (>= 18) and npm
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js 18+ was not found on PATH.
    echo         Download Node.js LTS from: https://nodejs.org/
    exit /b 1
)
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm was not found on PATH. Reinstall Node.js LTS from: https://nodejs.org/
    exit /b 1
)
for /f "delims=" %%N in ('node -v') do set "NODE_VER=%%N"
node -e "process.exit(parseInt(process.versions.node.split('.')[0], 10) >= 18 ? 0 : 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js 18 or newer is required. Found: %NODE_VER%
    echo         Download Node.js LTS from: https://nodejs.org/
    exit /b 1
)
echo [OK] Using Node.js %NODE_VER%.

:: 4. Note Git status (only needed for GitHub clone workflows; not required for ZIP extraction)
where git >nul 2>nul
if errorlevel 1 (
    echo [INFO] Git not detected (not required for ZIP installation; install from https://git-scm.com/ only for Git clone workflows).
) else (
    echo [OK] Git detected.
)

:: 5. Create fresh local Python virtual environment
echo.
echo [1/6] Setting up isolated Python virtual environment (.venv)...
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create local Python virtual environment.
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul

:: 6. Detect NVIDIA GPU & compatible CUDA driver (12.1+), else install CPU PyTorch fallback
echo [2/6] Detecting GPU/CUDA support and installing PyTorch...
set "USE_CUDA=0"
if /I "%VIBESHIFT_FORCE_CPU%"=="1" (
    echo [INFO] VIBESHIFT_FORCE_CPU=1 set - installing CPU PyTorch build.
) else (
    for /f "usebackq delims=" %%G in (`".venv\Scripts\python.exe" scripts\verify_install.py --check-cuda`) do (
        if "%%G"=="CUDA_OK" set "USE_CUDA=1"
    )
)

".venv\Scripts\python.exe" -c "import torch" >nul 2>nul
if errorlevel 1 (
    if "%USE_CUDA%"=="1" (
        echo [INFO] Compatible NVIDIA GPU + CUDA driver detected. Installing PyTorch CUDA 12.6 (~2.6 GB download)...
        ".venv\Scripts\python.exe" -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
        if errorlevel 1 (
            echo [WARNING] CUDA PyTorch installation failed; falling back to CPU PyTorch build...
            ".venv\Scripts\python.exe" -m pip install "torch>=2.5.0"
            if errorlevel 1 (
                echo [ERROR] PyTorch installation failed. Check internet connection and retry.
                exit /b 1
            )
        )
    ) else (
        echo [INFO] No compatible NVIDIA CUDA GPU detected. Installing CPU PyTorch build...
        echo        (CPU fallback is fully supported; note inference is slower than GPU.)
        ".venv\Scripts\python.exe" -m pip install "torch>=2.5.0"
        if errorlevel 1 (
            echo [ERROR] PyTorch CPU installation failed. Check internet connection and retry.
            exit /b 1
        )
    )
) else (
    echo [OK] PyTorch already installed in .venv.
)

:: 7. Install required backend packages
echo [3/6] Installing backend dependencies (requirements.txt)...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install backend requirements. Check internet connection and retry.
    exit /b 1
)

:: 8. Install frontend dependencies via npm ci
echo [4/6] Installing frontend dependencies (npm ci)...
pushd frontend
call npm ci
if errorlevel 1 (
    popd
    echo [ERROR] npm ci failed inside frontend\. Check internet connection and Node.js version.
    exit /b 1
)
popd

:: 9. Install / locate / verify CleanBTNVit Phase 12 checkpoint
echo [5/6] Checking Phase 12 CleanBTNVit emotion checkpoint...
if exist "backend\ai\models\clean_btn_vit_phase12.pt" (
    ".venv\Scripts\python.exe" scripts\install_model.py --verify-only
    if errorlevel 1 (
        echo [ERROR] Existing backend\ai\models\clean_btn_vit_phase12.pt failed SHA-256 verification.
        echo         Remove the corrupted file and re-run setup.bat with the authorized checkpoint.
        exit /b 1
    )
) else if defined VIBESHIFT_MODEL_PATH (
    ".venv\Scripts\python.exe" scripts\install_model.py --from-file "%VIBESHIFT_MODEL_PATH%"
    if errorlevel 1 (
        echo [ERROR] Failed to install checkpoint from VIBESHIFT_MODEL_PATH.
        exit /b 1
    )
) else if defined VIBESHIFT_MODEL_URL (
    ".venv\Scripts\python.exe" scripts\install_model.py --url "%VIBESHIFT_MODEL_URL%"
    if errorlevel 1 (
        echo [ERROR] Failed to install checkpoint from VIBESHIFT_MODEL_URL.
        exit /b 1
    )
) else if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" (
    ".venv\Scripts\python.exe" scripts\install_model.py --interactive
    if "!ERRORLEVEL!"=="1" (
        echo [ERROR] Provided checkpoint failed verification. Re-run setup.bat with the valid file.
        exit /b 1
    )
)

:: 10. Full runtime verification (DB + 137 songs + YuNet + CleanBTNVit load + FastAPI startup)
echo.
echo [6/6] Running end-to-end installation verification...
".venv\Scripts\python.exe" scripts\verify_install.py
set "VERIFY_RC=%ERRORLEVEL%"

echo.
if "%VERIFY_RC%"=="0" (
    echo ========================================================
    echo  SUCCESS: Full VibeShift AI installation verified!
    echo  - YuNet face detector:     READY
    echo  - CleanBTNVit Phase 12 AI: READY (SHA-256 verified)
    echo  - FastAPI + SQLite + UI:   READY
    echo.
    echo  Launch VibeShift anytime by double-clicking:
    echo      start_vibeshift.bat
    echo ========================================================
    exit /b 0
)

if "%VERIFY_RC%"=="2" (
    echo ========================================================
    echo  PARTIAL SETUP: Core App Ready - AI MODEL NOT INSTALLED
    echo ========================================================
    echo  - FastAPI + SQLite (137 songs) + React UI: READY
    echo  - YuNet face detector:                     READY
    echo  - CleanBTNVit Phase 12 AI checkpoint:      MISSING
    echo.
    echo  IMPORTANT: Full camera AI emotion recognition is NOT
    echo  active yet. Manual Mood Mode works right now, and to
    echo  enable real AI inference, obtain the authorized file
    echo  'clean_btn_vit_phase12.pt' and run:
    echo.
    echo    .venv\Scripts\python.exe scripts\install_model.py --from-file "PATH\TO\clean_btn_vit_phase12.pt"
    echo.
    echo  See FRIEND_INSTALLATION_GUIDE.md and docs\MODEL_SETUP.md.
    echo ========================================================
    exit /b 0
)

echo ========================================================
echo  SETUP FAILED (error code %VERIFY_RC%). Review messages above.
echo ========================================================
exit /b %VERIFY_RC%
endlocal
