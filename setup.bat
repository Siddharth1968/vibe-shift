@echo off
setlocal enabledelayedexpansion
title VibeShift Setup
cd /d "%~dp0"

echo ========================================================
echo  VibeShift — Portable Windows Setup
echo ========================================================

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
    if errorlevel 1 (
        echo [ERROR] Python 3.10-3.12 was not found on PATH.
        echo         Install Python 3.12 from https://www.python.org/downloads/ and retry.
        exit /b 1
    )
    set "PYTHON_CMD=python"
)

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js 18+ was not found on PATH.
    echo         Install Node.js from https://nodejs.org/ and retry.
    exit /b 1
)

echo [1/5] Creating local Python virtual environment (.venv) using %PYTHON_CMD%...
if not exist ".venv\Scripts\python.exe" (
    %PYTHON_CMD% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create Python virtual environment.
        exit /b 1
    )
)

echo [2/5] Installing Python backend dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if /I "%VIBESHIFT_INSTALL_CUDA%"=="1" (
    echo       Installing PyTorch CUDA 12.6 build...
    ".venv\Scripts\python.exe" -m pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install backend Python dependencies.
    exit /b 1
)

echo [3/5] Installing frontend dependencies (npm ci)...
pushd frontend
call npm ci
if errorlevel 1 (
    popd
    echo [ERROR] Failed to install frontend dependencies.
    exit /b 1
)
popd

echo [4/5] Initializing clean SQLite database from songs.csv...
".venv\Scripts\python.exe" -c "from backend.database.db import init_db, load_songs_csv_if_empty; init_db(); load_songs_csv_if_empty()"
if errorlevel 1 (
    echo [ERROR] Database initialization failed.
    exit /b 1
)

echo [5/5] Validating AI model assets...
if not exist "backend\ai\models\yunet-2023mar.onnx" (
    echo [WARNING] Missing YuNet ONNX model at backend\ai\models\yunet-2023mar.onnx
) else (
    echo [OK] YuNet face detector found: backend\ai\models\yunet-2023mar.onnx
)

if not exist "backend\ai\models\clean_btn_vit_phase12.pt" (
    if defined VIBESHIFT_MODEL_PATH (
        echo [INFO] Installing CleanBTNVit checkpoint from VIBESHIFT_MODEL_PATH...
        ".venv\Scripts\python.exe" scripts\install_model.py --from-file "%VIBESHIFT_MODEL_PATH%"
        if errorlevel 1 (
            echo [ERROR] Model installation from VIBESHIFT_MODEL_PATH failed.
            exit /b 1
        )
    ) else if defined VIBESHIFT_MODEL_URL (
        echo [INFO] Downloading CleanBTNVit checkpoint from VIBESHIFT_MODEL_URL...
        ".venv\Scripts\python.exe" scripts\install_model.py --url "%VIBESHIFT_MODEL_URL%"
        if errorlevel 1 (
            echo [ERROR] Model download from VIBESHIFT_MODEL_URL failed.
            exit /b 1
        )
    ) else (
        echo [NOTICE] CleanBTNVit AI checkpoint not found at:
        echo          backend\ai\models\clean_btn_vit_phase12.pt
        echo          For full facial-expression recognition, obtain the authorized checkpoint
        echo          from your academic project team and run:
        echo            .venv\Scripts\python.exe scripts\install_model.py --from-file "PATH\TO\clean_btn_vit_phase12.pt"
        echo          Until installed, the app still runs cleanly in Manual Mood Mode.
        echo          See docs\MODEL_SETUP.md for licensing details and verification steps.
    )
) else (
    echo [OK] CleanBTNVit checkpoint found. Verifying SHA-256...
    ".venv\Scripts\python.exe" scripts\install_model.py --verify-only
    if errorlevel 1 (
        echo [ERROR] Existing checkpoint failed SHA-256 verification.
        exit /b 1
    )
)

echo ========================================================
echo  Setup complete! Run run_backend.bat and run_frontend.bat
echo ========================================================
endlocal
