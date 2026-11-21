@echo off
setlocal enabledelayedexpansion
title VibeShift Launcher
cd /d "%~dp0"

echo ========================================================
echo  VibeShift - One-Click Application Launcher
echo ========================================================

:: 1. Verify installation prerequisites
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment (.venv) is not installed.
    echo         Please run setup.bat first.
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)
if not exist "frontend\node_modules" (
    echo [ERROR] Frontend dependencies (frontend\node_modules) are not installed.
    echo         Please run setup.bat first.
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)
if not exist "backend\ai\models\yunet-2023mar.onnx" (
    echo [ERROR] Required YuNet face detector is missing:
    echo         backend\ai\models\yunet-2023mar.onnx
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)

:: 2. Verify CleanBTNVit Phase 12 checkpoint before launching
set "EXPECT_AI=1"
if not exist "backend\ai\models\clean_btn_vit_phase12.pt" (
    set "EXPECT_AI=0"
    echo.
    echo [WARNING] CleanBTNVit AI checkpoint is NOT installed at:
    echo           backend\ai\models\clean_btn_vit_phase12.pt
    echo           Camera facial-expression recognition will NOT work until installed.
    echo           Install with:
    echo             .venv\Scripts\python.exe scripts\install_model.py --from-file "PATH\TO\clean_btn_vit_phase12.pt"
    echo.
    if /I not "%VIBESHIFT_ALLOW_MANUAL_ONLY%"=="1" (
        choice /C YN /M "Continue launching in Manual Mood Mode only"
        if errorlevel 2 exit /b 1
    )
) else (
    ".venv\Scripts\python.exe" scripts\install_model.py --verify-only
    if errorlevel 1 (
        echo [ERROR] Checkpoint SHA-256 verification failed. Refusing to start with invalid weights.
        if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
        exit /b 1
    )
)

:: 3. Check port 8000 (reuse running VibeShift backend; report conflicts cleanly)
set "BACKEND_STATE=FREE"
for /f "usebackq delims=" %%S in (`".venv\Scripts\python.exe" scripts\verify_install.py --port-state 8000`) do set "BACKEND_STATE=%%S"

if "%BACKEND_STATE%"=="OCCUPIED" (
    echo [ERROR] Port 8000 is already in use by another application.
    echo         Close the conflicting process on port 8000 and retry.
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)

if "%BACKEND_STATE%"=="VIBESHIFT" (
    echo [INFO] VibeShift backend is already running on http://127.0.0.1:8000 - reusing it.
) else (
    echo [1/3] Starting FastAPI backend on http://127.0.0.1:8000 ...
    start "VibeShift Backend (FastAPI :8000)" /D "%~dp0" cmd /k run_backend.bat
)

:: 4. Wait for backend /api/health and verify model_loaded & detector_loaded
echo [2/3] Waiting for backend health and AI initialization...
if "%EXPECT_AI%"=="1" (
    ".venv\Scripts\python.exe" scripts\verify_install.py --wait-backend --expect-ai
) else (
    ".venv\Scripts\python.exe" scripts\verify_install.py --wait-backend
)
if errorlevel 1 (
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)

:: 5. Check port 5173 (reuse running VibeShift frontend; report conflicts cleanly)
set "FRONTEND_STATE=FREE"
for /f "usebackq delims=" %%F in (`".venv\Scripts\python.exe" scripts\verify_install.py --port-state 5173`) do set "FRONTEND_STATE=%%F"

if "%FRONTEND_STATE%"=="OCCUPIED" (
    echo [ERROR] Port 5173 is already in use by another application.
    echo         Close the conflicting process on port 5173 and retry.
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)

if "%FRONTEND_STATE%"=="VIBESHIFT" (
    echo [INFO] VibeShift frontend is already running on http://localhost:5173 - reusing it.
) else (
    echo [3/3] Starting React/Vite frontend on http://localhost:5173 ...
    start "VibeShift Frontend (Vite :5173)" /D "%~dp0" cmd /k run_frontend.bat
)

:: 6. Wait until frontend responds, then open default browser
".venv\Scripts\python.exe" scripts\verify_install.py --wait-frontend
if errorlevel 1 (
    if /I not "%VIBESHIFT_NONINTERACTIVE%"=="1" pause
    exit /b 1
)

echo.
echo ========================================================
if "%EXPECT_AI%"=="1" (
    echo  VibeShift is LIVE with Full CleanBTNVit + YuNet AI!
) else (
    echo  VibeShift is LIVE in Manual Mood Mode (AI weights missing)
)
echo  Frontend: http://localhost:5173
echo  Backend:  http://127.0.0.1:8000
echo  Server logs remain visible in the Backend and Frontend windows.
echo ========================================================

if /I not "%VIBESHIFT_NO_BROWSER%"=="1" (
    start "" "http://localhost:5173"
)
endlocal
