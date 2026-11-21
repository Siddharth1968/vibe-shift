@echo off
setlocal
title VibeShift Backend (FastAPI :8000)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Local Python environment .venv not found.
    echo         Please run setup.bat first.
    pause
    exit /b 1
)

echo ========================================================
echo  VibeShift Backend Server (FastAPI + Uvicorn)
echo  Running on: http://127.0.0.1:8000
echo ========================================================
".venv\Scripts\python.exe" -B -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
endlocal
