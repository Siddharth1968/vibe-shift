@echo off
setlocal
title VibeShift Frontend (React / Vite :5173)
cd /d "%~dp0frontend"

if not exist "node_modules" (
    echo [ERROR] Frontend node_modules not found.
    echo         Please run setup.bat - or npm ci inside frontend\ - first.
    pause
    exit /b 1
)

echo ========================================================
echo  VibeShift Frontend Server (Vite Development Server)
echo  Running on: http://localhost:5173
echo ========================================================
call npm run dev
pause
endlocal
