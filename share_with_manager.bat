@echo off
title Sentinel Gaze - Share with Manager
echo =========================================================================
echo   SENTINEL GAZE // AI TELEPROMPTER & PROCTORING (PUBLIC INTERNET SHARE)
echo =========================================================================
echo.
echo [1/2] Starting local Proctoring Server on port 8000...
start /b python server.py
timeout /t 3 >nul

echo [2/2] Generating free public HTTPS link for your manager...
echo.
echo =========================================================================
echo   SEND THE PUBLIC URL SHOWN BELOW TO YOUR MANAGER:
echo =========================================================================
echo.
npx localtunnel --port 8000
pause
