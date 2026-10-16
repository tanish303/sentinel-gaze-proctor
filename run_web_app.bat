@echo off
title Sentinel Gaze Proctoring Server
echo =========================================================================
echo   SENTINEL GAZE // AI TELEPROMPTER & PROCTORING AUDIT SYSTEM
echo =========================================================================
echo Starting local web server on http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
python server.py
pause
