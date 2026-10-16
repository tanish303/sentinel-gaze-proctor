@echo off
title Push Sentinel Gaze to GitHub
echo =========================================================================
echo   SENTINEL GAZE // PUSH TO GITHUB REPOSITORY
echo =========================================================================
echo.
echo Step 1: Go to https://github.com/new and create an empty repository.
echo         Example name: sentinel-gaze-proctor
echo.
set /p REPO_URL="Step 2: Paste your GitHub repository URL (e.g. https://github.com/tanish303/sentinel-gaze-proctor.git): "

if "%REPO_URL%"=="" (
    echo Error: No URL entered. Exiting.
    pause
    exit /b
)

echo.
echo Linking repository to %REPO_URL% ...
git remote remove origin 2>nul
git remote add origin %REPO_URL%
git branch -M main

echo Pushing code to GitHub...
git push -u origin main

echo.
echo =========================================================================
echo   SUCCESSFULLY PUSHED TO GITHUB!
echo =========================================================================
echo Next Step: Go to https://render.com -> New Web Service -> select this repo.
echo It will automatically build and give you a permanent live public link!
echo =========================================================================
pause
