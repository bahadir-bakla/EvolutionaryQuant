@echo off
title EvolutionaryQuant - LIVE TRADING
echo ==========================================
echo   EvolutionaryQuant - LIVE TRADING MODE
echo   WARNING: Real money at risk!
echo ==========================================
echo.
set /p CONFIRM=Type LIVE to continue:
if /i not "%CONFIRM%"=="LIVE" (
    echo Cancelled.
    pause
    exit /b
)

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

if not exist "logs\ml_scalper"    mkdir "logs\ml_scalper"
if not exist "logs\algo_director" mkdir "logs\algo_director"

echo [1/5] Starting MiroFish (Docker)...
docker-compose -f mirofish/docker-compose.yml up -d
timeout /t 5 /nobreak >nul

echo [2/5] Starting Dashboard on port 8080...
start "Dashboard" cmd /k "python dashboard/app.py"
timeout /t 3 /nobreak >nul

echo [3/5] Starting ML Scalper (LIVE)...
start "ML Scalper LIVE" cmd /k "python ml_scalper_live.py >> logs\ml_scalper\ml_scalper.log 2>&1"
timeout /t 2 /nobreak >nul

echo [4/5] Starting Algo Director (LIVE)...
start "Algo Director LIVE" cmd /k "python algo_director.py >> logs\algo_director\algo_director.log 2>&1"
timeout /t 2 /nobreak >nul

echo [5/5] Starting Cloudflare Tunnel...
start "CF Tunnel" cmd /k "\"C:\Program Files (x86)\cloudflared\cloudflared.exe\" tunnel --url http://localhost:8080"

echo.
echo LIVE TRADING ACTIVE
echo Dashboard: http://localhost:8080
echo.
pause
