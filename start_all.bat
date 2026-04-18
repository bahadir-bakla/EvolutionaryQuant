@echo off
title EvolutionaryQuant - Launcher
echo ==========================================
echo   EvolutionaryQuant Trading System
echo   Paper Mode Launcher
echo ==========================================
echo.

set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo [1/5] Starting MiroFish (Docker)...
docker-compose -f mirofish/docker-compose.yml up -d
timeout /t 5 /nobreak >nul

echo [2/5] Starting Dashboard on port 8080...
start "Dashboard" cmd /k "python dashboard/app.py"
timeout /t 3 /nobreak >nul

echo [3/5] Starting ML Scalper (paper mode)...
start "ML Scalper" cmd /k "python ml_scalper_live.py --paper 2>&1 | tee logs/ml_scalper/ml_scalper.log"
timeout /t 2 /nobreak >nul

echo [4/5] Starting Algo Director (paper mode)...
start "Algo Director" cmd /k "python algo_director.py --paper 2>&1 | tee logs/algo_director/algo_director.log"
timeout /t 2 /nobreak >nul

echo [5/5] Starting Cloudflare Tunnel...
start "CF Tunnel" cmd /k ""C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8080"

echo.
echo All services started!
echo Dashboard: http://localhost:8080
echo Cloudflare URL will appear in the CF Tunnel window.
echo.
pause
