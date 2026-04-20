@echo off
title EvolutionaryQuant - Trading System
cd /d "%~dp0"

if not exist "logs\ml_scalper"    mkdir "logs\ml_scalper"
if not exist "logs\algo_director" mkdir "logs\algo_director"

echo [1/3] Starting Dashboard (port 8081)...
start "Dashboard" cmd /k "python dashboard\app.py"
timeout /t 3 /nobreak >nul

echo [2/3] Starting ML Scalper (LIVE)...
start "ML Scalper" cmd /k "python ml_scalper_live.py"
timeout /t 2 /nobreak >nul

echo [3/3] Starting Algo Director (Gold only, NQ disabled)...
start "Algo Director" cmd /k "python algo_director.py --disable-nq"

echo.
echo Sistem calisiyor!
echo Dashboard: http://localhost:8081
echo MiroFish:  http://localhost:5001
pause
