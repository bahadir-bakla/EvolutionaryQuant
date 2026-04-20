@echo off
title EvolutionaryQuant - Trading System
cd /d "%~dp0"

if not exist "logs\ml_scalper"    mkdir "logs\ml_scalper"
if not exist "logs\algo_director" mkdir "logs\algo_director"

echo [1/3] Starting Dashboard...
start "Dashboard" cmd /k "python dashboard\app.py"
timeout /t 3 /nobreak >nul

echo [2/3] Starting ML Scalper (LIVE)...
start "ML Scalper" cmd /k "python ml_scalper_live.py"
timeout /t 2 /nobreak >nul

echo [3/3] Starting Algo Director (LIVE, NQ=USTEC)...
start "Algo Director" cmd /k "python algo_director.py --nq-symbol USTEC"

echo.
echo Sistem calisiyor! Dashboard: http://localhost:8080
pause
