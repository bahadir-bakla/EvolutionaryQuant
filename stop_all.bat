@echo off
title Stop All Services
echo Stopping all trading services...

taskkill /f /fi "WINDOWTITLE eq Dashboard" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq ML Scalper*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq Algo Director*" >nul 2>&1
taskkill /f /fi "WINDOWTITLE eq CF Tunnel" >nul 2>&1

cd /d "%~dp0"
docker-compose -f mirofish/docker-compose.yml down

echo All services stopped.
pause
