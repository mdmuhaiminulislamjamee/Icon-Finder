@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\dashboard\start.ps1" -OpenBrowser
if errorlevel 1 pause
