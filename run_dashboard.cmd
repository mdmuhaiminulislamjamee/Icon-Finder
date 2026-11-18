@echo off
title MediaControl Collector Studio
echo ======================================================================
echo           Starting MediaControl Studio Dashboard...
echo ======================================================================
echo.
echo Dashboard URL: http://localhost:8000
echo.
start "" http://localhost:8000
py tools\dashboard\server.py
pause
