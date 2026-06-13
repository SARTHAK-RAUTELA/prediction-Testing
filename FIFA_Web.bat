@echo off
title FIFA 2026 - Web Predictor
chcp 65001 > nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo  Starting FIFA 2026 Web Predictor...
echo  Opening browser at http://localhost:8501
echo.

".venv\Scripts\streamlit.exe" run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false

pause
