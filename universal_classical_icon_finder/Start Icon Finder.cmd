@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -m streamlit run app.py --server.headless false
) else (
  python -m streamlit run app.py --server.headless false
)
if errorlevel 1 pause
