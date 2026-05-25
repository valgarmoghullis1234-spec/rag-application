@echo off
cd /d "%~dp0backend"
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
