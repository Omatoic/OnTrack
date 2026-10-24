@echo off
cd /d "%~dp0"
REM OnTrack must run on a Python that has the roto editor's dependencies
REM (torch, av, PySide6). Prefer the dedicated roto venv; then the
REM Sammie-Roto install's venv; then the plain .venv (no roto editor).
if exist "roto\.venv\Scripts\python.exe" (
    roto\.venv\Scripts\python.exe main.py
) else if exist "P:\Program Files\Sammie Roto\Sammie-Roto-2\.venv\Scripts\python.exe" (
    "P:\Program Files\Sammie Roto\Sammie-Roto-2\.venv\Scripts\python.exe" main.py
) else (
    .venv\Scripts\python main.py
)
pause
