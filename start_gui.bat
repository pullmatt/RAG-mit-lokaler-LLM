@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "chemie_gui.py"
) else (
  python "chemie_gui.py"
)
pause
