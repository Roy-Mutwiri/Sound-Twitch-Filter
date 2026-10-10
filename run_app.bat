@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [Stream Voice Focus] No virtual environment found. Running install.bat first...
  call install.bat || exit /b 1
)
set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" -m stream_voice_focus %*
endlocal
