@echo off
setlocal
cd /d "%~dp0"
echo [Stream Voice Focus] Creating virtual environment (.venv) with Python 3.11 ...
where uv >nul 2>nul
if %errorlevel%==0 (
  uv venv --python 3.11 .venv || exit /b 1
  uv pip install --python .venv\Scripts\python.exe --index-strategy unsafe-best-match -r requirements.txt || exit /b 1
) else (
  py -3.11 -m venv .venv || exit /b 1
  .venv\Scripts\python.exe -m pip install --upgrade pip || exit /b 1
  .venv\Scripts\python.exe -m pip install -r requirements.txt || exit /b 1
)
echo [Stream Voice Focus] Exporting the DeepFilterNet3 streaming ONNX model (one-time, ~1 min) ...
set PYTHONPATH=%~dp0
.venv\Scripts\python.exe tools\export_onnx.py || echo (export failed: the app will fall back to the PyTorch runtime)
echo [Stream Voice Focus] Done. Start with run_app.bat
endlocal
