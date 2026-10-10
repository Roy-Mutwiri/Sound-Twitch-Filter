@echo off
setlocal
cd /d "%~dp0"
REM Builds dist\StreamVoiceFocus\StreamVoiceFocus.exe (onedir). PyTorch is deliberately excluded:
REM the exe ships the exported DeepFilterNet3 ONNX graph instead (models_cache\dfn3_stream.onnx).
if not exist "models_cache\dfn3_stream.onnx" (
  echo models_cache\dfn3_stream.onnx missing - run: .venv\Scripts\python.exe tools\export_onnx.py
  exit /b 1
)
set PYTHONPATH=%~dp0
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onedir --windowed --name StreamVoiceFocus --icon "assets\icon.ico" ^
  --add-data "assets;assets" ^
  --add-data "models_cache\dfn3_stream.onnx;models_cache" ^
  --collect-all dpdfnet --collect-all pyrnnoise --collect-all onnxruntime --collect-all soxr ^
  --collect-all sounddevice --collect-all soundfile --hidden-import backports.tarfile ^
  --exclude-module torch --exclude-module torchaudio --exclude-module df --exclude-module librosa ^
  --exclude-module matplotlib --exclude-module pesq --exclude-module pystoi --exclude-module pytest ^
  --exclude-module PySide6.QtWebEngineCore --exclude-module PySide6.QtWebEngineWidgets --exclude-module PySide6.Qt3DCore ^
  --exclude-module PySide6.QtQuick --exclude-module PySide6.QtQml --exclude-module PySide6.QtPdf --exclude-module PySide6.QtCharts ^
  --exclude-module PySide6.QtMultimedia --exclude-module PySide6.QtDataVisualization ^
  app_entry.py
if errorlevel 1 exit /b 1
echo Built dist\StreamVoiceFocus\StreamVoiceFocus.exe
endlocal
