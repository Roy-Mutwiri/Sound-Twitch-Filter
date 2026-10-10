@echo off
setlocal
cd /d "%~dp0"
REM 1) build_exe.bat must have produced dist\StreamVoiceFocus. 2) Needs Inno Setup 6 (winget install JRSoftware.InnoSetup).
set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist %ISCC% ( echo Inno Setup 6 not found. Install with: winget install JRSoftware.InnoSetup & exit /b 1 )
if not exist "dist\StreamVoiceFocus\StreamVoiceFocus.exe" ( echo Run build_exe.bat first. & exit /b 1 )
%ISCC% installer.iss || exit /b 1
echo Installer written to installer_out\
endlocal
