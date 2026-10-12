@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
  echo runtime\python.exe is missing. Run the initial setup first.
  if not defined SIMPLE_NO_PAUSE pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\run.py" ocr-image --root "%~dp0."
set "SIMPLE_EXIT=%ERRORLEVEL%"
if not defined SIMPLE_NO_PAUSE pause
exit /b %SIMPLE_EXIT%
