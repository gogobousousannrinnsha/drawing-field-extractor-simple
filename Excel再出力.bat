@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
  echo runtime\python.exe is missing. Keep the full candidate folder together.
  if not defined SIMPLE_NO_PAUSE pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\run.py" export --root "%~dp0."
set "SIMPLE_EXIT=%ERRORLEVEL%"
if "%SIMPLE_EXIT%"=="2" "%~dp0runtime\python.exe" -I -B -X utf8 -c "print('\u4e00\u90e8\u306ePDF\u30fb\u30da\u30fc\u30b8\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002Excel\u306e\u5099\u8003\u3068JSON\u8a18\u9332\u3092\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044\u3002')"
if not defined SIMPLE_NO_PAUSE pause
exit /b %SIMPLE_EXIT%
