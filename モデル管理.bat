@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
  echo runtime\python.exe is missing. Keep the full candidate folder together.
  if not defined SIMPLE_NO_PAUSE pause
  exit /b 1
)
set "APP_ROOT=%~dp0."
if not exist "%~dp0runtime\pythonw.exe" goto console
"%~dp0runtime\pythonw.exe" -I -B -X utf8 "%~dp0source\run.py" models-gui --root "%APP_ROOT%"
set "APP_EXIT=%errorlevel%"
if "%APP_EXIT%"=="0" exit /b 0
"%~dp0runtime\python.exe" -I -B -X utf8 -c "print('\u30e2\u30c7\u30eb\u7ba1\u7406\u3092\u8d77\u52d5\u3067\u304d\u307e\u305b\u3093\u3067\u3057\u305f\u3002\u8a3a\u65ad\u306f\u5019\u88dc\u5185\u306elogs\u306b\u4fdd\u5b58\u3055\u308c\u307e\u3059\u3002')"
echo Exit code: %APP_EXIT%
:console
"%~dp0runtime\python.exe" -I -B -X utf8 -c "print('\u901a\u5e38\u306ePython\u3067\u8d77\u52d5\u3057\u3001\u8a3a\u65ad\u3092\u8868\u793a\u3057\u307e\u3059\u3002')"
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\run.py" models-gui --root "%APP_ROOT%"
set "APP_EXIT=%errorlevel%"
if not "%APP_EXIT%"=="0" if not defined SIMPLE_NO_PAUSE pause
exit /b %APP_EXIT%
