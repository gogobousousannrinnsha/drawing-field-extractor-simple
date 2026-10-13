@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
  echo runtime\python.exe is missing. Keep the full candidate folder together.
  if not defined SIMPLE_NO_PAUSE pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\run.py" prepare --root "%~dp0."
set "SIMPLE_EXIT=%ERRORLEVEL%"
if "%SIMPLE_EXIT%"=="0" "%~dp0runtime\python.exe" -I -B -X utf8 -c "print('\u524d\u51e6\u7406\u304c\u5b8c\u4e86\u3057\u307e\u3057\u305f\u3002\u62bd\u51fa.bat \u3092\u5b9f\u884c\u3067\u304d\u307e\u3059\u3002')"
if "%SIMPLE_EXIT%"=="2" "%~dp0runtime\python.exe" -I -B -X utf8 -c "print('\u4e00\u90e8\u306e\u6587\u66f8\u30fb\u30da\u30fc\u30b8\u306b\u5931\u6557\u3057\u307e\u3057\u305f\u3002\u4fdd\u5b58\u8a18\u9332\u3068\u30ed\u30b0\u3092\u78ba\u8a8d\u3057\u3066\u304f\u3060\u3055\u3044\u3002')"
if not defined SIMPLE_NO_PAUSE pause
exit /b %SIMPLE_EXIT%
