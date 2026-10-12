@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
  echo 先に初期設定.batを実行してください。
  if not defined SIMPLE_NO_PAUSE pause
  exit /b 1
)
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0scripts\setup_saluki.py" --root "%~dp0." %*
set "APP_EXIT=%errorlevel%"
echo Exit code: %APP_EXIT%
if not defined SIMPLE_NO_PAUSE pause
exit /b %APP_EXIT%
