@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Root "%~dp0."
set "APP_EXIT=%errorlevel%"
echo Exit code: %APP_EXIT%
if not defined SIMPLE_NO_PAUSE pause
exit /b %APP_EXIT%
