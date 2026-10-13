@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist "%~dp0runtime\python.exe" (
    echo 初期設定を行ってから再実行してください。
    if not defined SIMPLE_NO_PAUSE pause
    exit /b 1
)
if "%~1"=="" goto PICK_FILES
if "%~2"=="" goto ONE_FILE
if not "%~3"=="" goto EXPLICIT_DOCUMENT
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\marker_run.py" --root "%~dp0." "%~1" "%~2"
goto FINISH
:EXPLICIT_DOCUMENT
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\marker_run.py" --root "%~dp0." "%~1" "%~2" --document "%~3"
goto FINISH
:ONE_FILE
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\marker_run.py" --root "%~dp0." "%~1"
goto FINISH
:PICK_FILES
"%~dp0runtime\python.exe" -I -B -X utf8 "%~dp0source\marker_run.py" --root "%~dp0."
:FINISH
set "SIMPLE_EXIT=%ERRORLEVEL%"
if not "%SIMPLE_EXIT%"=="0" echo エラーが発生しました。上のメッセージを確認してください。
if not defined SIMPLE_NO_PAUSE pause
exit /b %SIMPLE_EXIT%
