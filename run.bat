@echo off
setlocal

rem ====================================================================
rem Project launcher -- pinned to base env D:\anaconda\python.exe
rem Bypasses conda activate / PATH issues to avoid ModuleNotFoundError
rem
rem Usage:
rem   run.bat                 -> python -m src.batch_analysis  (default)
rem   run.bat main            -> python -m src.main
rem   run.bat --skip-render   -> forward any args to batch_analysis
rem ====================================================================

set PY=D:\anaconda\python.exe

if not exist "%PY%" (
    echo [ERROR] Interpreter not found: %PY%
    echo         Confirm Anaconda is installed at D:\anaconda, or edit PY in this script.
    exit /b 1
)

cd /d "%~dp0"

if "%1"=="main" (
    "%PY%" -m src.main
) else (
    "%PY%" -m src.batch_analysis %*
)

endlocal
