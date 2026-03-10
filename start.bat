@echo off
title BuddyQ Voice Assistant

echo ================================================================================
echo BuddyQ Voice Assistant - Starting
echo ================================================================================
echo.

REM ----------------------------------------------------------------------
REM 0) Change to BuddyQ root folder so core\main.py exists
REM ----------------------------------------------------------------------
cd /d T:\BuddyQ

REM ----------------------------------------------------------------------
REM 1) Choose ONE Python interpreter for everything (STT + main)
REM    Here we use T:\pytorch\python.exe
REM ----------------------------------------------------------------------
set "PYTHON_EXE=T:\pytorch\python.exe"

REM Quick sanity check
"%PYTHON_EXE%" -c "import sys; print('Using Python:', sys.executable)" 2>nul
if errorlevel 1 (
    echo ERROR: Could not run %PYTHON_EXE%
    echo Please verify that T:\pytorch\python.exe exists.
    pause
    exit /b 1
)

REM ----------------------------------------------------------------------
REM 2) Ensure required packages are installed in THIS Python
REM    - faster-whisper, sounddevice, numpy  (STT)
REM    - mss, pillow                         (screen / vision)
REM ----------------------------------------------------------------------
"%PYTHON_EXE%" -c "import faster_whisper, sounddevice, numpy, mss, PIL" 2>nul
if errorlevel 1 (
    echo Missing one or more required packages - installing now...
    "%PYTHON_EXE%" -m pip install faster-whisper sounddevice numpy mss pillow
    echo.
)

echo [1/3] Environment checked (required Python packages available)
echo.

REM ----------------------------------------------------------------------
REM 3) Set environment variables
REM ----------------------------------------------------------------------
set "HF_HOME=T:\BuddyQ\stt\model_cache"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONPATH=T:\BuddyQ\core"

echo [2/3] Configuration set
echo.

echo [3/3] Starting main.py...
echo.

REM ----------------------------------------------------------------------
REM 4) Run main.py using the chosen Python
REM ----------------------------------------------------------------------
"%PYTHON_EXE%" core\main.py

if errorlevel 1 (
    echo.
    echo ================================================================================
    echo ERROR: main.py exited with an error
    echo ================================================================================
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo BuddyQ Voice Assistant - Stopped
echo ================================================================================
pause
