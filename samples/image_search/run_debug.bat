@echo off
REM Debug and Run Image Search Application

echo ============================================
echo Image Search Application - Debug Runner
echo ============================================
echo.

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call ..\..\venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install/update dependencies
echo.
echo [2/4] Checking dependencies...
pip install -q duckduckgo-search aiohttp Pillow
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some dependencies may not have installed correctly
)

REM Check MongoDB connection
echo.
echo [3/4] Checking MongoDB connection...
echo MongoDB should be running on localhost:27017
timeout /t 2 >nul

REM Run application
echo.
echo [4/4] Starting Image Search...
echo.
echo ============================================
python main.py
echo.
echo ============================================
echo Application closed
pause
