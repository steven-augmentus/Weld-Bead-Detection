@echo off
REM Create and activate a Python 3.11 virtual environment, then install packages

REM Check if Python 3.11 is available
python --version | findstr "3.11" >nul
if errorlevel 1 (
    echo Python 3.11 not found in PATH. Please install it or add it to PATH.
    pause
    exit /b 1
)

REM Create virtual environment
python -m venv venv

REM Activate the environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install packages
pip install open3d numpy scipy matplotlib

echo.
echo Setup complete. Virtual environment 'venv' is active.

python waypoint_estimation.py

pause
