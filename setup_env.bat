@echo off

echo Starting environment setup...

REM 1. Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: python could not be found. Please install Python 3 and ensure it is in your PATH.
    exit /b 1
)

REM 2. Create the virtual environment named '.venv'
if not exist ".venv\" (
    echo Creating virtual environment '.venv'...
    python -m venv .venv
) else (
    echo Virtual environment '.venv' already exists.
)

REM 3. Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM 4. Upgrade pip to the latest version
echo Upgrading pip...
python -m pip install --upgrade pip

REM 5. Install dependencies from requirements.txt
if exist "requirements.txt" (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo Error: requirements.txt not found!
    exit /b 1
)

echo Setup complete! You are ready to go.
echo To activate the environment manually in the future, run: .venv\Scripts\activate
