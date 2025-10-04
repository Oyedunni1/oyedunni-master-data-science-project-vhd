@echo off
echo ================================================
echo VHD Prediction Project - Windows Setup
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python detected
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv vhd_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call vhd_env\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo ✓ Pip upgraded
echo.

REM Install Windows-specific requirements
echo Installing Windows-specific requirements...
pip install -r requirements_windows.txt
if errorlevel 1 (
    echo WARNING: Some packages failed to install
    echo Trying alternative installation...
    pip install numpy pandas scipy librosa scikit-learn tensorflow matplotlib seaborn streamlit plotly opencv-python soundfile wfdb numba joblib tqdm requests urllib3
)

echo ✓ Dependencies installed
echo.

REM Create necessary directories
echo Creating project directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "models" mkdir models

echo ✓ Directories created
echo.

REM Run Windows setup script
echo Running Windows compatibility setup...
python windows_setup.py
if errorlevel 1 (
    echo WARNING: Windows setup script encountered issues
    echo The project should still work with fallback options
)

echo.
echo ================================================
echo Setup Complete!
echo ================================================
echo.
echo To use the project:
echo 1. Activate environment: vhd_env\Scripts\activate
echo 2. Train model: python train_model.py
echo 3. Run app: streamlit run app.py
echo.
echo Note: The system will automatically handle Windows compatibility issues
echo by falling back to random initialization if ImageNet weights fail.
echo.
pause
