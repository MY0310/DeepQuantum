@echo off
REM Installation script for Q-GAD conda environment (Windows)
REM Usage: scripts\setup_env.bat

echo ==================================
echo Q-GAD Environment Setup (Windows)
echo ==================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: conda not found. Please install Anaconda or Miniconda first.
    echo   Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Step 1: Creating conda environment...
echo ==================================

REM Ask user for environment type
set USE_CUDA=n
echo Choose environment type:
echo   1 - CPU only (recommended for most users)
echo   2 - CUDA enabled (requires NVIDIA GPU and CUDA)
set /p CHOICE="Enter choice (1 or 2): "

if "%CHOICE%"=="2" (
    set ENV_FILE=environment-cuda.yml
    set ENV_NAME=qgad-cuda
    echo Will use CUDA environment
) else (
    set ENV_FILE=environment.yml
    set ENV_NAME=qgad
    echo Will use CPU environment
)

echo.
echo Creating environment from %ENV_FILE%...

conda env create -f %ENV_FILE%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Environment creation failed, trying update...
    conda env update -f %ENV_FILE% --prune
)

echo.
echo Step 2: Verifying installations...
echo ==================================

REM Activate and check
call conda activate %ENV_NAME%

echo Checking installations...
echo   Python:
python --version

echo   PyTorch:
python -c "import torch; print(torch.__version__)"

echo   NumPy:
python -c "import numpy; print(numpy.__version__)"

echo   NetworkX:
python -c "import networkx; print(networkx.__version__)"

echo.
echo Step 3: Checking DeepQuantum...
echo ==================================

python -c "import deepquantum" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo   [OK] DeepQuantum is installed
    echo   Testing DeepQuantum...
    python scripts\verify_install.py
) else (
    echo   [WARN] DeepQuantum not installed
    echo   This is OK - system will use mock implementation
    echo   To install: pip install git+https://github.com/turingq/deepquantum.git
)

echo.
echo Step 4: Creating project directories...
echo ==================================

if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\cache" mkdir "data\cache"
if not exist "outputs" mkdir "outputs"
if not exist "logs" mkdir "logs"
if not exist "checkpoints" mkdir "checkpoints"

echo   [OK] Directories created

echo.
echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo To activate the environment, run:
echo   conda activate %ENV_NAME%
echo.
echo To test the installation, run:
echo   python scripts\verify_install.py
echo.
echo Recommended Elliptic++ training entrypoints:
echo   python run_elliptic_fast.py
echo   python run_elliptic.py
echo.
pause
