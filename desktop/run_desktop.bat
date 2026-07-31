@echo off
set PYTHON_EXE=D:\Tools\Miniconda3\envs\qgad\python.exe
if not exist "%PYTHON_EXE%" (
  echo Python interpreter not found: %PYTHON_EXE%
  exit /b 1
)
"%PYTHON_EXE%" "%~dp0main.py"
