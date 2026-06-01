@echo off
setlocal EnableDelayedExpansion

set "SCRIPTS_DIR=%~dp0"
cd /d "%SCRIPTS_DIR%.."

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo Warning: .env file not found, using defaults.
)

:: Detect logical CPU core count for numpy/torch thread tuning
for /f "tokens=2 delims==" %%a in ('wmic cpu get NumberOfLogicalProcessors /value 2^>nul') do (
    if not "%%a"=="" set "CPU_CORES=%%a"
)
if "%CPU_CORES%"=="" set "CPU_CORES=4"

:: Give numpy/OpenBLAS/MKL full access to all cores (host-side)
set "OMP_NUM_THREADS=%CPU_CORES%"
set "OPENBLAS_NUM_THREADS=%CPU_CORES%"
set "MKL_NUM_THREADS=%CPU_CORES%"
set "NUMEXPR_NUM_THREADS=%CPU_CORES%"

set "QT_OPENGL=desktop"
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo NVIDIA GPU detected. Enabling discrete GPU rendering.
    set "CUDA_VISIBLE_DEVICES=0"
) else (
    echo No NVIDIA GPU detected. Running CPU mode.
)

echo =====================================
echo   PET/CT App Launcher
echo =====================================

echo.
echo -- Launching PyQt GUI (high priority, %CPU_CORES% cores) --
start /wait /high /b "" uv run python -m src.main

echo.
echo App closed.
exit /b
