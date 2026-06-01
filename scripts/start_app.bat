@echo off
set "SCRIPTS_DIR=%~dp0"
cd /d "%SCRIPTS_DIR%.."

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
)

set "OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "OPENBLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "NUMEXPR_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "QT_OPENGL=desktop"

start /wait /high /b "" uv run python -m src.main
exit /b
