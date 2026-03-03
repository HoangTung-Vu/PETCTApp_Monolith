@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
        set "%%a=%%b"
    )
) else (
    echo Warning: .env file not found, using defaults.
)

:: [1] Configuration Variables
if "%ENGINE_NNUNET_PORT%"=="" set "ENGINE_NNUNET_PORT=8101"
if "%ENGINE_NNUNET_OLD_PORT%"=="" set "ENGINE_NNUNET_OLD_PORT=8104"
if "%ENGINE_AUTOPET_PORT%"=="" set "ENGINE_AUTOPET_PORT=8102"
if "%ENGINE_TOTALSEG_PORT%"=="" set "ENGINE_TOTALSEG_PORT=8103"

if "%ENGINE_NNUNET_IMAGE%"=="" set "ENGINE_NNUNET_IMAGE=engine-nnunet"
if "%ENGINE_NNUNET_OLD_IMAGE%"=="" set "ENGINE_NNUNET_OLD_IMAGE=engine-nnunet-old-ver"
if "%ENGINE_AUTOPET_IMAGE%"=="" set "ENGINE_AUTOPET_IMAGE=engine-autopet"
if "%ENGINE_TOTALSEG_IMAGE%"=="" set "ENGINE_TOTALSEG_IMAGE=engine-totalseg"

if "%ENGINE_NNUNET_CONTAINER%"=="" set "ENGINE_NNUNET_CONTAINER=engine-nnunet-container"
if "%ENGINE_NNUNET_OLD_CONTAINER%"=="" set "ENGINE_NNUNET_OLD_CONTAINER=engine-nnunet-old-ver-container"
if "%ENGINE_AUTOPET_CONTAINER%"=="" set "ENGINE_AUTOPET_CONTAINER=engine-autopet-container"
if "%ENGINE_TOTALSEG_CONTAINER%"=="" set "ENGINE_TOTALSEG_CONTAINER=engine-totalseg-container"

set "GPU_FLAG="
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo NVIDIA GPU detected. Enabling GPU passthrough.
    set "GPU_FLAG=--gpus all"
) else (
    echo No NVIDIA GPU detected. Running CPU mode ^(see README_SETUP.md for GPU^).
)

echo =====================================
echo   PET/CT Segmentation App Launcher (Windows)
echo =====================================

echo.
echo -- Step 1: Building Docker images --
echo Building Docker image: %ENGINE_NNUNET_IMAGE% ...
docker build -t "%ENGINE_NNUNET_IMAGE%" "%SCRIPT_DIR%engine_nnunet"
echo Building Docker image: %ENGINE_NNUNET_OLD_IMAGE% ...
docker build -t "%ENGINE_NNUNET_OLD_IMAGE%" "%SCRIPT_DIR%engine_nnunet_old_ver"
echo Building Docker image: %ENGINE_AUTOPET_IMAGE% ...
docker build -t "%ENGINE_AUTOPET_IMAGE%" "%SCRIPT_DIR%engine_autopet"
echo Building Docker image: %ENGINE_TOTALSEG_IMAGE% ...
docker build -t "%ENGINE_TOTALSEG_IMAGE%" "%SCRIPT_DIR%engine_totalseg"

echo.
echo -- Step 2: Starting containers --
call :start_container %ENGINE_NNUNET_CONTAINER% %ENGINE_NNUNET_IMAGE% %ENGINE_NNUNET_PORT%
call :start_container %ENGINE_NNUNET_OLD_CONTAINER% %ENGINE_NNUNET_OLD_IMAGE% %ENGINE_NNUNET_OLD_PORT%
call :start_container %ENGINE_AUTOPET_CONTAINER% %ENGINE_AUTOPET_IMAGE% %ENGINE_AUTOPET_PORT%
call :start_container %ENGINE_TOTALSEG_CONTAINER% %ENGINE_TOTALSEG_IMAGE% %ENGINE_TOTALSEG_PORT%

echo.
echo -- Step 3: Health checks --
call :wait_for_health %ENGINE_NNUNET_PORT% "nnUNet Engine"
call :wait_for_health %ENGINE_NNUNET_OLD_PORT% "nnUNet Old Engine"
call :wait_for_health %ENGINE_AUTOPET_PORT% "AutoPET Engine"
call :wait_for_health %ENGINE_TOTALSEG_PORT% "TotalSeg Engine"

echo.
echo -- Step 4: Launch PyQt GUI --
cd /d "%SCRIPT_DIR%"
uv run python -m src.main

echo.
echo -- Cleanup --
echo Stopping engine containers...
docker stop %ENGINE_NNUNET_CONTAINER% %ENGINE_NNUNET_OLD_CONTAINER% %ENGINE_AUTOPET_CONTAINER% %ENGINE_TOTALSEG_CONTAINER% >nul 2>&1
docker rm %ENGINE_NNUNET_CONTAINER% %ENGINE_NNUNET_OLD_CONTAINER% %ENGINE_AUTOPET_CONTAINER% %ENGINE_TOTALSEG_CONTAINER% >nul 2>&1
echo Done.

exit /b

:: =================================
:: Helper Functions
:: =================================

:start_container
set "c_name=%~1"
set "i_name=%~2"
set "p_port=%~3"

docker ps -a --format "{{.Names}}" | findstr /b /e "%c_name%" >nul
if !errorlevel! equ 0 (
    docker stop "%c_name%" >nul 2>&1
    docker rm "%c_name%" >nul 2>&1
)

echo Starting container: !c_name! (port !p_port!)
if "%GPU_FLAG%"=="" (
    docker run -d --name "!c_name!" -p "!p_port!:!p_port!" "!i_name!"
) else (
    docker run -d --name "!c_name!" %GPU_FLAG% -p "!p_port!:!p_port!" "!i_name!"
)
exit /b

:wait_for_health
set "h_port=%~1"
set "h_name=%~2"
set /a max_wait=60
set /a waited=0

<nul set /p "=Waiting for !h_name! (port !h_port!) "

:health_loop
curl -sf http://localhost:!h_port!/health >nul 2>&1
if !errorlevel! equ 0 (
    echo  OKAY
    exit /b
)
if !waited! geq !max_wait! (
    echo  TIMEOUT
    exit /b
)

<nul set /p "=."
:: wait 2 seconds
ping 127.0.0.1 -n 3 >nul 2>&1
set /a waited+=2
goto health_loop
