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

:: [1] Configuration
if "%ENGINE_NNUNET_PORT%"=="" set "ENGINE_NNUNET_PORT=8104"
if "%ENGINE_NNUNET_IMAGE%"=="" set "ENGINE_NNUNET_IMAGE=nnunet-engine"
if "%ENGINE_NNUNET_CONTAINER%"=="" set "ENGINE_NNUNET_CONTAINER=nnunet-engine-container"

:: Detect logical CPU core count for numpy/torch thread tuning
for /f "tokens=2 delims==" %%a in ('wmic cpu get NumberOfLogicalProcessors /value 2^>nul') do (
    if not "%%a"=="" set "CPU_CORES=%%a"
)
if "%CPU_CORES%"=="" set "CPU_CORES=4"

set "GPU_FLAG="
nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    echo NVIDIA GPU detected. Enabling GPU passthrough.
    set "GPU_FLAG=--gpus all"
    set "CUDA_VISIBLE_DEVICES=0"
) else (
    echo No NVIDIA GPU detected. Running CPU mode.
)

echo =====================================
echo   PET/CT AI Engine Launcher
echo =====================================

::echo.
::echo -- Step 1: Building Docker image --
::echo Building Docker image: %ENGINE_NNUNET_IMAGE% ...
::docker build -t "%ENGINE_NNUNET_IMAGE%" "AI_engines\engine_nnunet_old_ver"

echo.
echo -- Step 2: Starting container --
:: NOTE: On Windows the Docker daemon runs in WSL2. We must keep -p port mapping
:: so the Windows-side Python GUI can reach the container via localhost; --network host
:: would expose ports only inside the WSL2 VM and break the Windows-host connection.
:: (If you've enabled WSL2 mirrored networking in .wslconfig you can switch to --network host.)
call :start_container %ENGINE_NNUNET_CONTAINER% %ENGINE_NNUNET_IMAGE% %ENGINE_NNUNET_PORT%

echo.
echo -- Step 3: Health check (waits for model preload) --
call :wait_for_health %ENGINE_NNUNET_PORT% "nnUNet Engine"

echo.
echo =====================================
echo   Engine is running on port %ENGINE_NNUNET_PORT%.
echo   Press any key to stop the engine.
echo =====================================
pause >nul

echo.
echo -- Cleanup --
echo Stopping engine container...
docker stop %ENGINE_NNUNET_CONTAINER% >nul 2>&1
docker rm %ENGINE_NNUNET_CONTAINER% >nul 2>&1
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

echo Starting container: !c_name! (max resources, port !p_port!, %CPU_CORES% cores)
:: --ipc=host           : share host /dev/shm (no cap on PyTorch shared memory).
:: --ulimit memlock=-1  : unlimited pinned memory for CUDA DMA.
:: --ulimit stack=64MB  : larger stack for torch ops.
:: -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True : reduces VRAM fragmentation.
:: -e PYTHONUNBUFFERED=1 : flush prints (useful when tailing logs).
:: -e OMP/MKL/...      : let CPU ops in container use all host cores.
docker run -d --name "!c_name!" %GPU_FLAG% ^
    --ipc=host ^
    --ulimit memlock=-1:-1 ^
    --ulimit stack=67108864 ^
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ^
    -e PYTHONUNBUFFERED=1 ^
    -e OMP_NUM_THREADS=%CPU_CORES% ^
    -e MKL_NUM_THREADS=%CPU_CORES% ^
    -e OPENBLAS_NUM_THREADS=%CPU_CORES% ^
    -e NUMEXPR_NUM_THREADS=%CPU_CORES% ^
    -p "!p_port!:!p_port!" "!i_name!"
exit /b

:wait_for_health
set "h_port=%~1"
set "h_name=%~2"
:: Startup preloads model weights to VRAM; bump timeout from 60s to 180s.
set /a max_wait=180
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
