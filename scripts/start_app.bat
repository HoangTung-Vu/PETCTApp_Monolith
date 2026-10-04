@echo off
setlocal EnableExtensions

set "SCRIPTS_DIR=%~dp0"
cd /d "%SCRIPTS_DIR%.."
set "PROJECT_DIR=%CD%"

:: start_app.vbs launches this hidden (window style 0), so nothing printed here is
:: ever visible. Everything goes to a log next to the script instead.
set "LOG=%SCRIPTS_DIR%start_app.log"
> "%LOG%" echo [%DATE% %TIME%] start_app.bat - %PROJECT_DIR%

:: eol=# skips comment lines; the %%b test skips blank/valueless lines.
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%b"=="" set "%%a=%%b"
    )
)

if not defined NUMBER_OF_PROCESSORS set "NUMBER_OF_PROCESSORS=4"
set "OMP_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "OPENBLAS_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "MKL_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "NUMEXPR_NUM_THREADS=%NUMBER_OF_PROCESSORS%"
set "QT_OPENGL=desktop"

:: -- Interpreter resolution ------------------------------------------------
:: Prefer the project venv: on a deploy machine it already exists (created by
:: `uv sync`) and needs neither uv on PATH nor a dependency re-resolve at launch.
if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
    set "PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
    goto :run_venv
)

:: No venv - fall back to uv, which the user PATH often doesn't carry when the
:: app is started from a shortcut, so probe the standard install locations too.
call :find_uv
if defined UV goto :run_uv
goto :fail

:run_venv
>>"%LOG%" echo Using venv: %PY%
start /wait /high /b "" "%PY%" -m src.main >>"%LOG%" 2>&1
goto :done

:run_uv
>>"%LOG%" echo Using uv: %UV%
start /wait /high /b "" "%UV%" run python -m src.main >>"%LOG%" 2>&1
goto :done

:done
set "RC=%ERRORLEVEL%"
>>"%LOG%" echo [%DATE% %TIME%] exited with %RC%
exit /b %RC%

:: =================================
:: Helper Functions
:: =================================

:find_uv
:: PATH first, then the locations the common installers use.
for %%p in (uv.exe) do set "UV=%%~$PATH:p"
if defined UV exit /b
for %%c in (
    "%USERPROFILE%\.local\bin\uv.exe"
    "%LOCALAPPDATA%\uv\bin\uv.exe"
    "%LOCALAPPDATA%\Microsoft\WinGet\Links\uv.exe"
    "%USERPROFILE%\scoop\shims\uv.exe"
    "%USERPROFILE%\.cargo\bin\uv.exe"
    "%APPDATA%\Python\Scripts\uv.exe"
) do (
    if exist "%%~c" (
        set "UV=%%~c"
        exit /b
    )
)
exit /b

:fail
>>"%LOG%" echo [ERROR] No .venv\Scripts\python.exe and uv.exe not found on PATH or in the standard install locations.
>>"%LOG%" echo Fix - run these in PowerShell from the project directory:
>>"%LOG%" echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
>>"%LOG%" echo   uv sync
>>"%LOG%" echo Then start the app again. Current search paths:
>>"%LOG%" set PATH
mshta "javascript:new ActiveXObject('WScript.Shell').Popup('Khong khoi dong duoc PET/CT App.\n\nKhong tim thay .venv va cung khong tim thay uv tren may nay.\n\nMo file scripts\\start_app.log de xem cach cai dat.', 0, 'PET/CT App', 16); close();"
exit /b 1
