<#
Build the Windows installer: PyInstaller bundle -> self-check -> Inno Setup.

    pwsh packaging/build_windows.ps1 -Version 1.0.0

Needs uv on PATH. Inno Setup 6 is installed through Chocolatey if missing.
Output: dist\installer\PETCTApp-Setup-<Version>.exe
#>
param(
    [string]$Version = "0.0.0-dev"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== Installing locked dependencies"
uv sync --locked --no-dev --group build
if ($LASTEXITCODE) { throw "uv sync failed" }

Write-Host "== Building app bundle (PyInstaller)"
uv run --no-sync pyinstaller packaging/petct_app.spec --noconfirm --clean
if ($LASTEXITCODE) { throw "PyInstaller failed" }

Write-Host "== Self-check of the bundle"
# The exe is windowed, so its output goes to the app log rather than this console.
$exe = Join-Path $Root "dist\PETCTApp\PETCTApp.exe"
$proc = Start-Process -FilePath $exe -ArgumentList "--selfcheck" -Wait -PassThru
$log = Join-Path $env:LOCALAPPDATA "PETCTApp\logs\app.log"
if (Test-Path $log) { Get-Content $log }
if ($proc.ExitCode -ne 0) { throw "Self-check failed (exit code $($proc.ExitCode))" }

Write-Host "== Building installer (Inno Setup)"
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) {
    choco install innosetup -y --no-progress
    if ($LASTEXITCODE) { throw "Installing Inno Setup failed" }
    $iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
}
& $iscc "/DAppVersion=$Version" "packaging\installer.iss"
if ($LASTEXITCODE) { throw "Inno Setup failed" }

Get-ChildItem dist\installer\*.exe | ForEach-Object {
    Write-Host ("Built {0} ({1:N0} MB)" -f $_.FullName, ($_.Length / 1MB))
}
