$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$PythonExe = "D:\Tools\Miniconda3\envs\qgad\python.exe"
$SpecPath = Join-Path $PSScriptRoot "qgad_desktop.spec"
$DistPath = Join-Path $ProjectRoot "dist_release"
$WorkPath = Join-Path $ProjectRoot "build_release"

Set-Location $ProjectRoot

& $PythonExe -m PyInstaller --noconfirm --distpath $DistPath --workpath $WorkPath $SpecPath

Write-Host ""
Write-Host "Build complete:" -ForegroundColor Green
Write-Host "  dist_release\QGADDesktop\QGADDesktop.exe"
