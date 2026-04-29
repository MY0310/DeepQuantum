param(
    [string]$Executable,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$appRoot = Join-Path $PSScriptRoot "Deepquantum"
if (-not (Test-Path -LiteralPath $appRoot -PathType Container)) {
    Write-Error "App root not found: $appRoot"
    exit 1
}

if ([string]::IsNullOrWhiteSpace($Executable)) {
    Write-Host "App root: $appRoot"
    Write-Host "Usage: .\\dq.ps1 <executable> [args...]"
    Write-Host "Example: .\\dq.ps1 python run_elliptic.py --help"
    exit 0
}

Push-Location $appRoot
try {
    if (-not $Arguments -or $Arguments.Count -eq 0) {
        & $Executable
    } else {
        & $Executable @Arguments
    }
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) { $exitCode = 0 }
    exit $exitCode
}
finally {
    Pop-Location
}
