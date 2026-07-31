$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = "D:\Tools\Miniconda3\envs\qgad\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到 Python 解释器：$python"
}

& $python (Join-Path $repoRoot "desktop\main.py")
