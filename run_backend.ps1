$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "未找到虚拟环境 Python：$python"
}

$existing = Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "后端已经运行：http://127.0.0.1:8001"
    exit 0
}

Write-Host "正在启动后端：http://127.0.0.1:8001"
& $python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
