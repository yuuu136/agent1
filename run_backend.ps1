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

$localIps = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -ExpandProperty IPAddress

Write-Host "正在启动 Agent：http://127.0.0.1:8001"
foreach ($ip in $localIps) {
    Write-Host "局域网访问地址：http://$ip`:8001"
}
& $python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
