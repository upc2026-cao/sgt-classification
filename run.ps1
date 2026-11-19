# 一键运行：先建缓存（如已存在会跳过），再训练 baseline
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot ".mplcache"

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "未找到虚拟环境，请先创建 .venv" }

& $py "src\prepare_data.py"
& $py "src\train.py" @args
