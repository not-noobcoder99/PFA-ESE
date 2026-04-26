$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
