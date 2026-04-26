$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "[1/3] Running Phase 1 baseline..."
python .\src\phase1\train.py

Write-Host "[2/3] Running Phase 2 model comparison..."
python .\src\phase2\train_compare.py

Write-Host "[3/3] Running tests..."
python -m pytest .\tests\phase2 -q

Write-Host "All non-API workflows completed successfully."
