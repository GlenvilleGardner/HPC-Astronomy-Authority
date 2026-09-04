# HPC Astronomy Authority - canonical local Windows launcher.
#
# Invokes the repository virtual-environment interpreter by explicit path.
# It does not rely on an activated shell, PATH, or bare `python`, any of
# which could resolve to a different scientific runtime.
#
# This launcher records the canonical local startup path. It is NOT the
# scientific enforcement boundary: server.py performs the authoritative
# fail-closed A0.3 runtime certification before astronomy_solver is loaded.
#
# Arguments are intentionally not forwarded. The canonical local launch is
# fixed and auditable. Production/EC2 process configuration is governed
# separately.

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $python)) {
    throw "Repository virtual environment interpreter not found: $python"
}

& $python -m uvicorn server:app --host 127.0.0.1 --port 8000

exit $LASTEXITCODE
